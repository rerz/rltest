import contextlib
import warnings
from typing import Tuple, Dict

import tensordict
import torch
from tensordict import TensorDict, TensorDictBase, is_tensor_collection
from tensordict.nn import TensorDictModule, CompositeDistribution, set_composite_lp_aggregate, dispatch
from torch import nn
from torchrl._utils import _standardize
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import PettingZooWrapper, TransformedEnv, RewardSum
from torchrl.envs.libs.pettingzoo import _extract_nested_with_index
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
import torch.distributions as d
import torch.nn.functional as f
from torchrl.objectives import ClipPPOLoss, ValueEstimators, PPOLoss
from torchrl.objectives.utils import _sum_td_features, _reduce
from tqdm import tqdm

from rltest.env import Environment, PettingZooDictWrapper

set_composite_lp_aggregate(False).set()

class PettingZooWrapperWrapper(PettingZooWrapper):
    def _step_parallel(
        self,
        tensordict: TensorDictBase,
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        action_dict = {}
        for group, agents in self.group_map.items():
            group_action = tensordict.get((group, "action"))
            group_action_spec = (self.input_spec[
                "full_action_spec", group, "action"
            ])
            valid_keys = [(key,) for key in list(group_action_spec.keys())]
            group_action_filtered = group_action.select(*valid_keys)
            group_action_np = group_action_spec.to_numpy(group_action_filtered)
            for index, agent in enumerate(agents):
                # group_action_np can be a dict or an array. We need to recursively index it
                action = _extract_nested_with_index(group_action_np, index)
                action_dict[agent] = action

        return self._env.step(action_dict)

env = PettingZooWrapperWrapper(
    env=Environment(),
    categorical_actions=False,
)

env = TransformedEnv(
    env, RewardSum(in_keys=[env.reward_key], out_keys=[("agent", "episode_reward",)])
)

class DirichletExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear = nn.Linear(8, 2)

    def forward(self, latent: torch.Tensor):
        return f.softplus(self.linear(latent))

dirichlet_extractor = TensorDictModule(
    module=DirichletExtractor(),
    in_keys=("agent", "latent"),
    out_keys=[("params", "target", "concentration")],
)

class BetaExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear = nn.Linear(8, 4)

    def forward(self, latent: torch.Tensor):
        params = self.linear(latent)
        return {
            ("params", "strength", "concentration0"): f.softplus(params[..., 0:2]),
            ("params", "strength", "concentration1"): f.softplus(params[..., 2:4]),
        }

beta_extractor = TensorDictModule(
    module=BetaExtractor(),
    in_keys=("agent", "latent"),
    out_keys=[
        ("params", "strength", "concentration0"),
        ("params", "strength", "concentration1"),
    ],
)

class GammaExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear = nn.Linear(8, 4)

    def forward(self, latent: torch.Tensor):
        params = self.linear(latent)
        return {
            ("params", "healing", "concentration"): f.softplus(params[..., 0:2]),
            ("params", "healing", "rate"): f.softplus(params[..., 2:4]),
        }

gamma_extractor = TensorDictModule(
    module=GammaExtractor(),
    in_keys=("agent", "latent"),
    out_keys=[
        ("params", "healing", "concentration"),
        ("params", "healing", "rate"),
    ],
)

policy = TensorDictModule(
    module=MultiAgentMLP(
        n_agent_inputs=2,
        n_agent_outputs=8,
        n_agents=2,
        depth=2,
        centralized=False,
        share_params=False,
    ),
    in_keys=[
        ("agent", "observation", "rocks"),
    ],
    out_keys=[
        ("agent", "latent")
    ]
)

actor = tensordict.nn.TensorDictSequential(
    policy,
    dirichlet_extractor,
    beta_extractor,
    gamma_extractor,
)

actor = ProbabilisticActor(
    module=actor,
    in_keys="params",
    out_keys=[
        ("agent", "action", "target"),
        ("agent", "action", "strength"),
        ("agent", "action", "healing"),
    ],
    distribution_class=CompositeDistribution,
    distribution_kwargs={
        "distribution_map": {
            "target": d.Dirichlet,
            "strength": d.Beta,
            "healing": d.Gamma,
        },
        "name_map": {
            "target": ("agent", "action", "target"),
            "strength": ("agent", "action", "strength"),
            "healing": ("agent", "action", "healing"),
        }
    },
    return_log_prob=True,
    #log_prob_key=("agent", "sample_log_prob"),
)

class CustomPPOLoss(ClipPPOLoss):
    def __init__(self, *args, **kwargs):
        ClipPPOLoss.__init__(self, *args, **kwargs)

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            if advantage.numel() > tensordict.batch_size.numel() and not len(
                self.normalize_advantage_exclude_dims
            ):
                warnings.warn(
                    "You requested advantage normalization and the advantage key has more dimensions"
                    " than the tensordict batch. Make sure to pass `normalize_advantage_exclude_dims` "
                    "if you want to keep any dimension independent while computing normalization statistics. "
                    "If you are working in multi-agent/multi-objective settings this is highly suggested."
                )
            advantage = _standardize(advantage, self.normalize_advantage_exclude_dims)

        log_weight, dist, kl_approx = self._log_weight(tensordict)
        if is_tensor_collection(log_weight):
            log_weight = log_weight.update({
                ("agent", "action", "strength_log_prob"): log_weight[("agent", "action", "strength_log_prob")].sum(dim=-1, keepdim=False),
                ("agent", "action", "healing_log_prob"): log_weight[("agent", "action", "healing_log_prob")].sum(dim=-1, keepdim=False),
            })
            log_weight = [weight for weight in log_weight.values(include_nested=True) if isinstance(weight, torch.Tensor)]
            log_weight = torch.sum(torch.stack(log_weight), dim=0, keepdim=False)
            log_weight = log_weight.view(advantage.shape)
        neg_loss = log_weight.exp() * advantage
        td_out = TensorDict({"loss_objective": -neg_loss}, batch_size=[])
        td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
        if self.entropy_bonus:
            entropy = self._get_entropy(dist)
            if is_tensor_collection(entropy):
                # Reports the entropy of each action head.
                td_out.set("composite_entropy", entropy.detach())
                entropy = _sum_td_features(entropy)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy)
        if self.critic_coef is not None:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

collector = SyncDataCollector(
    create_env_fn=env,
    policy=actor,
    frames_per_batch=10,
    total_frames=100,
)

replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(1000),
    sampler=SamplerWithoutReplacement(),
    batch_size=10,
)

critic = MultiAgentMLP(
    n_agent_inputs=2,
    n_agent_outputs=1,
    n_agents=2,
    centralized=True,
    share_params=True,
    depth=2,
    num_cells=256,
    activation_class=nn.Tanh,
)

critic = TensorDictModule(
    module=critic,
    in_keys=("agent", "observation", "rocks"),
    out_keys=("agent", "state_value")
)

loss_module = CustomPPOLoss(
    actor_network=actor,
    critic_network=critic,
    entropy_coef=0.1,
    normalize_advantage=False,
)

loss_module.set_keys(
    reward=env.reward_key,
    action=("agent", "action"),
    value=("agent", "state_value"),
    done=("agent", "done"),
    terminated=("agent", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE,
    gamma=0.001,
    lmbda=0.001,
)

GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr=1e-3)

pbar = tqdm(total=10, desc="episode_reward_mean = 0")

episode_reward_mean_list = []
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agent", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agent", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )  # Compute GAE and add it to the data

    replay_buffer.extend(tensordict_data)

    for _ in range(10):
        for _ in range(10):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), 2.0
            )  # Optional

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    # Logging
    done = tensordict_data.get(("next", "agent", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agent", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()