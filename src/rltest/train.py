import contextlib
from typing import Tuple, Dict

import tensordict
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, CompositeDistribution, set_composite_lp_aggregate
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import PettingZooWrapper, TransformedEnv, RewardSum
from torchrl.envs.libs.pettingzoo import _extract_nested_with_index
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
import torch.distributions as d
import torch.nn.functional as f
from torchrl.objectives import ClipPPOLoss, ValueEstimators, PPOLoss
from torchrl.objectives.utils import _sum_td_features
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
            ("params", "strength", "concentration0"): f.softplus(params[..., :2]),
            ("params", "strength", "concentration1"): f.softplus(params[..., 2:]),
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
            ("params", "healing", "concentration"): f.softplus(params[..., :2]),
            ("params", "healing", "rate"): f.softplus(params[..., 2:]),
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

loss_module = PPOLoss(
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