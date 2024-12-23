import contextlib
from typing import Tuple

import tensordict
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, CompositeDistribution
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import PettingZooWrapper, TransformedEnv, RewardSum
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
import torch.distributions as d
import torch.nn.functional as f
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from torchrl.objectives.utils import _sum_td_features
from tqdm import tqdm

from rltest.env import Environment, PettingZooDictWrapper

env = PettingZooDictWrapper(
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

class SummingCompositeDistribution(CompositeDistribution):
    def log_prob(self, sample: TensorDictBase, *, aggregate_probabilities: bool | None = None,
                 **kwargs) -> torch.Tensor | TensorDictBase:
        prob_out = self.log_prob_composite(sample, include_sum=True)
        log_prob = prob_out.select(("agent", "sample_log_prob"))
        #out = log_prob.get(("agent", "sample_log_prob"))
        return log_prob

    def log_prob_composite(self, sample: TensorDictBase, include_sum=True, **kwargs) -> TensorDictBase:
        slp = torch.zeros([1])
        out_dict = {}
        for name, dist in self.dists.items():
            lp = dist.log_prob(sample.get(name))

            # TODO: beta and gamma distributions are univariate so they have one dim extra
            if name[-1] == "target":
                slp = slp + lp
            else:
                slp = slp + lp.sum(dim=-1)

        # TODO: only return combined sample_log_prob key here
        out_dict["agent"] = {
            self.log_prob_key: slp
        }
        sample.update(out_dict)
        return sample


actor = ProbabilisticActor(
    module=actor,
    in_keys="params",
    out_keys=("agent", "action"),
    distribution_class=SummingCompositeDistribution,
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
    log_prob_key=("agent", "sample_log_prob"),
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

class CustomPPOLoss(ClipPPOLoss):
    def __init__(self, *args, **kwargs):
        ClipPPOLoss.__init__(self, *args, **kwargs)

    def _log_weight(
        self, tensordict: TensorDictBase
    ) -> Tuple[torch.Tensor, d.Distribution]:
        # current log_prob of actions
        action = tensordict.get(self.tensor_keys.action)

        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            dist = self.actor_network.get_dist(tensordict)

        prev_log_prob = tensordict.get(self.tensor_keys.sample_log_prob)
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.sample_log_prob} requires grad."
            )

        if action.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.action} requires grad."
            )
        if isinstance(action, torch.Tensor):
            log_prob = dist.log_prob(action)
        else:
            if isinstance(dist, CompositeDistribution):
                is_composite = True
                kwargs = {
                    "inplace": False,
                    "aggregate_probabilities": False,
                    "include_sum": False,
                }
            else:
                is_composite = False
                kwargs = {}
            log_prob = dist.log_prob(tensordict, **kwargs)
            if is_composite and not isinstance(prev_log_prob, TensorDict):
                log_prob = _sum_td_features(log_prob)
                log_prob.view_as(prev_log_prob)

        # TODO: is this correct?
        log_prob = log_prob.get(("agent", "sample_log_prob"))
        prev_log_prob = prev_log_prob.get(("agent", "sample_log_prob"))

        log_prob.view_as(prev_log_prob)
        # END

        log_weight = (log_prob - prev_log_prob).unsqueeze(-1)
        kl_approx = (prev_log_prob - log_prob).unsqueeze(-1)

        return log_weight, dist, kl_approx

loss_module = CustomPPOLoss(
    actor_network=actor,
    critic_network=critic,
    entropy_coef=0.1,
    normalize_advantage=False,
)

loss_module.set_keys(
    reward=env.reward_key,
    action=("agent", "action"),
    sample_log_prob=("agent", "sample_log_prob"),
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