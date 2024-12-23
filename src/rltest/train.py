import tensordict
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, CompositeDistribution
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.envs import PettingZooWrapper
from torchrl.modules import ProbabilisticActor, MultiAgentMLP
import torch.distributions as d

from rltest.env import Environment

env = PettingZooWrapper(
    env=Environment(),
    categorical_actions=False,
)

class DirichletExtractor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.linear = nn.Linear(8, 2)

    def forward(self, latent: torch.Tensor):
        return self.linear(latent)

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
            ("params", "strength", "concentration0"): params[..., :2],
            ("params", "strength", "concentration1"): params[..., 2:],
        }

beta_extractor = TensorDictModule(
    module=BetaExtractor(),
    in_keys=("agent", "latent"),
    out_keys=[
        ("params", "strength", "concentration0"),
        ("params", "strength", "concentration1"),
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
)

class SummingCompositeDistribution(CompositeDistribution):
    def log_prob(self, sample: TensorDictBase, *, aggregate_probabilities: bool | None = None,
                 **kwargs) -> torch.Tensor | TensorDictBase:
        prob_out = self.log_prob_composite(sample, include_sum=True)
        log_prob = prob_out.select(("agent", "sample_log_prob"))
        #out = log_prob.get(("agent", "sample_log_prob"))
        return log_prob

    def log_prob_composite(self, sample: TensorDictBase, include_sum=True, **kwargs) -> TensorDictBase:
        slp = torch.zeros([2])
        d = {}
        for name, dist in self.dists.items():
            lp = dist.log_prob(sample.get(name))
            lp = lp.sum(dim=-1) if len(lp.shape) == 3 else lp
            slp = slp + lp
        d["agent"] = {
            self.log_prob_key: slp
        }
        sample.update(d)
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
        },
        "name_map": {
            "target": ("agent", "action", "target"),
            "strength": ("agent", "action", "strength"),
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

for i, data in enumerate(collector):
    ...