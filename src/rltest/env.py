from collections import defaultdict

import numpy as np
import pettingzoo
from pettingzoo.utils.env import AgentID, ActionType, ObsType
import gymnasium
from gymnasium import spaces

class Environment(pettingzoo.ParallelEnv):
    """
    Sample environment where two agents have two rocks in front of them.
    Each agent predicts a probability distribution of which rock to hit and what strength to hit it with.
    First agent to break a rock could win or something like that but doesn't really matter for demo purposes.
    """

    agent_strengths = {
        "agent_0": 100,
        "agent_1": 80,
    }

    rock_hps = [700, 400]

    def __init__(self):
        pettingzoo.ParallelEnv.__init__(self)

        agents = [
            "agent_0",
            "agent_1",
        ]

        self.agents = agents
        self.possible_agents = agents

    def reset(self, **kwargs):
        observations = {
            "agent_0": {
                "rocks": [1.0, 1.0]
            },
            "agent_1": {
                "rocks": [1.0, 1.0]
            }
        }
        return observations, { "agent_0": {}, "agent_1": {} }

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.Dict([
            ("target", spaces.Box(0, 1, [2])), # probability of hitting each rock
            ("strength", spaces.Box(0, 1, [2])), # percentage strength to hit it with
            ("healing", spaces.Box(0, 100, [2])) # each agent can heal rock for an absolute value < 100
        ])

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.Dict([
            ("rocks", spaces.Box(0, 1, [2])), # rock hp as a percentage
        ])

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        damage_dealt = {}

        for agent, agent_actions in actions.items():
            target = agent_actions["target"]
            target = np.argmax(target)

            damage = agent_actions["strength"] * self.agent_strengths[agent]

            damage_dealt[agent] = (target, damage)


        observations = {
            "agent_0": {
                "rocks": [1.0, 1.0],
            },
            "agent_1": {
                "rocks": [1.0, 1.0],
            }
        }
        rewards = {
            "agent_0": 0.0,
            "agent_1": 0.0,
        }
        terminations = {
            "agent_0": False,
            "agent_1": False,
        }
        truncations = {
            "agent_0": False,
            "agent_1": False,
        }
        info = {}

        return observations, rewards, terminations, truncations, info

