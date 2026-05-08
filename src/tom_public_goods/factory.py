from __future__ import annotations

import random

from .agents import (
    BeliefBasedAgent,
    ConditionalAgent,
    CooperatorAgent,
    DefectorAgent,
    LLMAgent,
    RandomAgent,
    TitForTatAgent,
)
from .agents.base import BaseAgent
from .config import GameConfig


def create_agents(config: GameConfig) -> list[BaseAgent]:
    """Create heterogeneous agent populations for the requested condition."""

    def aid(i: int) -> str:
        return f"A{i:03d}"

    config.validate()
    agents: list[BaseAgent] = []
    condition = config.condition.lower().strip()

    if condition == "baseline":
        classes = [CooperatorAgent, DefectorAgent, TitForTatAgent, RandomAgent, ConditionalAgent]
        for i in range(1, config.num_agents + 1):
            cls = classes[(i - 1) % len(classes)]
            agents.append(cls(aid(i), random.Random(config.seed + i)))

    elif condition == "belief_mixed":
        classes = [BeliefBasedAgent, BeliefBasedAgent, ConditionalAgent, TitForTatAgent, DefectorAgent]
        for i in range(1, config.num_agents + 1):
            cls = classes[(i - 1) % len(classes)]
            agents.append(cls(aid(i), random.Random(config.seed + i)))

    elif condition in {"llm_tom_mixed", "llm_notom_mixed"}:
        mode = "tom" if condition == "llm_tom_mixed" else "notom"
        llm_count = config.llm_agents if config.llm_agents > 0 else max(1, min(6, config.num_agents // 5))
        llm_count = min(llm_count, config.num_agents)

        for i in range(1, config.num_agents + 1):
            if i <= llm_count:
                agents.append(LLMAgent(aid(i), random.Random(config.seed + i), mode=mode))
            else:
                rest_index = i - llm_count
                cls_cycle = [BeliefBasedAgent, ConditionalAgent, TitForTatAgent, CooperatorAgent, DefectorAgent, RandomAgent]
                cls = cls_cycle[(rest_index - 1) % len(cls_cycle)]
                agents.append(cls(aid(i), random.Random(config.seed + i)))

    elif condition == "llm_tom_full":
        for i in range(1, config.num_agents + 1):
            agents.append(LLMAgent(aid(i), random.Random(config.seed + i), mode="tom"))

    elif condition == "llm_notom_full":
        for i in range(1, config.num_agents + 1):
            agents.append(LLMAgent(aid(i), random.Random(config.seed + i), mode="notom"))

    else:
        raise ValueError(
            "Unknown condition. Use one of: baseline, belief_mixed, "
            "llm_tom_mixed, llm_notom_mixed, llm_tom_full, llm_notom_full"
        )

    return agents
