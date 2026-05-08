from __future__ import annotations

from typing import Any

from ..config import GameConfig
from ..history import get_recent_rounds
from ..models import Decision
from ..utils import clamp_int, safe_mean
from .base import BaseAgent


class CooperatorAgent(BaseAgent):
    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        return Decision(contribution=self.rng.randint(7, config.endowment), reasoning_summary="Rule: mostly cooperate.")


class DefectorAgent(BaseAgent):
    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        return Decision(contribution=self.rng.randint(0, min(2, config.endowment)), reasoning_summary="Rule: mostly free-ride.")


class RandomAgent(BaseAgent):
    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        return Decision(contribution=self.rng.randint(0, config.endowment), reasoning_summary="Rule: random contribution.")


class TitForTatAgent(BaseAgent):
    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        if not history:
            contribution = self.rng.randint(5, min(8, config.endowment))
        else:
            last = history[-1]["contributions"]
            others = [v for aid, v in last.items() if aid != self.agent_id]
            contribution = clamp_int(safe_mean(others, 5), 0, config.endowment, 5)
        return Decision(contribution=contribution, reasoning_summary="Rule: match previous average contribution of others.")


class ConditionalAgent(BaseAgent):
    """Cooperates if the recent group average is high, withdraws if the group defects."""

    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        if not history:
            contribution = self.rng.randint(5, min(8, config.endowment))
        else:
            recent = get_recent_rounds(history, config.memory_window)
            recent_avgs = [row["group_average_contribution"] for row in recent]
            group_mean = safe_mean(recent_avgs, 5)
            noise = self.rng.choice([-1, 0, 0, 1])
            if group_mean >= 6.5:
                contribution = 8 + noise
            elif group_mean >= 4.0:
                contribution = 5 + noise
            else:
                contribution = 2 + noise
            contribution = clamp_int(contribution, 0, config.endowment, 5)
        return Decision(contribution=contribution, reasoning_summary="Rule: conditional cooperation based on recent group behavior.")
