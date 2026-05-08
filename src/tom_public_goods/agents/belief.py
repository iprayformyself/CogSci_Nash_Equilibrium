from __future__ import annotations

from typing import Any

from ..config import GameConfig
from ..history import agent_contributions
from ..models import Decision, Prediction
from ..utils import clamp_int, safe_mean, safe_stdev
from .base import BaseAgent


class BeliefBasedAgent(BaseAgent):
    """Non-LLM cognitive agent with simple belief updates from recent behavior."""

    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        predictions: list[Prediction] = []

        if not history:
            own = self.rng.randint(5, min(8, config.endowment))
            for aid in agent_ids:
                if aid != self.agent_id:
                    predictions.append(Prediction(aid, 6.0, confidence=0.45, inferred_type="unknown"))
            return Decision(contribution=own, predictions=predictions, reasoning_summary="Initial belief: moderate cooperation expected.")

        expected_values: list[float] = []
        for aid in agent_ids:
            if aid == self.agent_id:
                continue
            vals = agent_contributions(history, aid, memory_window=config.memory_window)
            pred = max(0.0, min(float(config.endowment), safe_mean(vals, 5.0)))
            expected_values.append(pred)

            if pred >= 7:
                inferred_type = "cooperative"
                inferred_belief = "probably expects mutual cooperation"
            elif pred <= 3:
                inferred_type = "self-interested"
                inferred_belief = "probably expects others to contribute enough while it free-rides"
            else:
                inferred_type = "conditional/adaptive"
                inferred_belief = "probably reacts to recent group cooperation"

            variability = safe_stdev(vals, default=3.0)
            confidence = max(0.2, min(0.95, 1.0 - variability / 6.0))
            predictions.append(Prediction(aid, round(pred, 2), round(confidence, 2), inferred_type, inferred_belief))

        expected_group = safe_mean(expected_values, 5.0)
        if expected_group >= 7:
            own = 8
        elif expected_group >= 5:
            own = 6
        elif expected_group >= 3:
            own = 4
        else:
            own = 2

        own += self.rng.choice([-1, 0, 0, 1])
        own = clamp_int(own, 0, config.endowment, 5)

        return Decision(
            contribution=own,
            predictions=predictions,
            reasoning_summary=f"Belief model: expected average contribution of others is {expected_group:.2f}.",
        )
