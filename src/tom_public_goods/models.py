from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prediction:
    target_agent_id: str
    predicted_contribution: float
    confidence: Optional[float] = None
    inferred_type: str = ""
    inferred_belief: str = ""


@dataclass
class Decision:
    contribution: int
    predictions: list[Prediction] = field(default_factory=list)
    reasoning_summary: str = ""
    raw_response: str = ""
    api_error: str = ""


@dataclass
class AgentRoundRecord:
    condition: str
    round_id: int
    agent_id: str
    agent_type: str
    contribution: int
    payoff: float
    group_total_contribution: int
    group_average_contribution: float
    reasoning_summary: str = ""
    api_error: str = ""
