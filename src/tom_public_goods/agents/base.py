from __future__ import annotations

import random
from typing import Any

from ..config import GameConfig
from ..models import Decision


class BaseAgent:
    """Base class for all experimental agents."""

    def __init__(self, agent_id: str, rng: random.Random):
        self.agent_id = agent_id
        self.rng = rng
        self.agent_type = self.__class__.__name__

    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        raise NotImplementedError
