from __future__ import annotations

import json
import re
import statistics
from typing import Any, Sequence


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    """Convert value to int and clamp into [low, high]."""
    try:
        if isinstance(value, str):
            value = value.strip()
        number = int(round(float(value)))
        return max(low, min(high, number))
    except Exception:
        return default


def clamp_float(value: Any, low: float, high: float, default: float | None = None) -> float | None:
    try:
        if isinstance(value, str):
            value = value.strip()
        number = float(value)
        return max(low, min(high, number))
    except Exception:
        return default


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def safe_stdev(values: Sequence[float], default: float = 0.0) -> float:
    return statistics.stdev(values) if len(values) >= 2 else default


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object from a model response, tolerating accidental markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse JSON object from model response.")


def agent_sort_key(agent_id: str) -> tuple[str, int]:
    match = re.search(r"(\D+)(\d+)$", agent_id)
    if match:
        return match.group(1), int(match.group(2))
    return agent_id, 0
