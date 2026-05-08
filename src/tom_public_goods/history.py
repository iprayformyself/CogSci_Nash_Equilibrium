from __future__ import annotations

from typing import Any

from .utils import safe_mean, safe_stdev


def get_recent_rounds(history: list[dict[str, Any]], memory_window: int) -> list[dict[str, Any]]:
    return history[-memory_window:] if memory_window > 0 else []


def agent_contributions(
    history: list[dict[str, Any]], agent_id: str, memory_window: int | None = None
) -> list[int]:
    rounds = history[-memory_window:] if memory_window else history
    values: list[int] = []
    for row in rounds:
        contributions = row.get("contributions", {})
        if agent_id in contributions:
            values.append(int(contributions[agent_id]))
    return values


def summarize_agent_behavior(
    history: list[dict[str, Any]],
    agent_ids: list[str],
    focal_agent_id: str,
    memory_window: int,
    max_agents: int,
) -> list[dict[str, Any]]:
    """Compact other-agent profiles for scalable LLM prompts."""
    profiles: list[dict[str, Any]] = []

    for aid in agent_ids:
        if aid == focal_agent_id:
            continue
        vals = agent_contributions(history, aid, memory_window=memory_window)
        if not vals:
            profile = {
                "agent_id": aid,
                "last": None,
                "mean_recent": None,
                "std_recent": None,
                "trend": "unknown",
                "label": "unknown",
            }
        else:
            mean_recent = safe_mean(vals)
            std_recent = safe_stdev(vals, default=0.0)
            split = max(1, len(vals) // 2)
            first_half = vals[:split]
            second_half = vals[split:]
            trend_delta = safe_mean(second_half, mean_recent) - safe_mean(first_half, mean_recent)
            if trend_delta > 1.0:
                trend = "increasing cooperation"
            elif trend_delta < -1.0:
                trend = "decreasing cooperation"
            else:
                trend = "stable"

            if mean_recent >= 7:
                label = "cooperative"
            elif mean_recent <= 3:
                label = "self-interested"
            else:
                label = "conditional/adaptive"

            profile = {
                "agent_id": aid,
                "last": vals[-1],
                "mean_recent": round(mean_recent, 2),
                "std_recent": round(std_recent, 2),
                "trend": trend,
                "label": label,
            }
        profiles.append(profile)

    def importance(p: dict[str, Any]) -> float:
        mean = p.get("mean_recent")
        std = p.get("std_recent")
        if mean is None:
            return 0.0
        return abs(float(mean) - 5.0) + float(std or 0.0) * 0.2

    profiles.sort(key=importance, reverse=True)
    return profiles[:max_agents]


def summarize_recent_rounds(history: list[dict[str, Any]], memory_window: int) -> dict[str, Any]:
    recent = get_recent_rounds(history, memory_window)
    if not recent:
        return {
            "available_rounds": 0,
            "average_group_contribution": None,
            "last_group_average": None,
            "group_trend": "unknown",
        }

    avg_values = [float(row["group_average_contribution"]) for row in recent]
    split = max(1, len(avg_values) // 2)
    first_half = avg_values[:split]
    second_half = avg_values[split:]
    trend_delta = safe_mean(second_half, avg_values[-1]) - safe_mean(first_half, avg_values[0])

    if trend_delta > 0.7:
        trend = "group cooperation increasing"
    elif trend_delta < -0.7:
        trend = "group cooperation decreasing"
    else:
        trend = "group cooperation stable"

    return {
        "available_rounds": len(recent),
        "average_group_contribution": round(safe_mean(avg_values), 2),
        "last_group_average": round(avg_values[-1], 2),
        "group_trend": trend,
    }
