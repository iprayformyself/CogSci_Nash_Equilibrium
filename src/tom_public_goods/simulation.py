from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .agents.llm import LLMAgent
from .config import GameConfig
from .factory import create_agents
from .models import AgentRoundRecord
from .utils import agent_sort_key, clamp_int, safe_mean, safe_stdev


class PublicGoodsSimulation:
    """Engine for repeated Public Goods Game experiments."""

    def __init__(self, config: GameConfig):
        config.validate()
        self.config = config
        self.agents = create_agents(config)
        self.agent_ids = [agent.agent_id for agent in self.agents]
        self.history: list[dict[str, Any]] = []
        self.event_records: list[AgentRoundRecord] = []
        self.prediction_records: list[dict[str, Any]] = []
        self.raw_llm_records: list[dict[str, Any]] = []

    def run(self, show_progress: bool = True) -> None:
        for round_id in range(1, self.config.rounds + 1):
            decisions = {}

            # Simultaneous decisions: agents see previous history only.
            for agent in self.agents:
                decision = agent.decide(self.history, self.agent_ids, self.config)
                decision.contribution = clamp_int(
                    decision.contribution, 0, self.config.endowment, self.config.endowment // 2
                )
                decisions[agent.agent_id] = decision

                if decision.raw_response:
                    self.raw_llm_records.append(
                        {
                            "round_id": round_id,
                            "agent_id": agent.agent_id,
                            "agent_type": agent.agent_type,
                            "raw_response": decision.raw_response,
                        }
                    )

            contributions = {aid: decisions[aid].contribution for aid in self.agent_ids}
            total_contribution = sum(contributions.values())
            group_average = total_contribution / len(self.agent_ids)
            public_return = self.config.multiplier * total_contribution / len(self.agent_ids)
            payoffs = {aid: self.config.endowment - c + public_return for aid, c in contributions.items()}

            # Prediction error for this same round.
            for agent in self.agents:
                decision = decisions[agent.agent_id]
                for pred in decision.predictions:
                    actual = contributions.get(pred.target_agent_id)
                    if actual is None:
                        continue
                    self.prediction_records.append(
                        {
                            "condition": self.config.condition,
                            "round_id": round_id,
                            "predictor_agent_id": agent.agent_id,
                            "predictor_agent_type": agent.agent_type,
                            "target_agent_id": pred.target_agent_id,
                            "predicted_contribution": pred.predicted_contribution,
                            "actual_contribution": actual,
                            "absolute_error": abs(float(pred.predicted_contribution) - actual),
                            "confidence": pred.confidence,
                            "inferred_type": pred.inferred_type,
                            "inferred_belief": pred.inferred_belief,
                        }
                    )

            for agent in self.agents:
                aid = agent.agent_id
                decision = decisions[aid]
                self.event_records.append(
                    AgentRoundRecord(
                        condition=self.config.condition,
                        round_id=round_id,
                        agent_id=aid,
                        agent_type=agent.agent_type,
                        contribution=contributions[aid],
                        payoff=round(payoffs[aid], 4),
                        group_total_contribution=total_contribution,
                        group_average_contribution=round(group_average, 4),
                        reasoning_summary=decision.reasoning_summary,
                        api_error=decision.api_error,
                    )
                )

            self.history.append(
                {
                    "round_id": round_id,
                    "contributions": contributions,
                    "payoffs": payoffs,
                    "group_total_contribution": total_contribution,
                    "group_average_contribution": group_average,
                }
            )

            if show_progress and (round_id % max(1, self.config.rounds // 10) == 0 or round_id == 1):
                print(
                    f"Round {round_id}/{self.config.rounds}: "
                    f"avg contribution={group_average:.2f}, total={total_contribution}"
                )

    def round_summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.history:
            round_id = row["round_id"]
            contribs = list(row["contributions"].values())
            payoffs = list(row["payoffs"].values())
            pred_errors = [p["absolute_error"] for p in self.prediction_records if p["round_id"] == round_id]
            rows.append(
                {
                    "condition": self.config.condition,
                    "round_id": round_id,
                    "num_agents": len(self.agent_ids),
                    "total_contribution": row["group_total_contribution"],
                    "average_contribution": round(safe_mean(contribs), 4),
                    "std_contribution": round(safe_stdev(contribs), 4),
                    "average_payoff": round(safe_mean(payoffs), 4),
                    "std_payoff": round(safe_stdev(payoffs), 4),
                    "average_prediction_error": round(safe_mean(pred_errors), 4) if pred_errors else "",
                    "num_predictions": len(pred_errors),
                }
            )
        return rows

    def agent_summary_rows(self) -> list[dict[str, Any]]:
        rows = []
        by_agent: dict[str, list[AgentRoundRecord]] = {}
        for record in self.event_records:
            by_agent.setdefault(record.agent_id, []).append(record)

        pred_by_agent: dict[str, list[float]] = {}
        for pred in self.prediction_records:
            pred_by_agent.setdefault(pred["predictor_agent_id"], []).append(float(pred["absolute_error"]))

        for aid, records in sorted(by_agent.items(), key=lambda kv: agent_sort_key(kv[0])):
            contributions = [r.contribution for r in records]
            payoffs = [r.payoff for r in records]
            errors = pred_by_agent.get(aid, [])
            rows.append(
                {
                    "condition": self.config.condition,
                    "agent_id": aid,
                    "agent_type": records[0].agent_type,
                    "average_contribution": round(safe_mean(contributions), 4),
                    "std_contribution": round(safe_stdev(contributions), 4),
                    "average_payoff": round(safe_mean(payoffs), 4),
                    "std_payoff": round(safe_stdev(payoffs), 4),
                    "average_prediction_error": round(safe_mean(errors), 4) if errors else "",
                    "num_predictions": len(errors),
                    "num_api_errors": sum(1 for r in records if r.api_error),
                }
            )
        return rows

    def condition_summary_row(self) -> dict[str, Any]:
        all_contribs = [r.contribution for r in self.event_records]
        all_payoffs = [r.payoff for r in self.event_records]
        all_errors = [float(p["absolute_error"]) for p in self.prediction_records]
        llm_count = sum(1 for agent in self.agents if isinstance(agent, LLMAgent))
        return {
            "condition": self.config.condition,
            "num_agents": self.config.num_agents,
            "rounds": self.config.rounds,
            "llm_agents": llm_count,
            "use_llm": self.config.use_llm,
            "model": self.config.model if self.config.use_llm else "",
            "average_contribution": round(safe_mean(all_contribs), 4),
            "std_contribution": round(safe_stdev(all_contribs), 4),
            "average_payoff": round(safe_mean(all_payoffs), 4),
            "std_payoff": round(safe_stdev(all_payoffs), 4),
            "average_prediction_error": round(safe_mean(all_errors), 4) if all_errors else "",
            "num_predictions": len(all_errors),
            "num_api_errors": sum(1 for r in self.event_records if r.api_error),
            "final_round_average_contribution": round(self.history[-1]["group_average_contribution"], 4)
            if self.history
            else "",
        }

    def event_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.event_records]
