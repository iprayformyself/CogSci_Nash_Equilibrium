from __future__ import annotations

import json
import time
from typing import Any

from ..config import GameConfig
from ..history import agent_contributions, summarize_agent_behavior, summarize_recent_rounds
from ..models import Decision, Prediction
from ..utils import clamp_float, clamp_int, extract_json_object, safe_mean
from .base import BaseAgent
from .belief import BeliefBasedAgent

def _call_llm(self, prompt: str, config: GameConfig) -> str:
    if config.llm_provider == "openai":
        return self._call_openai(prompt, config)
    if config.llm_provider == "groq":
        return self._call_groq(prompt, config)
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

def _get_groq_client(self):
    try:
        from groq import Groq  # type: ignore
    except Exception as exc:
        raise RuntimeError("The groq package is not installed. Install it with: pip install groq") from exc

    return Groq()

def _call_groq(self, prompt: str, config: GameConfig) -> str:
    client = self._get_groq_client()

    system_message = (
        "You are a decision-making agent in a controlled Cognitive Science experiment. "
        "Return valid JSON only. Do not use markdown. Do not reveal private chain-of-thought. "
        "Use a concise reasoning_summary field instead."
    )

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=config.max_output_tokens,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")
    return content

class LLMAgent(BaseAgent):
    """
    LLM-controlled agent.

    mode='tom' asks for explicit Theory of Mind reasoning.
    mode='notom' asks for action selection without explicit mental-state attribution.
    """

    def __init__(self, agent_id: str, rng, mode: str = "tom"):
        super().__init__(agent_id, rng)
        self.mode = mode
        self.agent_type = f"LLMAgent_{mode}"
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise RuntimeError("The openai package is not installed. Install it with: pip install openai") from exc
        self._client = OpenAI()
        return self._client

    def decide(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> Decision:
        if not config.use_llm:
            fallback = BeliefBasedAgent(self.agent_id, self.rng)
            decision = fallback.decide(history, agent_ids, config)
            decision.reasoning_summary = "LLM disabled; used local belief-based fallback."
            return decision

        prompt = self._build_prompt(history, agent_ids, config)
        last_error = ""
        for attempt in range(1, config.max_retries + 1):
            try:
                raw = self._call_llm(prompt, config)
                parsed = extract_json_object(raw)
                return self._parse_decision(parsed, raw, agent_ids, config)
            except Exception as exc:
                last_error = f"Attempt {attempt}/{config.max_retries}: {type(exc).__name__}: {exc}"
                if attempt < config.max_retries:
                    time.sleep(1.5 * attempt)

        if config.on_llm_error == "raise":
            raise RuntimeError(last_error)

        fallback = BeliefBasedAgent(self.agent_id, self.rng)
        decision = fallback.decide(history, agent_ids, config)
        decision.api_error = last_error
        decision.reasoning_summary = "LLM call failed; used local belief-based fallback."
        return decision

    def _call_openai(self, prompt: str, config: GameConfig) -> str:
        client = self._get_client()
        instructions = (
            "You are a decision-making agent in a controlled Cognitive Science experiment. "
            "Return valid JSON only. Do not use markdown. Do not reveal private chain-of-thought. "
            "Use a concise reasoning_summary field instead."
        )

        # Uses OpenAI Responses API. The model is configurable from CLI.
        response = client.responses.create(
            model=config.model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=config.max_output_tokens,
        )
        return response.output_text

    def _build_prompt(self, history: list[dict[str, Any]], agent_ids: list[str], config: GameConfig) -> str:
        recent_summary = summarize_recent_rounds(history, config.memory_window)
        other_profiles = summarize_agent_behavior(
            history=history,
            agent_ids=agent_ids,
            focal_agent_id=self.agent_id,
            memory_window=config.memory_window,
            max_agents=config.prediction_sample_size,
        )
        own_vals = agent_contributions(history, self.agent_id, memory_window=config.memory_window)
        own_summary = {
            "agent_id": self.agent_id,
            "own_recent_contributions": own_vals,
            "own_recent_average": round(safe_mean(own_vals), 2) if own_vals else None,
        }

        game_rules = {
            "game": "Repeated Public Goods Game",
            "number_of_agents": len(agent_ids),
            "endowment_per_round": config.endowment,
            "allowed_contribution_range": [0, config.endowment],
            "multiplier": config.multiplier,
            "payoff_formula": "payoff_i = endowment - own_contribution + multiplier * total_contribution / number_of_agents",
            "social_dilemma": "Group welfare increases with cooperation, but individual free-riding can increase short-term payoff.",
        }

        base = {
            "your_agent_id": self.agent_id,
            "current_round": len(history) + 1,
            "game_rules": game_rules,
            "recent_group_summary": recent_summary,
            "your_recent_behavior": own_summary,
            "other_agent_profiles_sample": other_profiles,
        }

        if self.mode == "tom":
            task = (
                "Use functional Theory of Mind. Infer whether sampled agents are cooperative, self-interested, "
                "or adaptive; infer what they may believe about the group; predict their next contribution; "
                "then choose your own contribution. Optimize long-run payoff, not only immediate payoff."
            )
            output_spec = {
                "own_contribution": "integer from 0 to endowment",
                "predictions": [
                    {
                        "target_agent_id": "string from other_agent_profiles_sample",
                        "predicted_contribution": "number from 0 to endowment",
                        "confidence": "number from 0 to 1",
                        "inferred_type": "cooperative/self-interested/conditional/adaptive/unknown",
                        "inferred_belief": "short phrase about what the target likely believes about the group",
                    }
                ],
                "reasoning_summary": "1-3 sentences, concise, no hidden chain-of-thought",
            }
        else:
            task = (
                "Choose your next contribution based on the game history summary. Do not explicitly discuss beliefs, "
                "intentions, or mental states of other agents. You may make numeric predictions only if useful."
            )
            output_spec = {
                "own_contribution": "integer from 0 to endowment",
                "predictions": [
                    {
                        "target_agent_id": "string from other_agent_profiles_sample",
                        "predicted_contribution": "number from 0 to endowment",
                        "confidence": "number from 0 to 1",
                        "inferred_type": "leave empty string",
                        "inferred_belief": "leave empty string",
                    }
                ],
                "reasoning_summary": "1-2 sentences, concise, no mental-state language",
            }

        return json.dumps(
            {
                "experiment_context": base,
                "task": task,
                "return_json_schema": output_spec,
                "important_constraints": [
                    "Return JSON only.",
                    "own_contribution must be an integer between 0 and endowment.",
                    "predictions should refer only to agents listed in other_agent_profiles_sample.",
                    "If no history is available, make a reasonable initial contribution.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    def _parse_decision(self, parsed: dict[str, Any], raw: str, agent_ids: list[str], config: GameConfig) -> Decision:
        contribution = clamp_int(parsed.get("own_contribution"), 0, config.endowment, default=config.endowment // 2)

        predictions: list[Prediction] = []
        pred_rows = parsed.get("predictions", [])
        if isinstance(pred_rows, list):
            for row in pred_rows:
                if not isinstance(row, dict):
                    continue
                target = str(row.get("target_agent_id", "")).strip()
                if target not in agent_ids or target == self.agent_id:
                    continue
                pred_val = clamp_float(row.get("predicted_contribution"), 0, config.endowment, default=None)
                if pred_val is None:
                    continue
                conf = clamp_float(row.get("confidence"), 0.0, 1.0, default=None)
                predictions.append(
                    Prediction(
                        target_agent_id=target,
                        predicted_contribution=round(float(pred_val), 2),
                        confidence=round(float(conf), 2) if conf is not None else None,
                        inferred_type=str(row.get("inferred_type", ""))[:80],
                        inferred_belief=str(row.get("inferred_belief", ""))[:200],
                    )
                )

        summary = str(parsed.get("reasoning_summary", ""))[:500]
        return Decision(contribution=contribution, predictions=predictions, reasoning_summary=summary, raw_response=raw)
