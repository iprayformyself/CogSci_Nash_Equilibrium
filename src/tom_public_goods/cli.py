from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Sequence

from .config import GameConfig
from .io import save_outputs
from .simulation import PublicGoodsSimulation


def parse_args(argv: Sequence[str] | None = None) -> GameConfig:
    parser = argparse.ArgumentParser(
        description="Scalable Theory-of-Mind Public Goods Game simulator with optional LLM agents."
    )
    
    parser.add_argument(
        "--condition",
        default="baseline",
        choices=[
            "baseline",
            "belief_mixed",
            "llm_tom_mixed",
            "llm_notom_mixed",
            "llm_tom_full",
            "llm_notom_full",
        ],
        help="Experimental condition.",
    )
    parser.add_argument("--agents", type=int, default=30, help="Number of agents. Use 20+ for scale.")
    parser.add_argument("--rounds", type=int, default=200, help="Number of repeated game rounds.")
    parser.add_argument("--endowment", type=int, default=10, help="Tokens per agent per round.")
    parser.add_argument("--multiplier", type=float, default=1.6, help="Public goods multiplier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--memory-window", type=int, default=8, help="Number of recent rounds visible to agents.")
    parser.add_argument(
        "--prediction-sample-size",
        type=int,
        default=12,
        help="Max number of other agents shown to each LLM for prediction.",
    )
    parser.add_argument("--use-llm", action="store_true", help="Actually call the OpenAI API for LLM agents.")
    parser.add_argument("--provider", default="groq", choices=["openai", "groq"], help="LLM API provider.")
    parser.add_argument("--llm-agents", type=int, default=0, help="Number of LLM agents in mixed LLM conditions.")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name for LLM agents.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per LLM call.")
    parser.add_argument("--max-output-tokens", type=int, default=900, help="Maximum model output tokens per LLM call.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/JSON results.")
    parser.add_argument(
        "--on-llm-error",
        default="fallback",
        choices=["fallback", "raise"],
        help="Use local fallback or stop on LLM error.",
    )

    args = parser.parse_args(argv)

    if not (1.0 < args.multiplier < args.agents):
        print(
            "Warning: typical public goods games use 1 < multiplier < number_of_agents. "
            "Your value may change the dilemma structure.",
            file=sys.stderr,
        )

    config = GameConfig(
        condition=args.condition,
        num_agents=args.agents,
        rounds=args.rounds,
        endowment=args.endowment,
        multiplier=args.multiplier,
        seed=args.seed,
        memory_window=args.memory_window,
        prediction_sample_size=args.prediction_sample_size,
        use_llm=args.use_llm,
        llm_agents=args.llm_agents,
        model=args.model,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        output_dir=args.output_dir,
        on_llm_error=args.on_llm_error,
        llm_provider=args.provider
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)

    print("Starting experiment")
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))

    if "llm" in config.condition and config.use_llm:
    if config.llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "On Windows PowerShell: $env:OPENAI_API_KEY='your_key_here'"
        )

    if config.llm_provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "On Windows PowerShell: $env:GROQ_API_KEY='your_key_here'"
        )

    sim = PublicGoodsSimulation(config)
    sim.run()
    out_dir = save_outputs(sim)

    print("\nExperiment finished.")
    print(f"Output directory: {out_dir}")
    print("Main files:")
    print(f"- {out_dir / 'events.csv'}")
    print(f"- {out_dir / 'predictions.csv'}")
    print(f"- {out_dir / 'round_summary.csv'}")
    print(f"- {out_dir / 'agent_summary.csv'}")
    print(f"- {out_dir / 'condition_summary.csv'}")

    print("\nCondition summary:")
    print(json.dumps(sim.condition_summary_row(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
