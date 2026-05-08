from __future__ import annotations

from tom_public_goods.config import GameConfig
from tom_public_goods.io import save_outputs
from tom_public_goods.simulation import PublicGoodsSimulation


EXPERIMENTS = [
    GameConfig(condition="baseline", num_agents=30, rounds=200, use_llm=False, output_dir="results"),
    GameConfig(condition="belief_mixed", num_agents=30, rounds=200, use_llm=False, output_dir="results"),
    # LLM conditions below will use local fallback unless use_llm=True is set.
    # This protects against accidental API spending.
    GameConfig(condition="llm_notom_mixed", num_agents=30, llm_agents=6, rounds=80, use_llm=False, output_dir="results"),
    GameConfig(condition="llm_tom_mixed", num_agents=30, llm_agents=6, rounds=80, use_llm=False, output_dir="results"),
]


def main() -> None:
    for config in EXPERIMENTS:
        print("=" * 80)
        print(f"Running: {config.condition}")
        sim = PublicGoodsSimulation(config)
        sim.run()
        out_dir = save_outputs(sim)
        print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
