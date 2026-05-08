from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from tom_public_goods.config import GameConfig
from tom_public_goods.io import save_outputs
from tom_public_goods.simulation import PublicGoodsSimulation


def load_config(path: Path) -> GameConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(GameConfig)}
    clean_data = {key: value for key, value in data.items() if key in allowed}
    config = GameConfig(**clean_data)
    config.validate()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment from a JSON config file.")
    parser.add_argument("config", type=Path, help="Path to JSON config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    sim = PublicGoodsSimulation(config)
    sim.run()
    out_dir = save_outputs(sim)
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
