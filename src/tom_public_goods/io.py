from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .simulation import PublicGoodsSimulation


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_outputs(sim: PublicGoodsSimulation) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(sim.config.output_dir) / f"{sim.config.condition}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "config.json", asdict(sim.config))
    write_csv(out_dir / "events.csv", sim.event_rows())
    write_csv(out_dir / "predictions.csv", sim.prediction_records)
    write_csv(out_dir / "round_summary.csv", sim.round_summary_rows())
    write_csv(out_dir / "agent_summary.csv", sim.agent_summary_rows())
    write_csv(out_dir / "condition_summary.csv", [sim.condition_summary_row()])

    if sim.raw_llm_records:
        write_json(out_dir / "raw_llm_responses.json", sim.raw_llm_records)

    return out_dir
