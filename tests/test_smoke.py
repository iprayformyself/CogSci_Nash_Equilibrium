from tom_public_goods.config import GameConfig
from tom_public_goods.simulation import PublicGoodsSimulation


def test_baseline_smoke_run():
    config = GameConfig(condition="baseline", num_agents=10, rounds=3, use_llm=False)
    sim = PublicGoodsSimulation(config)
    sim.run(show_progress=False)
    assert len(sim.event_records) == 30
    assert len(sim.round_summary_rows()) == 3
    assert sim.condition_summary_row()["num_agents"] == 10


def test_belief_mixed_has_predictions():
    config = GameConfig(condition="belief_mixed", num_agents=10, rounds=3, use_llm=False)
    sim = PublicGoodsSimulation(config)
    sim.run(show_progress=False)
    assert len(sim.prediction_records) > 0
