from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GameConfig:
    """Configuration for a repeated Public Goods Game experiment."""

    condition: str = "baseline"
    num_agents: int = 30
    rounds: int = 200
    endowment: int = 10
    multiplier: float = 1.6
    seed: int = 42
    memory_window: int = 8
    prediction_sample_size: int = 12
    use_llm: bool = False
    llm_agents: int = 0
    llm_provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    max_retries: int = 3
    output_dir: str = "results"
    on_llm_error: str = "fallback"  # fallback or raise
    max_output_tokens: int = 900

    def validate(self) -> None:
        if self.llm_provider not in {"openai", "groq"}:
            raise ValueError("llm_provider must be either 'openai' or 'groq'")
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        if self.rounds < 1:
            raise ValueError("rounds must be at least 1")
        if self.endowment < 1:
            raise ValueError("endowment must be at least 1")
        if self.memory_window < 0:
            raise ValueError("memory_window must be non-negative")
        if self.prediction_sample_size < 1:
            raise ValueError("prediction_sample_size must be at least 1")
        if self.llm_agents < 0:
            raise ValueError("llm_agents must be non-negative")
        if self.on_llm_error not in {"fallback", "raise"}:
            raise ValueError("on_llm_error must be either 'fallback' or 'raise'")
