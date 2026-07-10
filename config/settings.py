from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    database_path: Path = Path("lab.db")

    default_device: str = "cpu"
    max_concurrent_runs: int = 2
    max_concurrent_gpu_runs: int = 1
    stop_grace_seconds: int = 5
    stop_kill_seconds: int = 10
    # GPU yield — sync GPU every N training steps to prevent WSL2 display
    # compositor starvation.  Disable in Docker/native Linux where there's
    # no shared display compositor.
    gpu_yield_enabled: bool = True
    gpu_yield_interval: int = 5
    gpu_yield_sleep: float = 0.200  # 200ms — long enough to drain GPU driver queue

    data_dir: Path = Path("./data")

    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    # Dataset URLs
    shakespeare_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dinos_url: str = "https://drive.google.com/uc?id=1SzD-jBAyLuakrbR4nprQoU01LrjrkOZk&export=download"

    # Network
    http_timeout: int = 30

    # Reproducibility
    random_seed: int = 1337

    # Branding
    github_url: str = "https://github.com/NnamdiOdozi/llm-experiments-lab"

    # Grounded chatbot / Nebius Token Factory
    nebius_key: str | None = None
    token_factory_base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    # "Thinking-2507" doesn't exist on Token Factory (verified via GET /v1/models
    # on 2026-07-10) — Qwen3-Next-80B-A3B-Thinking is the closest match that still
    # has extended-reasoning mode, which is the property this chatbot needs.
    token_factory_model: str = "Qwen/Qwen3-Next-80B-A3B-Thinking"
    chatbot_log_tail_lines: int = 50
    chatbot_history_window_turns: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
