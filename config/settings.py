from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    database_path: Path = Path("lab.db")

    default_device: str = "cpu"
    max_concurrent_runs: int = 3

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
