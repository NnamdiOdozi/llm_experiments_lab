from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    database_url: str = "sqlite+aiosqlite:///./lab.db"

    default_device: str = "cpu"
    max_concurrent_runs: int = 3

    data_dir: Path = Path("./data")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
