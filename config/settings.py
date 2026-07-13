from pathlib import Path
from typing import Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    database_path: Path = Path("lab.db")

    default_device: str = "cpu"
    # Separate limits per device x execution backend — a laptop's local CPU/GPU
    # capacity is unrelated to how many concurrent serverless endpoint sessions
    # are allowed, so a single combined limit doesn't fit either constraint
    # well. See docs/DESIGN_DECISIONS.md.
    max_concurrent_local_cpu_runs: int = 2
    max_concurrent_local_gpu_runs: int = 1
    max_concurrent_serverless_cpu_runs: int = 3
    max_concurrent_serverless_gpu_runs: int = 3
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
    chatbot_error_tail_lines: int = 10
    # Lifecycle events (LAUNCHED/PAUSE/RESUME/STOP/CANCELLED) are naturally
    # sparse per run — this is a safety ceiling against a pathological
    # flapping run, not a real-world cap. See docs/DESIGN_DECISIONS.md.
    chatbot_training_event_tail_lines: int = 200
    # Loss/val history points kept in the chatbot's context, evenly sampled
    # across the whole run (not the most recent N) so "how did the run go"
    # can be answered after step 20. See _downsample_series in context.py.
    chatbot_loss_history_points: int = 25
    chatbot_history_window_turns: int = 10

    # Remote training backend — endpoint-only after the 2026-07-11 pivot away
    # from jobs (Nebius jobs have no public URL for the controller to route
    # training requests to; endpoints do). See docs/NEBIUS_SERVERLESS_IMPLEMENTATION_PLAN.md.
    training_backend: Literal["local", "nebius_endpoint"] = "local"
    gpu_idle_timeout_seconds: int = 600   # 10 min — matches plan doc's GPU worker policy
    cpu_idle_timeout_seconds: int = 1800  # 30 min — matches plan doc's CPU worker policy
    # Warn the user this many seconds before the idle timeout actually stops the
    # worker, so they get a "Continue session" chance — see 2026-07-11 session.
    gpu_idle_warning_seconds: int = 300   # warns at the 5-min-idle mark (10 min total)
    cpu_idle_warning_seconds: int = 600   # warns at the 20-min-idle mark (30 min total)
    idle_scan_interval_seconds: int = 30

    # Nebius endpoint client — image/subnet/CPU platform proven working via the
    # manual endpoint smoke test, see evidence/nebius-endpoint/endpoint.json.
    # nebius_cpu_preset was bumped 4vcpu-16gb -> 8vcpu-32gb on 2026-07-11
    # (requested size increase) — UNVERIFIED at this size, same 1:4 vCPU:RAM
    # ratio as the proven 4vcpu-16gb preset, but not yet smoke-tested. The CLI
    # has no preset-listing command; confirm by watching the next real
    # `nebius ai endpoint create`/`start` call succeed.
    # GPU platform+preset are also UNVERIFIED placeholders from the plan doc's
    # example command — confirm before first GPU use.
    # Split into two images on 2026-07-12 (see Dockerfile.trainer-cpu/-gpu and
    # scripts/build_push_trainer_*.sh) — the single llm-lab-backend:phase2
    # monolith pulled CUDA torch even for the CPU endpoint. Both default to
    # :latest; override per-deploy via env var once real tags are pushed.
    nebius_cpu_trainer_image: str = "cr.eu-north1.nebius.cloud/e00fjx4k9nbq206gh4/llm-lab-trainer-cpu:latest"
    nebius_gpu_trainer_image: str = "cr.eu-north1.nebius.cloud/e00fjx4k9nbq206gh4/llm-lab-trainer-gpu:latest"
    nebius_subnet_id: str = "vpcsubnet-your-subnet-id-here"
    nebius_cpu_platform: str = "cpu-d3"
    nebius_cpu_preset: str = "16vcpu-64gb"  # unverified at this size, see comment above
    nebius_gpu_platform: str = "gpu-l40s-a"       # unverified
    nebius_gpu_preset: str = "1gpu-8vcpu-32gb"    # unverified
    nebius_cpu_endpoint_name: str = "llm-lab-cpu-trainer"
    nebius_gpu_endpoint_name: str = "llm-lab-gpu-trainer"
    nebius_endpoint_container_port: int = 8000
    nebius_endpoint_poll_interval_seconds: int = 3
    # Endpoint creation can take up to ~5 min in practice (per 2026-07-11 session,
    # confirmed higher than the plan doc's 30-90s estimate) — 360s gives buffer.
    nebius_endpoint_ready_timeout_seconds: int = 360
    # Guards every single `nebius` CLI invocation — without this, a stuck CLI call
    # (seen live 2026-07-11: hung on stdin under uvicorn) blocks the request forever.
    nebius_cli_timeout_seconds: int = 60
    # Starting an already-created (stopped) endpoint is a different
    # operation from creating one from scratch — no image pull/provisioning
    # needed, just resuming an existing container over VM — so it should be
    # faster than the ~5min Nebius's own docs cite for creation. But 180s
    # was also confirmed too short live 2026-07-12: a start command timed
    # out while the endpoint was still successfully coming up on Nebius's
    # side, and the app abandoned it and created a duplicate. Bumped to
    # 300s for headroom. No official Nebius number found for this specific
    # case (checked docs.nebius.com/serverless/endpoints/manage) — this is
    # a reasoned estimate, not a documented figure. Adjust once real timing
    # data exists. Paired with worker_manager.py::ensure_worker() now
    # checking live status on timeout before assuming deleted, so an
    # under-estimate here is no longer as costly as it used to be.
    nebius_endpoint_start_timeout_seconds: int = 300

    @field_validator("nebius_subnet_id")
    @classmethod
    def _validate_subnet_id(cls, v: str) -> str:
        # Fails fast at settings-load time instead of ~5 minutes into a
        # build/push, deep inside a `nebius ai endpoint create` API error.
        # Caught live 2026-07-12: NEBIUS_SUBNET_ID in .env held a subnet's
        # NAME ("default-subnet-...") instead of its ID ("vpcsubnet-...") —
        # `nebius vpc subnet list`'s metadata.id is the value that belongs
        # here, not metadata.name.
        if not v.startswith("vpcsubnet-"):
            raise ValueError(
                f"nebius_subnet_id must start with 'vpcsubnet-' (got {v!r}) — "
                "this looks like a subnet NAME, not its ID. Check NEBIUS_SUBNET_ID "
                "in .env against `nebius vpc subnet list`'s metadata.id field."
            )
        return v

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
