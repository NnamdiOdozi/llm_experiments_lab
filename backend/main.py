"""FastAPI application entry point."""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend import db
from backend.api.experiments import router as experiments_router
from backend.api.training import router as training_router
from backend.api.codegen import router as code_router
from backend.api.chatbot import router as chatbot_router
from backend.api.nebius import router as nebius_router
from backend.logging_config import setup_logging, request_log, error_log, session_log
from backend.nebius import idle_monitor
from backend.training.runner import shutdown_all_workers
from config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_path = setup_logging()
    session_log.info("Initializing database")
    await db.init_db()
    orphaned = await db.reconcile_orphaned_runs()
    if orphaned:
        session_log.warning("Reconciled %d orphaned run(s) from previous session", orphaned)
    idle_task = asyncio.create_task(idle_monitor.run_forever())
    session_log.info("Application ready — serving requests")
    yield
    session_log.info("Server shutting down — stopping active workers")
    idle_task.cancel()
    shutdown_all_workers()
    session_log.info("Shutdown complete")


app = FastAPI(
    title="LLM Experiments Lab",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        request_log.info(
            "[%s] %s %s → %d (%.0fms)",
            req_id, request.method, request.url.path, response.status_code, elapsed_ms,
        )
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        error_log.error(
            "[%s] %s %s → EXCEPTION (%.0fms): %s",
            req_id, request.method, request.url.path, elapsed_ms, exc,
        )
        raise


app.include_router(experiments_router)
app.include_router(training_router)
app.include_router(code_router)
app.include_router(chatbot_router)
app.include_router(nebius_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
