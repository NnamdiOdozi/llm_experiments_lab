"""Manually create a Nebius endpoint outside the running app.

Calls backend/nebius/worker_manager.py::create_new_worker() — the exact
same function the running app uses automatically when a user starts a run
and no endpoint exists yet. Unlike an earlier version of this script, this
DOES write to the training_runs/worker_sessions DB tables (session_id
"worker-cpu"/"worker-gpu", same as the app), so the running app can find
and reuse an endpoint created this way instead of creating a duplicate.
See docs/DESIGN_DECISIONS.md.

Usage:
    uv run scripts/create_nebius_endpoint.py cpu
    uv run scripts/create_nebius_endpoint.py gpu
"""

import argparse
import asyncio
import sys

from backend import db
from backend.nebius import endpoints_client
from backend.nebius.worker_manager import create_new_worker
from backend.training.worker_status import session_id_for


async def main(device_type: str) -> None:
    await db.init_db()
    session_id = session_id_for(device_type)
    print(f"Creating {device_type} endpoint (session_id={session_id})...")
    endpoint_id = await create_new_worker(session_id, device_type)
    print(f"Created endpoint_id={endpoint_id}, recorded in worker_sessions as {session_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device_type", choices=["cpu", "gpu"])
    args = parser.parse_args()
    try:
        asyncio.run(main(args.device_type))
    except endpoints_client.NebiusEndpointError as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)
