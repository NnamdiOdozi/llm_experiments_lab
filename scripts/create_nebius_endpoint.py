"""Manually create a Nebius endpoint outside the running app.

Reuses backend/nebius/worker_manager.py::endpoint_create_kwargs() for the
settings lookup and backend/nebius/endpoints_client.py::create_endpoint()
for the actual `nebius ai endpoint create` call — same code the app uses
automatically when a user starts a run and no endpoint exists yet. This
script does NOT touch the training_runs/worker_sessions DB tables; it's for
one-off manual creation (e.g. right after pushing a new image), not for
replacing the app's own worker-reuse tracking.

Usage:
    uv run scripts/create_nebius_endpoint.py cpu
    uv run scripts/create_nebius_endpoint.py gpu
"""

import argparse
import asyncio
import sys

from backend.nebius import endpoints_client
from backend.nebius.worker_manager import endpoint_create_kwargs


async def main(device_type: str) -> None:
    kwargs = endpoint_create_kwargs(device_type)
    print(f"Creating {device_type} endpoint: {kwargs}")
    endpoint_id = await endpoints_client.create_endpoint(**kwargs)
    print(f"Created endpoint_id={endpoint_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device_type", choices=["cpu", "gpu"])
    args = parser.parse_args()
    try:
        asyncio.run(main(args.device_type))
    except endpoints_client.NebiusEndpointError as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        sys.exit(1)
