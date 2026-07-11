Below is a briefing note you can paste into the local agent. I’ve written it as instructions for the agent, with enough context to stop it wandering off into the wrong architecture.

---

# Briefing note: Nebius Serverless AI session-worker design for LLM Experiments Lab

We are working on `llm_experiments_lab`, a React + FastAPI application for hands-on tiny LLM training experiments. The current app can run locally: React frontend, FastAPI backend, SQLite, and local PyTorch training subprocesses. We have also successfully deployed the existing FastAPI backend image to a Nebius Serverless AI endpoint and verified that the local React frontend can proxy requests to that endpoint.

The next design target is **Design B: session-long remote training workers**.

The goal is not “one Nebius job per experiment.” The preferred model is:

```text
User opens / starts a training session
    ↓
Backend provisions one remote trainer worker
    ↓
User can run several experiments in that same worker
    ↓
Worker remains alive while the user is active
    ↓
Worker stops after explicit exit/logout or idle timeout
```

Idle timeout policy:

```text
GPU session worker:
    10 minutes idle timeout

CPU session worker:
    30 minutes idle timeout
```

This is intended to avoid repeated cold starts when the user runs several small experiments in one sitting.

## Important Nebius distinction

Nebius has two Serverless AI deployment types:

```text
Endpoint:
    interactive workload
    has a public managed HTTPS URL
    listens for requests until stopped/deleted
    can be stopped and started

Job:
    non-interactive background workload
    no public request URL
    runs until completion, timeout, or cancellation
    cannot be stopped/restarted like an endpoint
```

Nebius documents endpoints as request-listening workloads and jobs as background workloads that terminate on completion or timeout. Endpoints have public request URLs; jobs do not. ([Nebius AI Cloud][1])

Therefore, if we use **jobs** for session-long workers, the job must poll a command source because the frontend/backend cannot directly call the job over HTTP. If we use **endpoints** for session-long workers, the FastAPI controller can call them directly over their HTTPS URLs, but stopped endpoints need to be started before use and they are not active while stopped.

For this project, treat these as two implementation tracks:

## Track A: endpoint-based session workers

Maintain two trainer endpoints:

```text
CPU trainer endpoint
GPU trainer endpoint
```

Each endpoint hosts a trainer container and accepts one training request at a time. If inactive, stop the endpoint to avoid compute charges. When needed, start the endpoint again. Nebius states stopped endpoint compute is not charged, though mounted volumes remain chargeable. ([Nebius AI Cloud][2])

This is easier to implement because endpoints have HTTPS URLs.

Downside: starting a stopped endpoint still has cold-start/provisioning delay. Also, if many users arrive, one CPU endpoint and one GPU endpoint are not enough unless requests are queued or additional endpoints are created.

## Track B: job-based session workers

Create one Nebius job per active user session:

```text
User A → CPU/GPU Job A
User B → CPU/GPU Job B
User C → CPU/GPU Job C
```

This is the better scaling model conceptually because each user gets an isolated worker. Nebius jobs are suited to training and fine-tuning workloads and are billed while running. ([Nebius AI Cloud][3])

Downside: jobs do not expose a public request URL, so the job worker must poll for commands and write status/metrics somewhere shared.

Recommended command/status mechanism for first implementation:

```text
FastAPI controller
    owns sessions, experiments, runs, commands

Worker job
    polls FastAPI controller every few seconds
    asks: "Any command for my session_id?"
    executes train / pause / resume / prompt / shutdown
    posts status, metrics, and logs back to controller
```

Later this can move to Postgres, Redis, or object storage, but for a first prototype, controller polling is simpler.

## Container split

Current image `llm-lab-backend:phase2` is a monolith: FastAPI + training code + CUDA/PyTorch dependencies. That was useful for endpoint smoke testing, but the next architecture should split images:

```text
llm-lab-web
    React build or dev frontend support
    FastAPI controller
    Nebius CLI/API integration
    session and job lifecycle logic
    ideally no CUDA PyTorch

llm-lab-trainer-cpu
    training worker
    CPU PyTorch
    no React
    no public FastAPI dashboard unless endpoint track needs it

llm-lab-trainer-gpu
    training worker
    CUDA PyTorch
    GPU training support
    no React
```

For now, preserve local development mode. Add an explicit setting:

```text
TRAINING_BACKEND=local | nebius_job | nebius_endpoint
```

`local` must continue to work exactly as before.

## User flow to implement

Frontend Start button should eventually do this:

```text
POST /api/training/start
    body: { experiment_id, device: "cpu" | "gpu" }

FastAPI:
    creates run_id
    finds or creates session_id
    checks whether session already has an active worker for chosen device

If no active worker:
    create/start remote worker
    state = PROVISIONING

If worker exists:
    send train command to worker

Frontend:
    polls/streams run status and metrics
```

States to support:

```text
NO_WORKER
PROVISIONING
WORKER_READY
QUEUED
RUNNING
PAUSED
COMPLETED
FAILED
CANCELLED
IDLE
SHUTTING_DOWN
```

## Streaming/status/logs are important

Do not leave this as a backend-only feature. The frontend must show what is happening.

Minimum frontend panels:

```text
Training status:
    Provisioning worker...
    Worker ready
    Running step N / total
    Completed / Failed / Cancelled

Loss curve:
    structured metrics from training

Logs tab:
    recent worker events

Worker info:
    backend type
    endpoint_id or job_id
    device
    idle timeout countdown
```

Nebius itself provides lifecycle status, logs, GPU and vCPU utilization metrics in the console, but the application should not rely only on Nebius console logs for the user experience. Nebius logs are useful for evidence and debugging; the app should use structured training metrics for the actual UI. Nebius documents real-time log streaming for jobs with `nebius ai job logs <job_ID> --follow`, plus `--tail`, `--timestamps`, and time filters. ([Nebius AI Cloud][3])

Worker should emit structured events such as:

```text
WORKER_STARTED
WORKER_READY
COMMAND_RECEIVED train run_id=...
TRAINING_STARTED
STEP step=20 loss=...
CHECKPOINT_SAVED
TRAINING_COMPLETED
WORKER_IDLE
IDLE_TIMEOUT_REACHED
WORKER_SHUTDOWN
```

These should go both to stdout, for Nebius logs, and to the controller, for frontend display.

## Endpoint commands reference

Create endpoint:

```bash
nebius ai endpoint create \
  --name llm-lab-cpu-trainer \
  --image "$IMAGE" \
  --container-port 8000 \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --subnet-id "$SUBNET_ID"
```

Get managed HTTPS URL:

```bash
nebius ai endpoint get "$ENDPOINT_ID" --format json \
  | jq -r '.status.public_endpoints[] | select(startswith("https://"))' \
  | head -1
```

Call endpoint:

```bash
curl "$ENDPOINT_URL/api/health"
curl -X POST "$ENDPOINT_URL/api/training/start" \
  -H "Content-Type: application/json" \
  -d '{"experiment_id": 1, "device": "cpu"}'
```

Stop/start/delete endpoint:

```bash
nebius ai endpoint stop --id "$ENDPOINT_ID"
nebius ai endpoint start --id "$ENDPOINT_ID"
nebius ai endpoint delete --id "$ENDPOINT_ID"
```

Nebius docs confirm endpoint creation options include image, command, args, env vars, port, auth token, volumes, subnet, platform, preset, disk size and SSH key. They also document managed HTTPS URLs and stop/start commands. ([Nebius AI Cloud][2])

## Job commands reference

Create job:

```bash
nebius ai job create \
  --name llm-lab-job-smoke \
  --image "$IMAGE" \
  --container-command python \
  --args "--version" \
  --platform cpu-d3 \
  --preset 4vcpu-16gb \
  --timeout 1h \
  --subnet-id "$SUBNET_ID"
```

GPU nvidia-smi smoke test:

```bash
nebius ai job create \
  --name llm-lab-gpu-smoke \
  --image nvidia/cuda:13.1.1-runtime-ubuntu24.04 \
  --container-command bash \
  --args "-c nvidia-smi" \
  --platform gpu-l40s-a \
  --preset 1gpu-8vcpu-32gb \
  --timeout 1h \
  --subnet-id "$SUBNET_ID"
```

Nebius gives this `nvidia-smi` pattern as a job creation example. ([Nebius AI Cloud][3])

Stream logs:

```bash
nebius ai job logs "$JOB_ID" --follow --timestamps
```

Cancel/delete job:

```bash
nebius ai job cancel "$JOB_ID"
nebius ai job delete "$JOB_ID"
```

Nebius says cancelling a job immediately stops the container-over-VM and deletes the container disk, while mounted volumes are retained. ([Nebius AI Cloud][3])

## Implementation stages

Stage 1: preserve current local mode and add configuration.

```text
Add TRAINING_BACKEND setting.
Keep local subprocess training working.
Add session table/state in backend.
Add minimal worker status model.
```

Stage 2: frontend-visible remote smoke test.

```text
Add backend route:
    POST /api/nebius/jobs/smoke

It creates a Nebius job running python --version.

Frontend:
    show button or debug panel
    display job_id
    display lifecycle status
    display recent logs if available
```

This proves the frontend → FastAPI → Nebius job bridge.

Stage 3: one real remote training job.

```text
Build trainer image.
Job receives one experiment config.
Job runs training once.
Job posts metrics/status back.
Frontend shows loss curve.
Job exits.
```

Stage 4: session-long worker.

```text
Job starts and stays alive.
Worker polls controller for commands.
Multiple experiments can run in same worker.
Idle timeout:
    CPU 30 minutes
    GPU 10 minutes
Exit button sends shutdown command.
Backend cancels job on timeout.
```

Stage 5: endpoint variant.

```text
Create CPU trainer endpoint and GPU trainer endpoint.
Each accepts one request at a time.
Backend starts endpoint if stopped.
Backend sends train command when ready.
Stop endpoint after idle timeout.
```

## Guardrails

Do not break local laptop workflow. Do not remove local training mode. Do not require Nebius credentials for normal local development.

Do not hard-code endpoint URLs, job IDs, tokens, service-account keys, or registry paths in committed source. Use `.env.local` or environment variables.

Do not depend on browser close as the only shutdown signal. Implement explicit Exit plus backend-owned idle timeout. Browsers often fail to send final requests when tabs close.

Do not treat Nebius endpoint/job logs as the only source of training progress. Store structured metrics and status.

First target should be a working, visible, end-to-end smoke path from frontend to Nebius job, even if it only runs `python --version`. Then replace it with real training.

Frontend/backend routing policy

Do not hard-code Nebius endpoint URLs in React components.

The frontend should always call relative /api paths.

During local Vite development, use VITE_API_PROXY to decide where /api is proxied:
- http://localhost:8000 for local FastAPI
- https://<nebius-endpoint-url> for a deployed FastAPI endpoint

VITE_API_PROXY is a frontend/Vite setting and belongs in frontend/.env.local or frontend/.env.<mode>. It should not go in config/settings.py.

The FastAPI backend is the controller. It owns the decision about whether training runs locally, on a Nebius endpoint, or in a Nebius job. Those choices belong in backend settings / config/settings.py, not in React.

React should not directly call CPU trainer endpoints, GPU trainer endpoints, or Nebius jobs. React calls FastAPI; FastAPI calls or creates remote compute.


Development modes

Mode 1: fully local
- FastAPI runs on localhost:8000
- Vite runs on localhost:5173
- frontend/.env.local has VITE_API_PROXY=http://localhost:8000
- TRAINING_BACKEND=local

Mode 2: local frontend, Nebius FastAPI endpoint
- Vite runs on localhost:5173
- VITE_API_PROXY=https://<Nebius FastAPI endpoint URL>
- Useful for testing deployed backend image

Mode 3: local frontend + local FastAPI controller + Nebius jobs/endpoints
- Vite runs on localhost:5173
- VITE_API_PROXY=http://localhost:8000
- FastAPI settings use TRAINING_BACKEND=nebius_job or nebius_endpoint
- This is the preferred architecture test mode

Mode 4: deployed web app
- React build and FastAPI are served from a CPU VM or app host
- FastAPI controls Nebius jobs/endpoints

[1]: https://docs.nebius.com/serverless/overview "About Serverless AI - Nebius AI Cloud"
[2]: https://docs.nebius.com/serverless/endpoints/manage "Managing endpoints in Serverless AI - Nebius AI Cloud"
[3]: https://docs.nebius.com/serverless/jobs/manage "Managing jobs in Serverless AI - Nebius AI Cloud"
