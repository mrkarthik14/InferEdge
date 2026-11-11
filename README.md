# InferEdge

**Private, Cost-Controlled LLM Inference Gateway.** Offline, CPU-first inference
with explicit resource limits and reproducible performance measurements.

This repository currently implements **phase 1: the intentionally naive baseline
and benchmark harness**. It is a CLI experiment, not an HTTP gateway yet. Model
loading on each request and disabled KV caching are deliberate reference behavior.

## Quick start: deterministic mock

Python 3.11+ is required; this environment was tested with Python 3.12.14.
Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m inferedge.baseline --backend mock --prompt 'hello' --max-new-tokens 8
.venv/bin/python -m pytest
```

The mock has no third-party runtime dependencies, no model weights, no network
calls and no language understanding. It returns deterministic printable tokens
from a stable prefix hash. For `hello` and eight output tokens, the output is
`|leex6x~`. It exists to test the harness; its timing is not LLM performance.

If package installation is unavailable on a fully disconnected machine, run the
mock directly using `PYTHONPATH=src python3 -m inferedge.baseline --backend mock`.
Tests still require a provisioned copy of Pytest. Dependency installation is an
explicit provisioning step; inference never installs packages or fetches models.

## Benchmark the naive baseline

```bash
.venv/bin/python -m inferedge.benchmark --backend mock --scenario repeated \
  --runs 20 --warmup 3 --max-new-tokens 32 \
  --output benchmarks/results/mock-repeated.json

.venv/bin/python -m inferedge.benchmark --backend mock --scenario cold \
  --runs 5 --max-new-tokens 32 \
  --output benchmarks/results/mock-cold.json
```

The default prompt is `Private inference stays on this machine.`

- `repeated` runs in one interpreter and discards explicit warmups. It **reloads
  the model on every request**. It is not a resident-model warm benchmark.
- `cold` spawns a new Python process for each sample, requires zero warmups and
  records internal timings plus subprocess wall time. It does not flush OS caches.
- JSON records each sample, model/backend, synthetic flag, software/hardware,
  thread count, token counts and hashes, stage timings, p50/p95 and throughput.
  Percentiles use linear interpolation; small sample tails are descriptive only.
- Request timing includes load, tokenization, generation and detokenization, but
  excludes JSON serialization and interpreter teardown. Load timing includes
  imports when cold, thread setup, tokenizer and model construction, CPU/eval setup.
- Throughput uses actual generated tokens (including EOS) divided by summed
  generation time or summed internal request time. Neither includes process
  startup. `process_wall_seconds` and `measurement_wall_seconds` expose that cost.
- A failed request fails the experiment with a nonzero exit status; an existing
  output file is preserved. Successful reports are written atomically. Reports
  store prompt/output hashes rather than text; CLI prompts are still visible in
  process arguments, so use synthetic benchmark text on shared machines.

Use `--help` on either command for options. Console entry points
`inferedge-baseline` and `inferedge-benchmark` are also installed into `.venv/bin`.

## Optional local Transformers model

No local model weights, PyTorch or Transformers were present at initial setup.
Actual model latency and quality have **not** been measured. No model is
downloaded by this repository, including when a requested path is missing.

Provision a small, licensed causal model (for example a roughly 100–150M-parameter
GPT-2 architecture) into an absolute local directory through your organization's
approved transfer process. The directory must contain `config.json`, complete
tokenizer assets and `.safetensors` weights (plus an index for sharded weights).
Remote Hub IDs, remote Python and pickle-only `.bin` weights are unsupported.
FP32 weights alone need about four bytes per parameter; allow additional memory
for loading, activations and the Python/PyTorch runtime.

On a network-enabled provisioning machine, explicitly install CPU-only PyTorch
before the optional extra (or prepare equivalent verified wheels for offline use):

```bash
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[transformers]'
```

These optional dependency ranges have not been integration-tested in the initial
environment. Record exact resolved versions with each real benchmark (the harness
does this automatically), and pin approved versions for your deployment.

Then, with a provisioned model at `/absolute/path/to/model`:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python -m inferedge.baseline --backend transformers \
  --model-path /absolute/path/to/model --cpu-threads 1 --max-new-tokens 16

INFEREDGE_TEST_MODEL_PATH=/absolute/path/to/model .venv/bin/python -m pytest -m local_model

.venv/bin/python -m inferedge.benchmark --backend transformers \
  --model-path /absolute/path/to/model --cpu-threads 1 \
  --scenario repeated --runs 10 --warmup 1 --max-new-tokens 32 \
  --output benchmarks/results/local-model-repeated.json

.venv/bin/python -m inferedge.benchmark --backend transformers \
  --model-path /absolute/path/to/model --cpu-threads 1 \
  --scenario cold --runs 5 --max-new-tokens 32 \
  --output benchmarks/results/local-model-cold.json
```

Both asset loaders enforce `local_files_only=True` and `trust_remote_code=False`,
even without the environment flags. Weights use CPU FP32, eval/inference mode,
greedy `generate`, one beam and `use_cache=False`. No weights are retained by the
baseline runner. A known model context limit is checked before generation; full
service budgets, deadlines and queue limits belong to later phases. Early EOS
can reduce output count. The mock always produces the requested output count.

The optional real-model test skips only when `INFEREDGE_TEST_MODEL_PATH` is unset.
When set, missing dependencies/assets fail the test. It blocks socket connections
and checks exact greedy token repeatability; it is not a cache-equivalence test.

## Design and status

- [PRD](docs/PRD.md)
- [Architecture and proposed API contract](docs/architecture.md)
- [Phased implementation plan and measurement protocol](docs/implementation-plan.md)
- [Measured results and limitations](docs/performance.md)

Source modules separate contracts, model loading, decoding, the synthetic backend,
the naive runner and benchmarking. Tests cover reload/release behavior, offline
loader arguments, CPU/greedy settings, context overflow, EOS accounting,
determinism, timing summaries, CLI experiments and failure handling. Contract
tests use ML test doubles and do not establish real-model compatibility.

FastAPI lifespan/routes and Pydantic wire schemas, custom autoregressive decoding,
sampling, SSE, cancellation, KV caching, bounded scheduling, deadlines,
Prometheus, HTTP load tests, Docker Compose and CI are explicitly deferred to
phases 2–6. There are no claims about their behavior or performance yet.
