# Phase 1 measured results

**These are actual measurements of the deterministic mock, not language-model
performance.** No suitable local model weights were found in the workspace or
the default Hugging Face cache, and no configured model/cache path was present.
PyTorch and Transformers were not installed. The optional real-model backend has
contract-test coverage only; its integration check was skipped. No model weights
were downloaded and no inference speedup is claimed.

## Environment and workload

Measured on November 11, 2026, at approximately 20:46 IST (15:16 UTC).

- Linux x86_64, Intel Xeon Platinum 8573C, Python 3.12.14, InferEdge 0.1.0.
- Five visible/affinity CPUs; cgroup CPU quota `400000 100000` (four CPU-seconds
  per second). This is a shared cloud environment, not an isolated CPU benchmark.
- Backend: `deterministic-byte-mock-v1`; single-threaded Python. Requested
  `cpu_threads=1` applies to PyTorch when that backend is used, not to mock work.
- Prompt: `Private inference stays on this machine.` (40 UTF-8 bytes, 41 mock
  tokens including BOS); 32 generated tokens per sample, no EOS, no KV cache.
- Fresh mock model/tokenizer objects on **every** request, including warmups.
- Repeated: 3 discarded warmups, 20 measured requests. Process-cold: 5 separate
  Python processes, no warmups. All measured outputs had identical token hashes.

## Observations

| Scenario | Samples | Mock load p50 (µs) | Internal request p50 (ms) | Internal request p95 (ms) | Process wall p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repeated reloads | 20 | 0.590 | 0.1175 | 0.4322 | Not measured |
| Fresh process | 5 | 2.473 | 0.1372 | 0.1697 | 47.3248 |

Raw reports contain all timings, percentiles, token counts, hashes and throughput:

- [Repeated reload report](../benchmarks/results/mock-repeated.json)
- [Fresh-process report](../benchmarks/results/mock-cold.json)

The mock's load time measures Python object construction, not model deserialization.
The fresh-process wall time includes interpreter startup, imports, serialization
and teardown. Its internal request duration excludes most of those costs. These
metrics have different boundaries and must not be compared as a speedup ratio.

The repeated run had scheduling/timing variation (maximum internal request time
approximately 0.8844 ms). Its p95 exceeded the cold run's p95 in this small sample;
that is not evidence that reloading in a warm interpreter is slower in general.
Short synthetic work is sensitive to instrumentation and scheduling noise. No
confidence intervals, steady-state throughput or production SLOs are established.

## Reproduce

From the repository root, after the README installation steps:

```bash
.venv/bin/python -m inferedge.benchmark --backend mock --scenario repeated \
  --runs 20 --warmup 3 --max-new-tokens 32 \
  --output benchmarks/results/mock-repeated.json
.venv/bin/python -m inferedge.benchmark --backend mock --scenario cold \
  --runs 5 --max-new-tokens 32 \
  --output benchmarks/results/mock-cold.json
```

Outputs should be deterministic; durations will vary. JSON timings are seconds.
Percentiles use linear interpolation at `(n - 1) * q`. Request and generation
throughput use actual generated token counts divided by summed corresponding
durations. OS page caches are not flushed; process-cold does not mean disk-cold.

## Validation and remaining measurements

The final test command `.venv/bin/python -m pytest` completed with **46 passed,
1 skipped**. `.venv/bin/python -m pip check` reported no broken requirements.
The real-model integration test is deliberately skipped until
`INFEREDGE_TEST_MODEL_PATH` points to approved
local model assets. Loader contract tests use test doubles to check CPU placement,
offline/safe loading flags, inference mode, greedy/uncached settings, EOS handling
and known-context overflow; they do not execute PyTorch or Transformers.

Real model load/latency, memory use, cache equivalence, cache speedup, resident
warm inference, concurrent HTTP load, time to first token, cancellation and
overload behavior are **unmeasured**. Follow README provisioning instructions for
the optional CPU dependencies and complete local tokenizer/safetensors assets,
then run the opt-in integration test and both baseline benchmarks. Later phases
will introduce the serving mechanisms needed for the remaining measurements.
