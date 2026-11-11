# Implementation plan and acceptance gates

Proceed in order and report test/measurement results at each gate. This task stops
after phase 1; later phases require a subsequent task.

1. **Naive baseline and harness (current).** Build a local-only CPU loader and a
   greedy, uncached baseline that reloads on every request. Add a dependency-free
   deterministic mock, CLI, fresh-process and repeated-call experiments, raw JSON
   samples and timing summaries. Test reload behavior, deterministic outputs,
   local-only settings, validation, timing calculations and CLI failure behavior.
   Record real model measurements only when local assets exist; otherwise record
   explicitly synthetic mock results and the missing dependency.
2. **Lifespan and synchronous API.** Add FastAPI/Pydantic, lifespan model ownership,
   an executor off the event loop, custom uncached greedy decoding, typed JSON
   generation, health and readiness. Compare custom token IDs to the phase-1
   reference. Test one load per lifespan, failures and event-loop responsiveness.
3. **Streaming and cancellation.** Add incremental decoding, bounded SSE bridge,
   per-job cancellation, disconnect detection, heartbeat and terminal events.
   Test first tokens arrive before completion and abandoned jobs release resources.
4. **KV cache.** Implement per-request caches and correct mask/position updates.
   Compare exact greedy token IDs against uncached decoding across short/long
   prompts, EOS and repeated/concurrent requests. Benchmark both modes at matching
   output lengths. Record numerical or model-specific differences honestly.
5. **Budgets, scheduling and observability.** Enforce token/body/context budgets,
   finite concurrency and queue, deadline and overload errors. Add sampling with
   per-job RNG. Instrument Prometheus; add a scrape configuration and cold/warm/
   concurrent HTTP benchmarks. Verify queue fairness, cancellation races and
   capacity recovery under stress.
6. **Reliability and delivery.** Add failure injection, slow-consumer/disconnect,
   shutdown and memory stability tests; dependency and offline CI checks; a CPU
   Docker image and Compose stack with read-only model mount and Prometheus.
   Publish a measured performance report and deployment/resource limitations.

## Phase 1 measurement protocol

- CPU only; fixed requested intra-op threads (default one), greedy decoding and
  no KV cache. No runtime model acquisition or remote code.
- `repeated`: warmup requests are discarded, then measure N requests in one Python
  process. Every request still reloads the model. This is not resident-model warm
  inference; only imports, allocators and filesystem caches may be warmed.
- `cold`: one new Python process per sample, zero warmups. Capture internal load
  and request timings plus parent-observed subprocess wall time. OS file caches
  are not flushed, so this is process-cold, not guaranteed disk-cold.
- Use monotonic `perf_counter` and report seconds. Load includes tokenizer and
  model construction, imports if not yet loaded, CPU/eval setup and thread setup.
  Request timing covers load through output detokenization, excluding result
  serialization and interpreter teardown. Process wall time includes both.
- Summaries use linear interpolation at rank `(n - 1) * q` for p50/p95. Small
  sample percentiles are descriptive, not statistical evidence of tail latency.
- Record raw durations, actual output counts, output hashes, configuration and
  machine metadata. Do not publish a success report if a request fails.
- No local weights are present in the initial environment. The synthetic mock
  verifies measurement plumbing; it cannot validate Transformers integration,
  model quality, real resource use, KV correctness or performance improvements.
