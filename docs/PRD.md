# InferEdge: Private, Cost-Controlled LLM Inference Gateway

## Problem and users

Organizations running private text workloads need inference without outbound data
transfer, bounded CPU and memory consumption, and measurements they can reproduce.
The initial target is a single-host CPU deployment with one small causal language
model and operational visibility for backend engineers.

## Product requirements

- Provision model assets explicitly; all serving and benchmarks use local files.
- Load one model during application lifespan and reject readiness until loaded.
- Support deterministic greedy generation, then explicit optional sampling.
- Provide JSON generation and incremental token SSE, with cancellation on client
  disconnect and deadlines that stop generation and release capacity.
- Bound prompt, output and total context tokens, active jobs and waiting jobs.
- Keep each request's KV cache private and release it on every exit path.
- Expose latency, time to first token, throughput, queue, rejection and resource
  measurements with no prompts or request identifiers in metric labels.
- Establish a naive baseline and report measured cold, warm and concurrent runs.

## Success criteria

Offline startup and generation succeed with a supported local model. Cache-on and
cache-off greedy token IDs agree on a fixed corpus. Streaming delivers deltas
before completion. Reliability tests prove capacity is recovered after success,
failure, overload, deadline and disconnect. Performance reports identify model,
hardware, settings, sample counts and errors; no speedup or latency target is
claimed until measured on the intended deployment hardware.

## Non-goals for the initial gateway

GPU serving, model downloads, training, cross-request prefix caches, continuous
batching, multi-node scheduling, speculative decoding and a public multi-tenant
service. Authentication and production ingress policy require separate deployment
decisions. Phase 1 is a local CLI experiment, not a deployed gateway.

## Current delivery boundary

The first task delivers the architecture, implementation plan, intentionally naive
baseline and benchmark harness only. A deterministic byte-level mock validates
the tooling when real weights are unavailable. Mock timings are tooling
measurements and are never evidence of language-model speed or quality.

FastAPI endpoints, custom decoding, streaming, caching, scheduling, Prometheus,
Docker Compose and CI are planned in later phases, not implemented in phase 1.
