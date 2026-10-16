<h1 align="center">WILL IT JEV? </h1>

![Polyglot UI](/assets/will-it-jev.jpeg)

## Cascade Router Engine

An in-process, high-throughput request routing engine written in Rust. It classifies incoming prompts and routes them to discriminative (Jev) or generative (LLM) execution pathways with a **p99 decision latency under 5 ms** for the overwhelming majority of traffic.

The engine is a **five-tier deterministic fallback cascade**. Each tier either resolves the request with sufficient confidence or passes it to the next, more expensive tier. Cheap tiers absorb most traffic, so the expensive tiers only see the ambiguous remainder.

<table width="100%"><tr><td>
 <h3>Most LLM requests do not need a generative model.</h3>
  <h3>Route the ones that don't to a much cheaper decision engine, and escalate only when necessary.</h3>
</td></tr></table>


---

## Table of Contents

1. [Design Goals](#design-goals)
2. [Cascade Semantics](#cascade-semantics)
3. [Tier Reference](#tier-reference)
4. [Latency Budget and Target Coverage](#latency-budget-and-target-coverage)
5. [Repository Layout](#repository-layout)
6. [Runtime and Performance Model](#runtime-and-performance-model)
7. [Configuration](#configuration)
8. [Building](#building)
9. [Model Artifacts](#model-artifacts)
10. [Testing](#testing)
11. [Benchmarking](#benchmarking)
12. [Observability](#observability)
13. [Extending the Router](#extending-the-router)
14. [Failure Modes and Fallback Behavior](#failure-modes-and-fallback-behavior)
15. [Known Constraints](#known-constraints)
16. [Contributing](#contributing)
17. [License](#license)

---

## Design Goals

- **Bounded latency.** Every tier has an explicit p99 budget. Total decision latency for tiers 1 to 3 stays within the 5 ms SLA.
- **Early exit.** A request leaves the cascade at the first tier whose confidence meets its configured threshold.
- **No GC pauses, minimal allocation.** The hot path is native Rust with C++ bindings, a replaced global allocator, and zero-copy ingress for the cheapest tiers.
- **Deterministic behavior.** Given the same input, model artifacts, and configuration, the router returns the same decision.
- **Pluggable evaluators.** Tier 2 and tier 3 engines sit behind a common trait, so models can be swapped through configuration without touching the dispatcher.

---

## Cascade Semantics

A request is a prompt payload. The dispatcher in `src/router/cascade.rs` runs tiers in order and stops at the first tier that returns a resolved outcome.

Each tier returns one of two outcomes:

- **Resolved:** the tier produced a routing decision whose confidence meets its threshold. The cascade terminates and the decision is returned.
- **PassThrough:** the tier could not decide with enough confidence (or is disabled). The payload moves to the next tier.

Tier 5 is the terminal tier. It always produces a decision, so the cascade is total: every request receives a routing outcome.

Disabled tiers are skipped without cost. Tiers are always evaluated in numeric order, and a tier is never re-entered once passed.

---

## Tier Reference

### Tier 1: Static (Fuzzy) Rules and Regex Guardrails

- **Low-Latency Deterministic Pipeline:** Replaces conventional regex guardrails with Aho-Corasick automata and Damerau-Levenshtein fuzzy matching, scanning the immutable payload through exact positional, fuzzy-token, imperative-zone, prefix, and polarity checks.
- **Ambiguity as a First-Class Output:** The policy explicitly models competing intent. When strong LLM generative cues conflict with Jev discriminative signals, Tier 1 enters an explicit conflict state and defers to Tier 2 rather than forcing a premature guess.
- **Deterministic Query Fast Paths:** Recognizes requests whose answers can be resolved through closed-form computation or other deterministic operations, bypassing semantic model inference entirely. Math dependent requests are the simplest case; the same layer is designed to absorb quantitative and other structured queries as additional fast paths.

### Tier 2: Learned Decision Layer

- **Subword Route Classification:** FastText provides the first learned routing layer, using subword features to classify prompts into Jev or generative execution paths. The classifier exposes the full probability distribution rather than discarding the model's raw evidence after selecting the top class. n-gram embeddings evaluated via in-process static C++ bindings (`libfasttext`).
- **Confidence-Gated Escalation:** Tier 2 does not treat the top prediction as sufficient by itself. Its calibrated score is compared against a configurable confidence threshold; high-confidence Jev predictions terminate the cascade, while uncertain cases pass through to the next tier. The threshold therefore controls the tradeoff between Jev coverage and false-Jev commits. The classifier produces the top-1 probability P1 and top-2 probability P2. A margin-scaled confidence score is computed as:

  `S = P1 * (1.0 + (P1 - P2))`

  This tier resolves when `S >= confidence_threshold`; otherwise it returns PassThrough. A wide margin between the top two classes raises confidence, and a narrow margin lowers it, even when P1 alone is high.
- **Empirical Threshold Calibration:** Thresholds are evaluated against held-out labeled data rather than chosen by hand. The evaluation harness sweeps the actual score boundaries produced by the model, reports Jev precision and coverage with 95% Wilson confidence intervals, and rejects evaluation sets that overlap normalized training prompts. Source-held-out experiments provide an additional test of cross-source generalization.
- **Robust** to typos, inflections, and fuzzy keyword variation, because subword features generalize across spellings.
- **Artifact:** quantized binary `router_fastpath.ftz`.

### Tier 3: Micro-Transformer / Embedding Evaluators

- **Exemplars at Runtime:** Tier 3 does not use a separately trained routing model. It loads the current labeled exemplar set at runtime and evaluates the incoming request directly against those examples, keeping the routing knowledge explicit and updateable without retraining.
- **Pluggable Semantic Engines:** Harrier is the default evaluator, with ModernBERT as an upcoming alternative and Oryn available as an optional engine. The exemplar set remains constant while the semantic engine can change independently.
- **Example-Driven Resolution:** Rather than learning a permanent decision boundary from the exemplars, Tier 3 uses them directly to determine whether the request is sufficiently supported by the observed Jev or LLM examples. Weak, conflicting, or out-of-distribution evidence passes through to Tier 4.

### Tier 4: Jev Self-Evaluation

- **Capability-Level Judgment:** Tier 4 uses Jev itself to determine whether Jev can handle the incoming request. The system therefore moves from predicting the route to asking the destination capability whether the request belongs within its executable domain.
- **Final Discriminative Gate:** Tier 4 provides the last opportunity to resolve a request without invoking a generative model. A positive Jev determination terminates the cascade; uncertainty or a negative determination escalates to Tier 5.
- **Recursive Architecture:** The cascade progressively asks increasingly capable systems whether generation is actually necessary, culminating in the unusual case where Jev evaluates whether Jev can solve the problem.

### Tier 5: Generative LLM Fallback
- **Open-Ended Generation:** Tier 5 is the terminal generative pathway for requests that survive every discriminative tier or inherently require open-ended language generation, synthesis, or reasoning.
- **Guaranteed Termination:** Tier 5 always produces the final routing outcome, making the cascade total even when every cheaper evaluator abstains or fails, or are not unavailable.
- **Expensive by Design:** The generative model is deliberately reserved for the residual traffic that earlier tiers cannot confidently establish as JEV-capable, minimizing unnecessary token consumption while preserving full LLM capability for genuinely generative requests.
---

## Latency Budget and Target Coverage

Latency figures are per-tier p99 budgets. Coverage figures are **design targets** for the share of traffic each tier should resolve. Actual values depend on your traffic mix and must be measured (see [Benchmarking](#benchmarking) and [Observability](#observability)).

| Tier | Engine | p99 Budget | Target Coverage |
| :--- | :--- | :--- | :--- |
| 1 | `regex::RegexSet` and hash maps | < 0.1 ms | 25% to 35% |
| 2 | FastText subword (`libfasttext`) | 0.2 ms to 0.4 ms | 40% to 50% |
| 3 | ModernBERT / Harrier / Oryn via ONNX Runtime | 1.5 ms to 8.0 ms | 15% to 25% |
| 4 | Secondary heavy model | 20 ms to 50 ms | 3% to 5% |
| 5 | Frontier LLM | Variable | < 1% |

Note that the 5 ms SLA holds for traffic that exits at tiers 1 to 3 within their lower budget range. The upper end of tier 3 (8 ms), tier 4, and tier 5 intentionally exceed it, which is why keeping their coverage small matters.

---

## Repository Layout

```text
src/
  config/
    mod.rs            Configuration loader with environment overrides
    schema.rs         Serde schemas for tier thresholds, model paths, flags
  logging/
    mod.rs            Low-overhead tracing and telemetry initialization
  router/
    cascade.rs        Pipeline dispatcher and lifecycle orchestration
    tier1.rs          Static hash, length bound, and regex evaluator
    tier2.rs          FastText subword classifier runner
    tier3.rs          In-process micro-transformer runner
    tier4.rs          Heavy fallback model runner
    tier5.rs          Frontier LLM escalation runner
  tier3/
    calibration.rs    Margin calculation and probability calibration
    trait.rs          Core evaluator trait and contract definitions
    inprocess/
      fasttext.rs     FFI wrapper for libfasttext
      modernbert.rs   ONNX Runtime implementation for ModernBERT
      harrier.rs      ONNX Runtime implementation for Harrier
      oryn.rs         Oryn in-process engine
    remote/
      client.rs       Asynchronous gRPC client
      proto.rs        Generated Protocol Buffer bindings
```

---

## Runtime and Performance Model

- **Global allocator:** the system allocator is replaced with `mimalloc` via `#[global_allocator]` to avoid heap lock contention across worker threads.
- **Threading:** a fixed worker pool sized by `router.workers`. Pinning workers to physical cores is recommended for stable tail latency.
- **Zero-copy ingress:** tiers 1 and 2 receive the payload as `&str`, so no copy or allocation is made during pre-filtering.
- **Quantization:** tier 3 models run Int8 to fit the latency budget on CPU.
- **Static linking:** `libfasttext` is linked statically to avoid dynamic loading overhead and deployment drift.

---

## Configuration

The router reads `config/router.toml`. Any key can be overridden by an environment variable with the `ROUTER__` prefix and double-underscore nesting, for example `ROUTER__TIER2__CONFIDENCE_THRESHOLD=0.9`.

```toml
[router]
workers = 16
max_payload_bytes = 1048576

[router.tier1]
enabled = true
regex_patterns_path = "config/rules.regex"

[router.tier2]
enabled = true
model_path = "models/router_fastpath.ftz"
confidence_threshold = 0.85

[router.tier3]
enabled = true
engine = "modernbert"            # modernbert | harrier | oryn
model_path = "models/tier3_quantized.onnx"
tokenizer_path = "models/tokenizer.json"
centroids_path = "models/centroids.bin"
confidence_threshold = 0.80

[router.tier4]
enabled = true
timeout_ms = 50

[router.tier5]
enabled = true
endpoint_url = "https://api.frontier.internal/v1/classify"
```

The values above are illustrative defaults. Thresholds must be calibrated on your own validation data before production use.

### Key parameters

| Key | Description |
| :--- | :--- |
| `router.workers` | Number of worker threads in the routing pool |
| `router.max_payload_bytes` | Payloads above this size are rejected at tier 1 |
| `tier1.regex_patterns_path` | File containing the compiled-at-startup regex rule set |
| `tier2.confidence_threshold` | Minimum margin-scaled score `S` to resolve at tier 2 |
| `tier3.engine` | Which in-process evaluator to load |
| `tier3.centroids_path` | Pre-computed route centroids for cosine scoring |
| `tier3.confidence_threshold` | Minimum similarity-derived confidence to resolve at tier 3 |
| `tier4.timeout_ms` | Hard deadline for the heavy fallback model |
| `tier5.endpoint_url` | Frontier model endpoint |

---

## Building

### Prerequisites

- A stable Rust toolchain
- A C++ toolchain and `clang`, required to build the static FastText library
- `pkg-config`
- Python 3 (for model training, exemplar dataset loading/serving, and calibration)

```bash
# Debian / Ubuntu
sudo apt-get install build-essential clang pkg-config

# macOS
xcode-select --install
```

### Compile

```bash
cargo build --release
```

For best SIMD performance, build with native CPU features enabled:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Binaries built with `target-cpu=native` are not portable across CPU generations. Build on, or for, the deployment hardware.

---

## Model Artifacts

The router expects these artifacts at the paths configured above. They are not checked into the repository.

| Artifact | Used by | Produced by |
| :--- | :--- | :--- |
| `router_fastpath.ftz` | Tier 2 | `scripts/train_fasttext.py` |
| `tier3_quantized.onnx` | Tier 3 | Your ONNX export and Int8 quantization pipeline |
| `tokenizer.json` | Tier 3 | The tokenizer matching the chosen Tier 3 model |
| `centroids.bin` | Tier 3 | Route centroid computation over labeled embeddings |

### Training the Tier 2 model

```bash
python3 scripts/train_fasttext.py \
  --input data/training_prompts.txt \
  --output models/router_fastpath.ftz
```

After training, recalibrate `tier2.confidence_threshold` against a held-out set. A retrained model invalidates the previous threshold.

---

## Testing

```bash
cargo test --all-targets
cargo run --example eval_tier1
cargo run --example eval_tier2
```

The Tier 1 eval reads `evals/tier1/cases.jsonl` and reports Jev precision, coverage,
LLM-verdict precision, and deferred cases. Review false Jev commits before changing
the Tier 1 thresholds or pattern set.

The Tier 2 eval reads `calibration/ft_valid.txt`, the held-out split produced from
the labeled calibration exemplars, and sweeps Jev precision and coverage across
confidence thresholds. It checks for normalized prompt overlap with
`calibration/ft_train.txt`, reports Wilson confidence intervals for precision,
evaluates each distinct Jev score boundary as well as `0` and `1`, and lists false
Jev commits at the current `0.95` threshold. To evaluate an independent labeled set,
pass it with `--cases` and provide its training source with `--training-cases`. Use a
separate representative validation set before changing the production threshold.

For a source-held-out stress test, train without one exemplar file and write all
generated artifacts outside the repository:

```bash
python3 calibration/make_fasttext_train.py \
  --holdout-source calibration/exemplars_ac-ct.jsonl \
  --output-dir /tmp/tier2-source-holdout
cargo run --example eval_tier2 -- \
  --model /tmp/tier2-source-holdout/tier2.bin \
  --cases /tmp/tier2-source-holdout/ft_holdout.txt \
  --training-cases /tmp/tier2-source-holdout/ft_train.txt
```

This checks cross-source generalization, not independent production performance;
the held-out prompts come from the same labeled exemplar collection.

To see tier-level output while debugging:

```bash
cargo test --all-targets -- --nocapture
```

The suite should cover, at minimum:

- Per-tier resolve and pass-through behavior at threshold boundaries
- End-to-end cascade ordering and short-circuiting
- Disabled-tier skipping
- Payload bound rejection at tier 1
- Calibration math in `calibration.rs`

---

## Benchmarking

Micro-benchmarks use Criterion.

```bash
cargo bench
```

When validating the SLA:

- Benchmark each tier in isolation first, then the full cascade.
- Report p50, p99, and p99.9, not means.
- Run on the target deployment hardware with the production allocator and build flags.
- Use a representative traffic mix. Coverage targets in this document are only meaningful against realistic input distributions.

---

## Observability

Runtime observability records are written to ClickHouse. `tracing` provides local console diagnostics; the gateway does not export logs or traces to Loki, Tempo, or another remote telemetry service.

### Span Attributes and Evaluation Tracking
The Tier1Engine::classify method instruments spans with individual feature metrics and routing decisions:

- Exact & Fuzzy Feature Counters: matches_llm_prefix, starts_polar, jev_strong_hits, jev_mod_hits, llm_strong_hits
- Fast-Path Markers: is_binary_question
- Decision State: confidence, routing_reason

Tracking resolved-at-tier distribution over time is the primary health signal. A shift of traffic toward Tiers 4 and 5 indicates threshold, model, or prompt-distribution drift, causing tail latency and operational cost to rise.

### Adjusting Log Verbosity
Verbosity is controlled via RUST_LOG. The default level is info.
```RUST_LOG=info cargo run --release```
Recommended production configuration:
```RUST_LOG=info,cascade_router=debug```

Tracking **resolved-at-tier distribution over time** is the most important signal. A shift of traffic toward tiers 4 and 5 means thresholds, models, or the input distribution have drifted, and both cost and tail latency will rise.

### Request Capture

Each completed route is sent through a bounded, non-blocking Tokio channel to a background writer, which appends one JSON object per line to `logs/routing_capture.jsonl` and inserts it into ClickHouse's `route_records_raw` table. If that queue is full, the record is dropped with a warning rather than delaying the response. ClickHouse insert failures are logged and the local spool is retained for replay. The log directory is created at startup. The record includes a UUID v4 request ID, the request-start UTC timestamp, the raw prompt, SHA-256 hashes of the Tier 1 TOML and Tier 2 model files, the outcomes from tiers that ran, and the final backend, tier, and latency in microseconds. Tier 2 records its raw probability vector; Tier 3 records its score. The sampling probability currently defaults to `1.0`.

The raw prompt is stored verbatim and may contain sensitive data. Restrict access to the capture file and define appropriate retention and cleanup; the writer appends indefinitely and does not rotate the file.

### Running the Gateway

`cargo run --release` starts the HTTP gateway on `127.0.0.1:8080` by default. It exposes
`GET /healthz`, `POST /v1/completions`, and `POST /v1/decisions`. Prompt-only requests are sent
to the configured generative model. Structured requests may include `state` and `questions`; when the cascade
selects Jev, Ollaya/Laya is asked to assess Jev eligibility and answer those typed questions in
one decision request. If Jev is not eligible or the Laya call fails, the request falls through to
the Gemma/Ollama generator.

```bash
docker run -d --name ollaya -p 11435:11435 ghcr.io/ollaya-dev/ollaya
ollaya run laya
ollama pull gemma3:latest
ollama pull qwen3:1.7b
cargo run --release
```

Ollaya publishes multi-architecture images (including `linux/amd64` and `linux/arm64`); Docker
selects the host architecture. Run ClickHouse with the migrations in `platform/clickhouse/init`
applied to enable typed route-decision and task-tag ingestion. The route and tag pipelines retain
JSONL spool files under `logs/`; ClickHouse insert failures are logged and do not block gateway
responses. Spool lines remain available for manual replay if ClickHouse is unavailable.

The principal runtime overrides are `ROUTER_BIND`, `TIER4_ENDPOINT` (default
`http://127.0.0.1:11435/api/decide`), `OLLAMA_GENERATE_URL` (default
`http://127.0.0.1:11434/api/generate`), `GEMMA_MODEL` (default `gemma3:latest`),
`TAGGER_OLLAMA_URL` (default `http://127.0.0.1:11434`), `TAGGER_MODEL` (default
`qwen3:1.7b`), and `CLICKHOUSE_URL` (default `http://127.0.0.1:8123`).

Example structured request:

```bash
curl -s http://127.0.0.1:8080/v1/decisions \
  -H 'content-type: application/json' \
  -d '{"state":"Is 2 + 2 equal to 4?","questions":{"answer":{"type":"choice","instructions":"Does the statement hold?","criteria":{"yes":"It is true.","no":"It is false."}}}}'
```

### Adjusting the Log Level

Log verbosity is controlled with the `RUST_LOG` environment variable. The default level is `info`. Invalid `RUST_LOG` values cause startup to fail with a configuration error.

Set `RUST_LOG=debug` for more detail:

```bash
RUST_LOG=info cargo run --release
```

Common settings:

| Value | Effect |
| :--- | :--- |
| `RUST_LOG=info` | Recommended for normal operation and production |
| `RUST_LOG=debug` | Router decision detail without frame-level transport output |
| `RUST_LOG=warn` | Warnings and errors only |
| `RUST_LOG=trace` | Everything, including frame-level transport output. Use only for short debugging sessions |
| `RUST_LOG=trace,h2=warn,hyper=warn` | Full trace output for the router, with `h2` and `hyper` silenced |

Trace-level logging adds overhead and can distort the latency figures in this document. Benchmark and deploy with `info` or higher.


---

## Extending the Router

Tier 2 and tier 3 engines implement a common evaluator contract defined in `src/tier2/trait.rs`. To add an engine:

1. Add an implementation under `src/tier2/inprocess/` (or a remote one under `src/tier2/remote/`).
2. Implement the evaluator trait, returning a resolved decision with a confidence score, or a pass-through.
3. Register the engine name in the configuration schema in `src/config/schema.rs`.
4. Wire selection in the corresponding tier runner in `src/router/`.
5. Add threshold-boundary tests and a Criterion benchmark.

Engines must not allocate on the per-request hot path where avoidable, and must keep their documented p99 within the tier's budget.

---

## Failure Modes and Fallback Behavior

- **Tier pass-through:** low confidence is not an error. It is the normal escalation path.
- **Model load failure at startup:** the router should fail fast rather than silently run with a tier missing. Disabling a tier must be an explicit configuration choice.
- **Tier 4 timeout:** when `timeout_ms` elapses, the request escalates to tier 5 rather than blocking.
- **Tier 5 unavailable:** the terminal tier is an external dependency. Define and test your degraded-mode policy (default route, error, or retry) before production.
- **Oversize or malformed payloads:** rejected at tier 1 based on `max_payload_bytes`.

---

## Known Constraints

- The 5 ms p99 SLA applies to traffic resolved within tiers 1 to 3 at the lower end of tier 3's range. Tiers 4 and 5 exceed it by design.
- Coverage percentages are targets, not guarantees.
- AVX-512 and NEON acceleration depend on the host CPU. Throughput and latency on other hardware will differ.
- Thresholds are coupled to specific model artifacts. Swapping or retraining a model requires recalibration.

---

## Contributing

1. Open an issue describing the change before starting large work.
2. Keep changes to tiers 1 to 3 allocation-free on the hot path.
3. Include tests and, for any hot-path change, Criterion benchmark results before and after.
4. Run `cargo fmt`, `cargo clippy --all-targets`, and `cargo test --all-targets` before opening a pull request.

---

## License

See the `LICENSE` file in the repository root.
