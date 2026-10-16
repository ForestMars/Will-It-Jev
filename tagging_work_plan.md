# Implementation Work Plan: V1 Asynchronous Domain Tagging Sidecar

## Context & Objectives
You are tasked with integrating the asynchronous `tagging.rs` module into `OvenRouter` (formerly `jev-router`). The goal is to record a post-routing domain classification record (`TagRecord`) for incoming requests without adding latency to the hot request path.

All queue enqueueing operations MUST execute non-blockingly on the main Tokio thread.

---

## Phase 1: Environment & State Configuration

### Step 1.1: Update State Structures (`src/state.rs` or `src/main.rs`)
1. Import `TagQueue` from `crate::tagging`.
2. Add `tag_queue: TagQueue` to your `AppState` struct:
   ```rust
   pub struct AppState {
       pub tag_queue: TagQueue,
       // ... existing router dependencies
   }
   ```

### Step 1.2: Initialize Pipeline on Startup (`src/main.rs`)
1. In the service startup routine:
   - Instantiate your concrete `Tagger` implementation wrapped in an `Arc`.
   - Load `TagConfig` (set `base_path` to `PathBuf::from("logs/task_tags.jsonl")`).
   - Call `tagging::spawn(tagger_arc, config)`.
   - Store the returned `TagQueue` inside `AppState`.
   - Hold the returned `JoinHandle` in the main server task so it drains gracefully on shutdown.

---

## Phase 2: Concrete V1 Tagger Implementation

### Step 2.1: Define V1 Domain Taxonomy (`src/tagging/v1_domains.rs` or inside `tagging.rs`)
Define the 7 target domain identifiers:
```rust
pub const V1_DOMAINS: &[&str] = &[
    "engineering",
    "company_knowledge_base",
    "help_desk",
    "customer_support",
    "web_search",
    "other_company_task",
    "other_general_request",
];
```

### Step 2.2: Implement `Tagger` Trait
Create `OvenV1Tagger` struct implementing `tagging::Tagger`:
```rust
pub struct OvenV1Tagger {
    // Inject local model client, embedding matrix, or Ollama fallback client
}

impl Tagger for OvenV1Tagger {
    fn model(&self) -> &str { "oven-domain-tagger-v1" }
    fn version(&self) -> &str { "1.0.0" }
    fn taxonomy_version(&self) -> &str { "v1.0" }

    async fn tag(&self, input: &TagInput) -> Result<TaggerOutput, TagError> {
        // 1. Run domain classification against prompt (+ optional response text)
        // 2. Return TaggerOutput containing TaggerVerdict::Tagged or TaggerVerdict::NoMatch
    }
}
```

---

## Phase 3: Request Handler Integration

### Step 3.1: Hook `TagQueue::enqueue()` in Request Handler (`src/router.rs` / `src/handlers.rs`)
Locate the primary Axum completion endpoint (e.g., `handle_chat_completion`):

1. Extract the request prompt string and unique `request_id`.
2. Execute the normal routing / upstream provider call to build the HTTP response.
3. **Immediately prior to returning the response**, fire the non-blocking enqueue call:

```rust
state.tag_queue.enqueue(TagInput {
    request_id: request_id.clone(),
    prompt: extracted_prompt,
    response: Some(downstream_response_text),
});
```

*Note: Do NOT await the tag queue or block response delivery.*

---

## Phase 4: Validation & Testing

### Step 4.1: Unit & Pipeline Tests
Run tests to confirm channel isolation and metric counting:
```bash
cargo test tagging::tests
```

### Step 4.2: Integration Verification
1. Launch `OvenRouter`.
2. Send test cURL requests across different prompt domains (`engineering`, `help_desk`, etc.).
3. Confirm daily rotating log file creation (`logs/task_tags.YYYY-MM-DD.jsonl`).
4. Validate JSONL structure:
   - Check `tagger_usage` field naming.
   - Confirm `latency_ms` and `queue_wait_ms` are present.
   - Verify non-sampled or timed-out requests produce clean `Untagged` records without dropping log lines.