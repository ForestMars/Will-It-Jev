#!/usr/bin/env bash
#
# ollaya-healthcheck.sh
#
# Readiness check for Ollaya's native decision API.
# Verifies: HTTP liveness, API version, local model availability, and a real
# typed inference through POST /api/decide.
#
# Requirements: bash, curl, jq
#
# Configuration:
#   OLLAYA_URL       Base URL (default: http://127.0.0.1:11435)
#   OLLAYA_MODEL     Model to exercise (default: laya:multilingual)
#   OLLAYA_KEEP_ALIVE Duration to keep the model loaded (default: 5m)
#   OLLAYA_API_KEY   Optional API key, sent as Bearer authorization
#   OLLAYA_CONNECT_TIMEOUT_SECONDS (default: 3)
#   OLLAYA_HTTP_TIMEOUT_SECONDS    (default: 15)
#   OLLAYA_INFERENCE_TIMEOUT_SECONDS (default: 120)

set -Eeuo pipefail

OLLAYA_URL="${OLLAYA_URL:-http://127.0.0.1:11435}"
OLLAYA_URL="${OLLAYA_URL%/}"
OLLAYA_MODEL="${OLLAYA_MODEL:-laya:multilingual}"
OLLAYA_KEEP_ALIVE="${OLLAYA_KEEP_ALIVE:-5m}"
CONNECT_TIMEOUT_SECONDS="${OLLAYA_CONNECT_TIMEOUT_SECONDS:-3}"
HTTP_TIMEOUT_SECONDS="${OLLAYA_HTTP_TIMEOUT_SECONDS:-15}"
INFERENCE_TIMEOUT_SECONDS="${OLLAYA_INFERENCE_TIMEOUT_SECONDS:-120}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    printf '[FAIL] %s\n' "$*" >&2
    exit 1
}
pass() { printf '[PASS] %s\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }
section() {
    printf '\n================================================================\n  %s\n================================================================\n' "$1"
}

for cmd in curl jq; do
    command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found: $cmd"
done

# Apply optional auth without expanding an empty Bash array. With `set -u`,
# `${EMPTY_ARRAY[@]}` can fail on macOS's system Bash as well as newer Bash.
curl_with_auth() {
    if [[ -n "${OLLAYA_API_KEY:-}" ]]; then
        curl -sS "$@" --header "Authorization: Bearer ${OLLAYA_API_KEY}"
    else
        curl -sS "$@"
    fi
}

# Capture the HTTP status without curl -f so API error JSON remains available.
http_get() {
    local url="$1" output="$2" timeout_seconds="$3"
    curl_with_auth \
        --connect-timeout "$CONNECT_TIMEOUT_SECONDS" \
        --max-time "$timeout_seconds" \
        --output "$output" \
        --write-out '%{http_code}' \
        "$url"
}

http_post_json() {
    local url="$1" body="$2" output="$3" timeout_seconds="$4"
    curl_with_auth \
        --connect-timeout "$CONNECT_TIMEOUT_SECONDS" \
        --max-time "$timeout_seconds" \
        --header 'Content-Type: application/json' \
        --data-binary "$body" \
        --output "$output" \
        --write-out '%{http_code}' \
        "$url"
}

show_api_error() {
    local body_file="$1"
    if jq -e . "$body_file" >/dev/null 2>&1; then
        jq -r '"code: \(.code // "unknown")", "error: \(.error // .message // "no error message")"' "$body_file" >&2
        jq -c '.detail // empty' "$body_file" >&2 || true
    else
        head -c 1200 "$body_file" >&2 || true
        printf '\n' >&2
    fi
}

section 'OLLAYA CONFIGURATION'
info "Base URL: ${OLLAYA_URL}"
info "Model:    ${OLLAYA_MODEL}"
info "Timeout:  ${INFERENCE_TIMEOUT_SECONDS}s inference"

# 1. Liveness: GET / is the documented Ollaya liveness endpoint.
section 'LIVENESS'
LIVE_BODY="${TMP_DIR}/live.txt"
LIVE_STATUS="$(http_get "${OLLAYA_URL}/" "$LIVE_BODY" "$HTTP_TIMEOUT_SECONDS")" \
    || fail "Cannot connect to Ollaya at ${OLLAYA_URL}"
[[ "$LIVE_STATUS" == '200' ]] || fail "GET / returned HTTP ${LIVE_STATUS}"
LIVE_TEXT="$(cat "$LIVE_BODY")"
[[ "$LIVE_TEXT" == 'Ollaya is running' ]] \
    || fail "GET / returned an unexpected liveness body: ${LIVE_TEXT}"
pass 'Ollaya HTTP service is alive'

# 2. API version: confirms the native API is answering, not just an open port.
section 'API VERSION'
VERSION_BODY="${TMP_DIR}/version.json"
VERSION_STATUS="$(http_get "${OLLAYA_URL}/api/version" "$VERSION_BODY" "$HTTP_TIMEOUT_SECONDS")" \
    || fail 'Could not retrieve /api/version'
[[ "$VERSION_STATUS" == '200' ]] || {
    printf '[FAIL] GET /api/version returned HTTP %s\n' "$VERSION_STATUS" >&2
    show_api_error "$VERSION_BODY"
    exit 1
}
API_VERSION="$(jq -er '.version | select(type == "string" and length > 0)' "$VERSION_BODY" 2>/dev/null)" \
    || fail "/api/version did not return a JSON string field named version"
pass "Ollaya API version ${API_VERSION}"

# 3. Model presence: /api/tags is the authoritative local-model inventory.
section 'MODEL AVAILABILITY'
TAGS_BODY="${TMP_DIR}/tags.json"
TAGS_STATUS="$(http_get "${OLLAYA_URL}/api/tags" "$TAGS_BODY" "$HTTP_TIMEOUT_SECONDS")" \
    || fail 'Could not retrieve /api/tags'
[[ "$TAGS_STATUS" == '200' ]] || {
    printf '[FAIL] GET /api/tags returned HTTP %s\n' "$TAGS_STATUS" >&2
    show_api_error "$TAGS_BODY"
    exit 1
}
jq -e '.models | type == "array"' "$TAGS_BODY" >/dev/null 2>&1 \
    || fail '/api/tags did not return a models array'

# Ollaya canonicalizes an untagged name to :latest.
CANONICAL_MODEL="$OLLAYA_MODEL"
if [[ "$CANONICAL_MODEL" != *:* ]]; then
    CANONICAL_MODEL="${CANONICAL_MODEL}:latest"
fi
if ! jq -e --arg model "$CANONICAL_MODEL" '
    any(.models[]?; ((.name // "") | ascii_downcase) == ($model | ascii_downcase))
' "$TAGS_BODY" >/dev/null; then
    printf '[FAIL] Model %s is not installed locally. Models reported by Ollaya:\n' "$CANONICAL_MODEL" >&2
    jq -r '.models[]?.name // empty' "$TAGS_BODY" >&2
    exit 1
fi
pass "Model ${CANONICAL_MODEL} is installed locally"

# 4. Real inference. Ollaya's /api/decide does not accept a `prompt` field.
# Its contract requires model + state + typed questions.
section 'DECISION INFERENCE'
REQUEST_BODY="$(jq -cn \
    --arg model "$OLLAYA_MODEL" \
    --arg keep_alive "$OLLAYA_KEEP_ALIVE" \
    '{
      model: $model,
      state: "Ollaya health check probe: this is a synthetic service-readiness test, not a request for assistance.",
      questions: {
        health: {
          type: "choice",
          instructions: "Classify the supplied state.",
          criteria: {
            health_check: "A synthetic service-readiness probe.",
            user_request: "A real request for assistance."
          }
        }
      },
      keep_alive: $keep_alive
    }')"

printf 'Exact typed request:\n'
printf '%s\n' "$REQUEST_BODY" | jq .

DECIDE_BODY="${TMP_DIR}/decide.json"
DECIDE_STATUS="$(http_post_json \
    "${OLLAYA_URL}/api/decide" \
    "$REQUEST_BODY" \
    "$DECIDE_BODY" \
    "$INFERENCE_TIMEOUT_SECONDS")" \
    || fail "Inference request could not reach Ollaya (connectivity or timeout; limit ${INFERENCE_TIMEOUT_SECONDS}s)"

[[ "$DECIDE_STATUS" == '200' ]] || {
    printf '[FAIL] POST /api/decide returned HTTP %s\n' "$DECIDE_STATUS" >&2
    show_api_error "$DECIDE_BODY"
    exit 1
}

jq -e '
    .done_reason == "decide" and
    .state_truncated == false and
    (.model | type == "string" and length > 0) and
    (.answers.health.type == "choice") and
    (.answers.health.choice == "health_check" or .answers.health.choice == "user_request") and
    (.answers.health.confidence | type == "number" and . >= 0 and . <= 1) and
    (.answers.health.probabilities | type == "object") and
    (.usage.input_tokens | type == "number" and . > 0) and
    (.usage.output_tokens | type == "number" and . == 0)
' "$DECIDE_BODY" >/dev/null 2>&1 || {
    printf '[FAIL] Ollaya returned HTTP 200 but the body was not a valid completed typed decision:\n' >&2
    jq . "$DECIDE_BODY" >&2 2>/dev/null || cat "$DECIDE_BODY" >&2
    exit 1
}

ANSWERED_MODEL="$(jq -r '.model' "$DECIDE_BODY")"
CHOICE="$(jq -r '.answers.health.choice' "$DECIDE_BODY")"
CONFIDENCE="$(jq -r '.answers.health.confidence' "$DECIDE_BODY")"
INPUT_TOKENS="$(jq -r '.usage.input_tokens' "$DECIDE_BODY")"
TOTAL_DURATION_NS="$(jq -r '.total_duration // 0' "$DECIDE_BODY")"
LOAD_DURATION_NS="$(jq -r '.load_duration // 0' "$DECIDE_BODY")"
EVAL_DURATION_NS="$(jq -r '.eval_duration // 0' "$DECIDE_BODY")"

pass 'POST /api/decide completed a typed inference'
info "Answering model: ${ANSWERED_MODEL}"
info "Health probe choice: ${CHOICE} (confidence ${CONFIDENCE})"
info "Input tokens: ${INPUT_TOKENS}"
info "Total duration: $((TOTAL_DURATION_NS / 1000000)) ms"
info "Load duration:  $((LOAD_DURATION_NS / 1000000)) ms"
info "Eval duration:  $((EVAL_DURATION_NS / 1000000)) ms"

section 'RESULT'
pass 'Ollaya is live, the configured model is installed, and typed inference succeeds'
