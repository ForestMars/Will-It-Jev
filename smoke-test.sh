#!/usr/bin/env bash
#
# Router -> Tier 5 (Ollaya/Laya) -> ClickHouse end-to-end smoke test.
#
# Sends a raw routing prompt plus the typed `state`/`questions` payload required
# by Tier 5. Verifies that the router actually completed Tier 5 without a
# Tier 6 failover, returned the expected typed decision, and persisted the
# correlated route and task-tag records to ClickHouse's raw JSON tables.
#
# Requirements: Bash 3.2+, curl, jq
# Run from <repo>/scripts/ so .auth is found at the repository root.
#
# Optional environment:
#   ROUTER_URL                 default http://127.0.0.1:8080/v1/decisions
#   CLICKHOUSE_URL             default http://127.0.0.1:8123
#   #   AUTH_FILE                  default <repo>/.auth
#   ROUTER_TIMEOUT_SECONDS     default 150
#   ROUTE_WAIT_SECONDS         default 60
#   TAG_WAIT_SECONDS           default 60
#   REQUIRE_TAGGED             default 0 (1 also requires a successful tag)
#   EXPECTED_TAG_CATEGORY      default other_general_request
#
# Important: the checked-in Tier5Runner currently POSTs directly to Ollaya
# (default endpoint :11435/api/decide, model alias "laya"). It does not make a
# second HTTP call to a separate Jev daemon. This test therefore proves the
# router's Ollaya-backed Tier 5 path, as represented by the router's "jev"
# backend label; it does not prove a distinct Jev service was invoked.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8080/v1/decisions}"
CLICKHOUSE_URL="${CLICKHOUSE_URL:-http://127.0.0.1:8123}"
CLICKHOUSE_URL="${CLICKHOUSE_URL%/}"
# The Rust sink currently inserts into this database literally.
CLICKHOUSE_DATABASE="llm_observability"
ROUTER_TIMEOUT_SECONDS="${ROUTER_TIMEOUT_SECONDS:-150}"
ROUTE_WAIT_SECONDS="${ROUTE_WAIT_SECONDS:-60}"
TAG_WAIT_SECONDS="${TAG_WAIT_SECONDS:-60}"
REQUIRE_TAGGED="${REQUIRE_TAGGED:-0}"
EXPECTED_TAG_CATEGORY="${EXPECTED_TAG_CATEGORY:-other_general_request}"
AUTH_FILE="${AUTH_FILE:-${PROJECT_ROOT}/.auth}"

# Validate the database identifier before interpolating it into SQL.
case "$CLICKHOUSE_DATABASE" in
    ''|*[!A-Za-z0-9_]*)
        printf '[FAIL] CLICKHOUSE_DATABASE must contain only letters, digits, and underscores\n' >&2
        exit 1
        ;;
esac
case "$REQUIRE_TAGGED" in
    0|1) ;;
    *) printf '[FAIL] REQUIRE_TAGGED must be 0 or 1\n' >&2; exit 1 ;;
esac

if command -v uuidgen >/dev/null 2>&1; then
    SMOKE_KEY="smoke-$(uuidgen | tr '[:upper:]' '[:lower:]')"
else
    SMOKE_KEY="smoke-$(date +%s)-$$-${RANDOM}${RANDOM}"
fi

PROMPT="What is 18% of 450? [${SMOKE_KEY}]"
STATE="What is 18% of 450? [${SMOKE_KEY}]"
EXPECTED_ANSWER="81"

RESPONSE_FILE="$(mktemp)"
FAILURE_COUNT=0
FAILURE_MESSAGES=""
ROUTE_LINE=""
TAG_LINE=""
REQUEST_ID=""

cleanup() { rm -f "$RESPONSE_FILE"; }
trap cleanup EXIT

section() {
    printf '\n================================================================\n  %s\n================================================================\n' "$1"
}
info() { printf '[INFO] %s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail_now() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }
record_failure() {
    printf '[FAIL] %s\n' "$*" >&2
    FAILURE_COUNT=$((FAILURE_COUNT + 1))
    FAILURE_MESSAGES="${FAILURE_MESSAGES}\n- $*"
}

# macOS ships Bash 3.2. Avoid expanding an empty optional curl-argument array
# under `set -u`; construct the auth option inside this function instead.
ch_query() {
    local query="$1"
    if [[ -n "${CLICKHOUSE_USER:-}" ]]; then
        curl -sS --fail-with-body --connect-timeout 3 --max-time 15 \
            --user "${CLICKHOUSE_USER}:${CLICKHOUSE_PASSWORD:-}" \
            --get --data-urlencode "query=${query}" "${CLICKHOUSE_URL}/"
    else
        curl -sS --fail-with-body --connect-timeout 3 --max-time 15 \
            --get --data-urlencode "query=${query}" "${CLICKHOUSE_URL}/"
    fi
}

section 'SETUP'
for cmd in curl jq; do
    command -v "$cmd" >/dev/null 2>&1 || fail_now "Required command not found: ${cmd}"
done

if [[ -f "$AUTH_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "$AUTH_FILE"
    set +a
fi
if [[ -n "${CLICKHOUSE_USER:-}" && -z "${CLICKHOUSE_PASSWORD:-}" ]]; then
    fail_now "CLICKHOUSE_PASSWORD is set to empty; check ${AUTH_FILE}"
fi

info "Router:              ${ROUTER_URL}"
info "ClickHouse:          ${CLICKHOUSE_URL} (${CLICKHOUSE_DATABASE})"
info "Unique smoke key:    ${SMOKE_KEY}"
info "Require tagged:      ${REQUIRE_TAGGED}"
info "Expected tag:        ${EXPECTED_TAG_CATEGORY} (checked when tagged)"
info "Router timeout:      ${ROUTER_TIMEOUT_SECONDS}s"

ch_query "SELECT 1 FORMAT TabSeparated" >/dev/null \
    || fail_now 'ClickHouse is unreachable or authentication failed'
for table in route_records_raw task_tags_raw; do
    count="$(ch_query "SELECT count() FROM system.tables WHERE database='${CLICKHOUSE_DATABASE}' AND name='${table}' FORMAT TabSeparated")" \
        || fail_now "Could not inspect ClickHouse table ${CLICKHOUSE_DATABASE}.${table}"
    [[ "${count//[[:space:]]/}" == '1' ]] \
        || fail_now "Required raw table is missing: ${CLICKHOUSE_DATABASE}.${table}"
done
pass 'ClickHouse reachable; route_records_raw and task_tags_raw exist'

section 'ROUTER REQUEST'
# Keep `prompt` separate from `state`: the cascade sees the simple routing
# prompt, while Tier 5 receives the complete typed decision request. The
# current service handler explicitly supports this combination.
REQUEST_BODY="$(jq -cn --arg prompt "$PROMPT" --arg state "$STATE" '
    {
      prompt: $prompt,
      state: $state,
      questions: {
        answer: {
          type: "choice",
          instructions: "What is 18 percent of 450? Select the exact result.",
          criteria: {
            "81": "The exact arithmetic result is 81.",
            "other": "The arithmetic result is any value other than 81."
          }
        }
      }
    }
')"
printf 'Prompt used for cascade routing:\n  %s\n\nTyped state sent to Tier 5:\n  %s\n\nRequest body:\n' "$PROMPT" "$STATE"
printf '%s\n' "$REQUEST_BODY" | jq .

if ! HTTP_STATUS="$(curl -sS --connect-timeout 3 --max-time "$ROUTER_TIMEOUT_SECONDS" \
    --output "$RESPONSE_FILE" --write-out '%{http_code}' \
    --request POST --header 'Content-Type: application/json' \
    --data-binary "$REQUEST_BODY" "$ROUTER_URL")"; then
    fail_now "Router request could not complete (connection failure or timeout after ${ROUTER_TIMEOUT_SECONDS}s)"
fi
if [[ "$HTTP_STATUS" == '200' ]]; then
    pass 'Router returned HTTP 200'
else
    record_failure "Router returned HTTP ${HTTP_STATUS}"
fi
if jq -e . "$RESPONSE_FILE" >/dev/null 2>&1; then
    printf '\nRouter response:\n'
    jq . "$RESPONSE_FILE"
else
    cat "$RESPONSE_FILE"
    record_failure 'Router response was not valid JSON'
fi

section 'ROUTE RECORD'
# Query the raw JSON line emitted by the current Rust RouteRecord serializer.
# This avoids relying on flattened legacy columns in route_decisions views.
ROUTE_SQL="
SELECT line
FROM ${CLICKHOUSE_DATABASE}.route_records_raw
WHERE position(JSONExtractString(line, 'prompt_raw'), '${SMOKE_KEY}') > 0
ORDER BY _part DESC, _part_offset DESC
LIMIT 1
FORMAT TabSeparatedRaw
"
for ((i = 1; i <= ROUTE_WAIT_SECONDS; i++)); do
    ROUTE_LINE="$(ch_query "$ROUTE_SQL")" \
        || fail_now 'ClickHouse query failed while polling route_records_raw'
    [[ -n "$ROUTE_LINE" ]] && break
    printf '[INFO] Waiting for route record (%s/%ss)\r' "$i" "$ROUTE_WAIT_SECONDS"
    sleep 1
done
printf '\n'
[[ -n "$ROUTE_LINE" ]] || fail_now "No route_records_raw row found for ${SMOKE_KEY}"
jq -e . <<<"$ROUTE_LINE" >/dev/null 2>&1 \
    || fail_now 'route_records_raw.line is not valid JSON'

REQUEST_ID="$(jq -r '.request_id // empty' <<<"$ROUTE_LINE")"
[[ -n "$REQUEST_ID" ]] || fail_now 'Route record is missing request_id'

ROUTE_BACKEND="$(jq -r '.final_routing.backend // empty' <<<"$ROUTE_LINE")"
ROUTE_TIER="$(jq -r '.final_routing.tier // 0' <<<"$ROUTE_LINE")"
CASCADE_TIER="$(jq -r '.cascade_decision_tier // 0' <<<"$ROUTE_LINE")"
DOWNSTREAM_TIER="$(jq -r '.downstream_tier // empty' <<<"$ROUTE_LINE")"
DOWNSTREAM_OUTCOME="$(jq -r '.downstream_outcome // empty' <<<"$ROUTE_LINE")"
IS_FAILOVER="$(jq -r '.is_failover // false' <<<"$ROUTE_LINE")"
FAILURE_REASON="$(jq -r '.failure_reason // empty' <<<"$ROUTE_LINE")"
DOWNSTREAM_MS="$(jq -r '.downstream_invocation_latency_ms // 0' <<<"$ROUTE_LINE")"

jq -r '
  "Request ID:              \(.request_id)",
  "Prompt recorded:         \(.prompt_raw)",
  "Cascade decision tier:   \(.cascade_decision_tier)",
  "Final backend:           \(.final_routing.backend)",
  "Final tier:              \(.final_routing.tier)",
  "Downstream tier:         \(.downstream_tier // "null")",
  "Downstream outcome:      \(.downstream_outcome // "null")",
  "Failover:                \(.is_failover)",
  "Failure reason:          \(.failure_reason // "null")",
  "Downstream latency:      \(.downstream_invocation_latency_ms) ms"
' <<<"$ROUTE_LINE"
pass "Correlated route record found: ${REQUEST_ID}"

if [[ "$ROUTE_BACKEND" == 'jev' && "$ROUTE_TIER" == '5' && \
      "$DOWNSTREAM_TIER" == 'tier5' && "$DOWNSTREAM_OUTCOME" == 'success' && \
      "$IS_FAILOVER" == 'false' && -z "$FAILURE_REASON" ]]; then
    pass 'Router completed Tier 5 without Tier 6 failover'
else
    record_failure "Tier 5 did not complete cleanly (backend=${ROUTE_BACKEND:-missing}, tier=${ROUTE_TIER}, downstream=${DOWNSTREAM_TIER:-missing}/${DOWNSTREAM_OUTCOME:-missing}, failover=${IS_FAILOVER}, reason=${FAILURE_REASON:-none})"
fi

# The response must be a completed Ollaya decision, not merely HTTP 200 or a
# Tier 6 generative fallback. Ollaya's decision contract emits output_tokens=0.
if jq -e '
    .done_reason == "decide" and
    .state_truncated == false and
    (.model | type == "string" and length > 0) and
    .answers.answer.type == "choice" and
    (.answers.answer.choice | type == "string") and
    (.answers.answer.confidence | type == "number" and . >= 0 and . <= 1) and
    (.usage.input_tokens | type == "number" and . > 0) and
    (.usage.output_tokens | type == "number" and . == 0)
' "$RESPONSE_FILE" >/dev/null 2>&1; then
    ANSWERED_MODEL="$(jq -r '.model' "$RESPONSE_FILE")"
    ANSWER="$(jq -r '.answers.answer.choice' "$RESPONSE_FILE")"
    ANSWER_CONFIDENCE="$(jq -r '.answers.answer.confidence' "$RESPONSE_FILE")"
    info "Ollaya model:          ${ANSWERED_MODEL}"
    info "Answer confidence:    ${ANSWER_CONFIDENCE}"
    if [[ "$ANSWER" == "$EXPECTED_ANSWER" ]]; then
        pass "Ollaya returned the expected typed answer: ${ANSWER}"
    else
        record_failure "Ollaya returned '${ANSWER}', expected '${EXPECTED_ANSWER}'"
    fi
else
    record_failure 'Router response does not match Ollaya completed-decision response contract (possible Tier 6 fallback or invalid Tier 5 payload)'
fi

section 'TASK TAG RECORD'
TAG_SQL="
SELECT line
FROM ${CLICKHOUSE_DATABASE}.task_tags_raw
WHERE JSONExtractString(line, 'request_id') = '${REQUEST_ID}'
ORDER BY _part DESC, _part_offset DESC
LIMIT 1
FORMAT TabSeparatedRaw
"
for ((i = 1; i <= TAG_WAIT_SECONDS; i++)); do
    TAG_LINE="$(ch_query "$TAG_SQL")" \
        || fail_now 'ClickHouse query failed while polling task_tags_raw'
    [[ -n "$TAG_LINE" ]] && break
    printf '[INFO] Waiting for task-tag record (%s/%ss)\r' "$i" "$TAG_WAIT_SECONDS"
    sleep 1
done
printf '\n'

if [[ -n "$TAG_LINE" ]]; then
    jq -e . <<<"$TAG_LINE" >/dev/null 2>&1 \
        || fail_now 'task_tags_raw.line is not valid JSON'
    TAG_STATUS="$(jq -r '.result.status // "unknown"' <<<"$TAG_LINE")"
    TAG_CATEGORY="$(jq -r '.result.primary.category_id // empty' <<<"$TAG_LINE")"
    TAG_REASON="$(jq -r '.result.reason // empty' <<<"$TAG_LINE")"
    info "Task tag request ID:   $(jq -r '.request_id' <<<"$TAG_LINE")"
    info "Tagger model:          $(jq -r '.tagger_model // "unknown"' <<<"$TAG_LINE")"
    info "Tag status:            ${TAG_STATUS}"
    info "Primary category:      ${TAG_CATEGORY:-none}"
    [[ -n "$TAG_REASON" ]] && info "Untagged reason:       ${TAG_REASON}"

    if [[ "$(jq -r '.request_id' <<<"$TAG_LINE")" != "$REQUEST_ID" ]]; then
        record_failure 'Task-tag record request_id does not match route record'
    else
        pass 'Task-tag record correlated to the same router request_id'
    fi

    if [[ "$TAG_STATUS" == 'tagged' ]]; then
        if [[ "$TAG_CATEGORY" == "$EXPECTED_TAG_CATEGORY" ]]; then
            pass "Tagger assigned expected category: ${EXPECTED_TAG_CATEGORY}"
        else
            warn "Tagger category is '${TAG_CATEGORY}', expected '${EXPECTED_TAG_CATEGORY}'"
            [[ "$REQUIRE_TAGGED" == '1' ]] && record_failure "Unexpected tag category '${TAG_CATEGORY}'"
        fi
    else
        warn "Tag record persisted with status=${TAG_STATUS}${TAG_REASON:+, reason=${TAG_REASON}}"
        [[ "$REQUIRE_TAGGED" == '1' ]] && record_failure "Tagger did not produce tagged status (status=${TAG_STATUS})"
    fi
else
    record_failure "No task_tags_raw row found for request ${REQUEST_ID} within ${TAG_WAIT_SECONDS}s"
fi

section 'RESULT'
if [[ "$FAILURE_COUNT" -eq 0 ]]; then
    pass 'Router selected and completed the Ollaya-backed Tier 5 path'
    pass 'Typed decision response and ClickHouse route/tag correlation verified'
    exit 0
fi

printf '[FAIL] Smoke test completed with %s required failure(s):%b\n' \
    "$FAILURE_COUNT" "$FAILURE_MESSAGES" >&2
exit 1
