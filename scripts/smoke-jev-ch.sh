#!/usr/bin/env bash
#
# scripts/smoke-jev-ch.sh
#
# Smoke test: embed a unique key in the prompt, POST it to the router, then find
# the router's route_decisions row by searching prompt_raw for that key, and
# follow its router-generated request_id into task_tags.
#
# Requirements: bash, curl, jq

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8080/v1/decisions}"
CLICKHOUSE_URL="${CLICKHOUSE_URL:-http://127.0.0.1:8123}"
CLICKHOUSE_DATABASE="${CLICKHOUSE_DATABASE:-llm_observability}"
ROUTE_TABLE="${CLICKHOUSE_DATABASE}.route_decisions"
TAG_TABLE="${CLICKHOUSE_DATABASE}.task_tags"

ROUTE_WAIT_SECONDS="${ROUTE_WAIT_SECONDS:-60}"
TAG_WAIT_SECONDS="${TAG_WAIT_SECONDS:-45}"
ROUTER_TIMEOUT_SECONDS="${ROUTER_TIMEOUT_SECONDS:-60}"

# Unique key: only [a-z0-9-], so it is safe to embed in SQL and JSON.
if command -v uuidgen >/dev/null 2>&1; then
    SMOKE_KEY="smoke-$(uuidgen | tr '[:upper:]' '[:lower:]')"
else
    SMOKE_KEY="smoke-$(date +%s)-${RANDOM}${RANDOM}"
fi

# Key is appended so the leading text the router classifies stays unchanged.
PROMPT="What is 18% of 450? [${SMOKE_KEY}]"

AUTH_FILE="${PROJECT_ROOT}/.auth"
RESPONSE_FILE="$(mktemp)"
CH_CURL_ARGS=()

cleanup() { rm -f "$RESPONSE_FILE"; }
trap cleanup EXIT

fail()    { printf '[FAIL] %s\n' "$*" >&2; exit 1; }
info()    { printf '[INFO] %s\n' "$*"; }
pass()    { printf '[PASS] %s\n' "$*"; }
section() { printf '\n================================================================\n  %s\n================================================================\n' "$1"; }

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
section "SETUP"

for cmd in curl jq; do
    command -v "$cmd" >/dev/null 2>&1 || fail "Required command not found: $cmd"
done

[[ -f "$AUTH_FILE" ]] || fail "Missing credentials file: ${AUTH_FILE}"
set -a
# shellcheck disable=SC1090
source "$AUTH_FILE"
set +a
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER missing from .auth}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD missing from .auth}"
CH_CURL_ARGS=(--user "${CLICKHOUSE_USER}:${CLICKHOUSE_PASSWORD}")

# On a ClickHouse error, the error text is returned and the exit status is non-zero.
ch_query() {
    curl -sS --fail-with-body "${CH_CURL_ARGS[@]}" \
        --get --data-urlencode "query=$1" "${CLICKHOUSE_URL}/"
}

info "Router:      ${ROUTER_URL}"
info "ClickHouse:  ${CLICKHOUSE_URL} (${CLICKHOUSE_DATABASE})"
info "Unique key:  ${SMOKE_KEY}"

ch_query "SELECT 1 FORMAT TabSeparated" >/dev/null || fail "ClickHouse unreachable or auth failed"
for t in route_decisions task_tags; do
    n="$(ch_query "SELECT count() FROM system.tables WHERE database='${CLICKHOUSE_DATABASE}' AND name='${t}' FORMAT TabSeparated")" \
        || fail "Could not inspect table ${t}"
    [[ "${n//[[:space:]]/}" == "1" ]] || fail "Table missing: ${CLICKHOUSE_DATABASE}.${t}"
done
pass "ClickHouse reachable; route_decisions and task_tags exist"

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
section "ROUTER REQUEST"

# NOTE: this body is the one part not verified against the router's handler.
REQUEST_BODY="$(jq -cn --arg prompt "$PROMPT" '{prompt: $prompt}')"

printf 'Exact prompt sent to router:\n  %s\n\nExact request body:\n' "$PROMPT"
printf '%s\n' "$REQUEST_BODY" | jq .

HTTP_STATUS="$(curl -sS --max-time "$ROUTER_TIMEOUT_SECONDS" \
    --output "$RESPONSE_FILE" --write-out '%{http_code}' \
    --request POST --header 'Content-Type: application/json' \
    --data-binary "$REQUEST_BODY" "$ROUTER_URL")" || fail "Router HTTP request failed"

if [[ "$HTTP_STATUS" =~ ^2[0-9][0-9]$ ]]; then
    pass "Router HTTP ${HTTP_STATUS}"
else
    printf '[WARN] Router HTTP %s (continuing: failures must still be logged)\n' "$HTTP_STATUS"
fi
printf '\nRouter response:\n'
jq . "$RESPONSE_FILE" 2>/dev/null || cat "$RESPONSE_FILE"

# ---------------------------------------------------------------------------
# route_decisions: find by unique key in prompt_raw
# ---------------------------------------------------------------------------
section "ROUTE DECISION"

ROUTE_SQL="
SELECT toString(request_id) AS request_id, toString(timestamp) AS timestamp,
       policy_version, prompt_raw,
       tier1_confidence, tier1_decided, tier1_reason,
       tier2_p_jev, tier2_p_llm, tier2_reason,
       tier3_score, tier3_reason, tier4_score, tier4_reason,
       decided_by_tier, decided_backend, final_backend,
       downstream_tier, downstream_outcome,
       router_classification_latency_us,
       downstream_invocation_latency_ms,
       request_latency_primary_ms, request_latency_failover_ms,
       is_failover, failure_reason, total_latency_us, sample_prob
FROM ${ROUTE_TABLE}
WHERE position(prompt_raw, '${SMOKE_KEY}') > 0
ORDER BY timestamp DESC
LIMIT 1
FORMAT JSONEachRow
"

ROUTE_JSON=""
printf '[INFO] Polling route_decisions for key (up to %ss) ' "$ROUTE_WAIT_SECONDS"
for ((i = 1; i <= ROUTE_WAIT_SECONDS; i++)); do
    ROUTE_JSON="$(ch_query "$ROUTE_SQL")" || { printf '\n'; fail "route_decisions query error: ${ROUTE_JSON}"; }
    [[ -n "$ROUTE_JSON" ]] && break
    printf '.'
    sleep 1
done
printf '\n'
[[ -n "$ROUTE_JSON" ]] || fail "No route_decisions row contains key ${SMOKE_KEY}"

REQUEST_ID="$(jq -r '.request_id' <<<"$ROUTE_JSON")"
pass "Route record found; router-generated request ID: ${REQUEST_ID}"

jq -r '
    "  Request ID:               \(.request_id)",
    "  Timestamp:                \(.timestamp)",
    "  Policy version:           \(.policy_version)",
    "  Prompt (as recorded):     \(.prompt_raw)",
    "  Decided by tier:          \(.decided_by_tier)",
    "  Decided backend:          \(.decided_backend)",
    "  Final backend:            \(.final_backend)",
    "  Downstream tier:          \(.downstream_tier // "null")",
    "  Downstream outcome:       \(.downstream_outcome // "null")",
    "  Failover:                 \(.is_failover)",
    "  Failure reason:           \(.failure_reason // "null")",
    "  Tier 1:                   conf=\(.tier1_confidence) decided=\(.tier1_decided) reason=\(.tier1_reason)",
    "  Tier 2:                   p_jev=\(.tier2_p_jev // "null") p_llm=\(.tier2_p_llm // "null") reason=\(.tier2_reason // "null")",
    "  Tier 3:                   score=\(.tier3_score // "null") reason=\(.tier3_reason // "null")",
    "  Tier 4:                   score=\(.tier4_score // "null") reason=\(.tier4_reason // "null")",
    "  Classification latency:   \(.router_classification_latency_us) us",
    "  Downstream latency:       \(.downstream_invocation_latency_ms) ms",
    "  Primary latency:          \(.request_latency_primary_ms // "null") ms",
    "  Failover latency:         \(.request_latency_failover_ms // "null") ms",
    "  Total latency:            \(.total_latency_us) us",
    "  Sample probability:       \(.sample_prob)"
' <<<"$ROUTE_JSON"

# ---------------------------------------------------------------------------
# task_tags: SELECT * because its columns have not been verified
# ---------------------------------------------------------------------------
section "TASK TAGGING"

TAG_SQL="SELECT * FROM ${TAG_TABLE} WHERE request_id = toUUID('${REQUEST_ID}') LIMIT 1 FORMAT Vertical"

TAG_OUT=""
printf '[INFO] Polling task_tags (up to %ss) ' "$TAG_WAIT_SECONDS"
for ((i = 1; i <= TAG_WAIT_SECONDS; i++)); do
    TAG_OUT="$(ch_query "$TAG_SQL")" || { printf '\n'; fail "task_tags query error: ${TAG_OUT}"; }
    [[ -n "$TAG_OUT" ]] && break
    printf '.'
    sleep 1
done
printf '\n'

section "RESULT"
if [[ -n "$TAG_OUT" ]]; then
    pass "Task-tag record found"
    printf '%s\n' "$TAG_OUT"
    pass "Router request, route_decisions, and task_tags all correlated"
else
    printf '[WARN] route_decisions verified; no task_tags row for %s within %ss\n' "$REQUEST_ID" "$TAG_WAIT_SECONDS"
    exit 2
fi
