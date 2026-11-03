#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${PROJECT_ROOT}/.auth" ]] && source "${PROJECT_ROOT}/.auth"

CLICKHOUSE_URL="${CLICKHOUSE_URL:-http://127.0.0.1:8123}"
CH_ARGS=(--user "${CLICKHOUSE_USER:-app}:${CLICKHOUSE_PASSWORD:-app}")

echo "=== 1. ROW COUNT ==="
curl -sS "${CH_ARGS[@]}" "${CLICKHOUSE_URL}/?query=SELECT+count()+FROM+llm_observability.route_decisions"
echo

echo -e "\n=== 2. RECENT ROWS ==="
curl -sS "${CH_ARGS[@]}" "${CLICKHOUSE_URL}/?query=SELECT+timestamp,+decided_backend,+final_backend,+failure_reason+FROM+llm_observability.route_decisions+ORDER+BY+timestamp+DESC+LIMIT+10+FORMAT+PrettyCompact"
echo
