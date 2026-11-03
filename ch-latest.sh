#!/usr/bin/env bash
# Show the newest rows from the ClickHouse raw tables, pretty-printed with jq.
# Usage: ./ch-latest.sh [N] [REQUEST_ID]
#   N           rows per table (default 1)
#   REQUEST_ID  only show rows whose request_id matches (looks at the newest 200 rows)
set -euo pipefail

[ -f .auth ] && { set -a; . ./.auth; set +a; }
URL="${CLICKHOUSE_URL:-http://127.0.0.1:8123}"
DB="llm_observability"
N="${1:-1}"
RID="${2:-}"
[ -n "$RID" ] && N=200

# If a column holds a JSON string (the raw tables store the JSONL line), expand it.
EXPAND='with_entries(.value |= (. as $v | if ($v|type)=="string" and ($v|test("^\\s*[\\[{]")) then (try ($v|fromjson) catch $v) else $v end))'
FILTER="$EXPAND"
[ -n "$RID" ] && FILTER="$EXPAND | select(.. | .request_id? == \"$RID\")"

for table in task_tags_raw route_records_raw task_tags route_decisions; do
  echo "=== $DB.$table ==="
  curl -sS --fail-with-body -u "${CLICKHOUSE_USER}:${CLICKHOUSE_PASSWORD}" -G "$URL/" \
    --data-urlencode "query=SELECT * FROM $DB.$table ORDER BY _part DESC, _part_offset DESC LIMIT $N FORMAT JSONEachRow" \
    | jq "$FILTER"
done
