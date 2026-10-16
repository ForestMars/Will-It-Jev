-- 001_route_decisions.smoke.sql
--
-- Smoke test for 001_route_decisions.sql. Run on a scratch or staging instance that has no
-- llm_observability database: the fixture rows stay in the tables until
-- 001_route_decisions.down.sql runs.
--
--   1. clickhouse-client --multiquery < 001_route_decisions.sql
--   2. Ingest the fixture twice, the second pass tests replay safety:
--        for i in 1 2; do
--          clickhouse-client --input_format_allow_errors_num=10 \
--            --query "INSERT INTO llm_observability.route_records_raw FORMAT JSONAsString" \
--            < 001_route_decisions.fixture.jsonl
--        done
--   3. clickhouse-client --multiquery < 001_route_decisions.smoke.sql
--        Every row must show pass = 1, including the final ALL_CHECKS_PASS row.
--   4. clickhouse-client --multiquery < 001_route_decisions.down.sql
--
-- The fixture holds 11 records covering every decision path (tier 1, tier 2, tier 3, the
-- tier 4 floor, failover, a failed tier 5 call, a NaN tier 3 score, an empty tier 2 input,
-- escapes and unicode) plus one poison line that the MV guard must reject. Request ids all
-- start with 00000000-0000-4000-8000- so the checks ignore any other data in the tables.

WITH
fixture AS (
    SELECT *, policy_version FROM llm_observability.route_decisions FINAL
    WHERE toString(request_id) LIKE '00000000-0000-4000-8000-%'
),
raw_lines AS (
    SELECT line FROM llm_observability.route_records_raw
    WHERE line = '{}' OR line LIKE '%"request_id":"00000000-0000-4000-8000-%'
),
checks AS (
    SELECT 'projected_rows' AS check_name, '11' AS expected, toString(count()) AS actual FROM fixture
    UNION ALL SELECT 'raw_rows_after_two_passes', '24', toString(count()) FROM raw_lines
    UNION ALL SELECT 'rejected_lines', '1', toString(uniqExact(line) - (SELECT count() FROM fixture)) FROM raw_lines
    UNION ALL SELECT 'decided_by_tier_1', '4', toString(countIf(decided_by_tier = 1)) FROM fixture
    UNION ALL SELECT 'decided_by_tier_2', '2', toString(countIf(decided_by_tier = 2)) FROM fixture
    UNION ALL SELECT 'decided_by_tier_3', '1', toString(countIf(decided_by_tier = 3)) FROM fixture
    UNION ALL SELECT 'decided_by_tier_4', '4', toString(countIf(decided_by_tier = 4)) FROM fixture
    UNION ALL SELECT 'decided_backend_jev', '5', toString(countIf(decided_backend = 'jev')) FROM fixture
    UNION ALL SELECT 'decided_backend_llm', '6', toString(countIf(decided_backend = 'llm')) FROM fixture
    UNION ALL SELECT 'invariants_tier1_floor_failover',  '0,0,0', concat(
        toString(countIf(decided_by_tier = 1 AND decided_backend != 'jev')), ',',
        toString(countIf(decided_by_tier = 4 AND decided_backend != 'llm')), ',',
        toString(countIf(is_failover AND decided_backend != 'jev'))) FROM fixture
    UNION ALL SELECT 'failover_final_to_decided', 'llm>jev', anyIf(concat(final_backend, '>', decided_backend), is_failover) FROM fixture
    UNION ALL SELECT 'policy_version', 'a1b2c3d4-99887766', any(policy_version) FROM fixture
    UNION ALL SELECT 'tier3_null_score_kept', '1', toString(countIf(tier3_score IS NULL AND tier3_reason IS NOT NULL)) FROM fixture
    UNION ALL SELECT 'unicode_newline_roundtrip', '1', toString(countIf(position(prompt_raw, char(10)) > 0 AND position(prompt_raw, '日本語') > 0)) FROM fixture
    UNION ALL SELECT 'timestamp_microseconds', '2026-10-16 14:03:22.123456', any(toString(timestamp)) FROM fixture
)
SELECT check_name, expected, actual, toUInt8(expected = actual) AS pass FROM checks
UNION ALL
SELECT 'ALL_CHECKS_PASS', '1', toString(min(toUInt8(expected = actual))), min(toUInt8(expected = actual)) FROM checks;
