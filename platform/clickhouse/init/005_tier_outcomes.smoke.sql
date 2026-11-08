-- 005_tier_outcomes.smoke.sql
--
-- Run against a scratch instance after 001 to 005, having inserted
-- 005_tier_outcomes.fixture.jsonl into llm_observability.route_records_raw (one JSON
-- object per line in the `line` column).

WITH fixture AS (
    SELECT * FROM llm_observability.route_decisions FINAL
    WHERE request_id IN (
        toUUID('00000000-0000-4000-8000-0000000000e1'),
        toUUID('00000000-0000-4000-8000-0000000000e2')
    )
),
checks AS (
    SELECT 'outcomes_list_has_both_tiers' AS check_name, '1' AS expected,
        toString(countIf(JSONLength(tier_outcomes_json) = 2)) AS actual FROM fixture
    UNION ALL SELECT 'tier_without_ordinal_is_named', '1',
        toString(countIf(JSONExtractString(tier_outcomes_json, 2, 'tier') = 'triage')) FROM fixture
    UNION ALL SELECT 'final_tier_name_projected', '1',
        toString(countIf(final_tier_name = 'jev' AND decided_by_tier_name = 'triage')) FROM fixture
    UNION ALL SELECT 'pre_005_row_defaults', '1',
        toString(countIf(tier_outcomes_json = '[]' AND final_tier_name = 'llm')) FROM fixture
)
SELECT check_name, expected, actual FROM checks
UNION ALL
SELECT 'all_checks_pass', '1', toString(min(toUInt8(expected = actual))) FROM checks;
