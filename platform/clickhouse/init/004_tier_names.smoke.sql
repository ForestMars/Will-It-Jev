-- 004_tier_names.smoke.sql
--
-- Run against a scratch instance after 001 to 004, having inserted
-- 004_tier_names.fixture.jsonl into llm_observability.route_records_raw (one JSON
-- object per line in the `line` column). Verifies that tier names are projected and
-- that a row written before 004 gets the name derived from its numeric tier.

WITH fixture AS (
    SELECT * FROM llm_observability.route_decisions FINAL
    WHERE request_id IN (
        toUUID('00000000-0000-4000-8000-0000000000a1'),
        toUUID('00000000-0000-4000-8000-0000000000a2'),
        toUUID('00000000-0000-4000-8000-0000000000a3')
    )
),
checks AS (
    SELECT 'rules_named' AS check_name, '1' AS expected,
        toString(countIf(decided_by_tier_name = 'rules' AND decided_by_tier = 1)) AS actual FROM fixture
    UNION ALL SELECT 'new_tier_named_without_ordinal', '1',
        toString(countIf(decided_by_tier_name = 'triage' AND decided_by_tier = 0)) FROM fixture
    UNION ALL SELECT 'pre_004_row_name_derived_from_ordinal', '1',
        toString(countIf(decided_by_tier_name = 'modernbert' AND decided_by_tier = 3)) FROM fixture
    UNION ALL SELECT 'task_json_has_tier_name', '1',
        toString(countIf(JSONExtractString(tasks_json, 1, 'resolving_tier_name') = 'rules')) FROM fixture
)
SELECT check_name, expected, actual FROM checks
UNION ALL
SELECT 'all_checks_pass', '1', toString(min(toUInt8(expected = actual))) FROM checks;
