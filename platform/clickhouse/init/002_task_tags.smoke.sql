-- 002_task_tags.smoke.sql
--
-- Run this against a scratch instance after 001_route_decisions.sql and 002_task_tags.sql.
-- It verifies that raw task-tag lines are parsed and mapped into the typed table.

WITH fixture AS (
    SELECT *, tag_status FROM llm_observability.task_tags FINAL
    WHERE request_id IN (
        toUUID('00000000-0000-4000-8000-000000000001'),
        toUUID('00000000-0000-4000-8000-000000000002'),
        toUUID('00000000-0000-4000-8000-000000000003')
    )
)
SELECT
    'tagged_rows' AS check_name,
    '1' AS expected,
    toString(countIf(tag_status = 'tagged')) AS actual
UNION ALL
SELECT
    'no_match_rows',
    '1',
    toString(countIf(tag_status = 'no_match'))
UNION ALL
SELECT
    'untagged_rows',
    '1',
    toString(countIf(tag_status = 'untagged'))
UNION ALL
SELECT
    'engineering_category',
    'engineering',
    any(primary_category)
FROM fixture
WHERE tag_status = 'tagged'
UNION ALL
SELECT
    'all_checks_pass',
    '1',
    toString(min(toUInt8(expected = actual)))
FROM (
    SELECT 'tagged_rows' AS check_name, '1' AS expected, toString(countIf(tag_status = 'tagged')) AS actual FROM fixture
    UNION ALL
    SELECT 'no_match_rows', '1', toString(countIf(tag_status = 'no_match')) FROM fixture
    UNION ALL
    SELECT 'untagged_rows', '1', toString(countIf(tag_status = 'untagged')) FROM fixture
);
