-- 006_tier_instance_identity.smoke.sql
SELECT throwIf(
    countIf(name = 'tier_instance_id') != 1 OR countIf(name = 'tier_id') != 1,
    'tier instance identity columns are missing from task_tags'
)
FROM system.columns
WHERE database = 'llm_observability'
  AND table = 'task_tags';
