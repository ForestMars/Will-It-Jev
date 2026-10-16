-- 002_task_tags.down.sql
--
-- Roll back the task-tag tables created by 002_task_tags.sql. This is intentionally
-- narrow: it drops only the task-tagging objects and refuses to run if the database
-- contains any unrelated objects added by later migrations.

SELECT throwIf(count() > 0, 'rollback 002 refused: llm_observability contains objects this migration did not create')
FROM system.tables
WHERE database = 'llm_observability'
  AND name NOT IN ('route_records_raw', 'route_decisions', 'route_decisions_mv', 'task_tags_raw', 'task_tags', 'task_tags_mv')
FORMAT Null;

DROP TABLE IF EXISTS llm_observability.task_tags_mv SYNC;
DROP TABLE IF EXISTS llm_observability.task_tags SYNC;
DROP TABLE IF EXISTS llm_observability.task_tags_raw SYNC;
