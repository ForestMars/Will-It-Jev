-- 001_route_decisions.down.sql
--
-- Rollback for 001_route_decisions.sql. Destructive: drops the llm_observability database and
-- all captured route data.
--
-- Up refuses to run when the database already exists, so the database and everything in it was
-- created by 001, and dropping it restores the system state from before 001 exactly. The first
-- statement is a guard: if llm_observability holds any object this migration did not create (a
-- later migration, or anything added by hand), the rollback is refused before anything is
-- dropped. Roll back later migrations first. Safe to run against a partially applied 001 and
-- safe to run twice. SYNC makes the drop complete before the statement returns.

SELECT throwIf(count() > 0, 'rollback 001 refused: llm_observability contains objects this migration did not create')
FROM system.tables
WHERE database = 'llm_observability'
  AND name NOT IN ('route_records_raw', 'route_decisions', 'route_decisions_mv')
FORMAT Null;

DROP DATABASE IF EXISTS llm_observability SYNC;
