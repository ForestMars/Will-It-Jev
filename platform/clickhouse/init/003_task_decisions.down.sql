-- 003_task_decisions.down.sql
--
-- Revert task-level execution provenance.
-- Restore the pre-003 materialized view definitions.

DROP TABLE IF EXISTS llm_observability.route_decisions_mv;

CREATE MATERIALIZED VIEW llm_observability.route_decisions_mv
TO llm_observability.route_decisions AS
SELECT
    -- RESTORE THE COMPLETE SELECT FROM THE PRE-003 MIGRATION HERE
;

DROP TABLE IF EXISTS llm_observability.task_tags_mv;

CREATE MATERIALIZED VIEW llm_observability.task_tags_mv
TO llm_observability.task_tags AS
SELECT
    -- RESTORE THE COMPLETE SELECT FROM THE PRE-003 MIGRATION HERE
;

ALTER TABLE llm_observability.route_decisions
    DROP COLUMN IF EXISTS tasks_json;

ALTER TABLE llm_observability.task_tags
    DROP COLUMN IF EXISTS task_id;

ALTER TABLE llm_observability.task_tags
    DROP COLUMN IF EXISTS output_kind;

ALTER TABLE llm_observability.task_tags
    DROP COLUMN IF EXISTS resolving_tier;

ALTER TABLE llm_observability.task_tags
    DROP COLUMN IF EXISTS decision_model_id;

ALTER TABLE llm_observability.task_tags
    DROP COLUMN IF EXISTS decision_scores_json;