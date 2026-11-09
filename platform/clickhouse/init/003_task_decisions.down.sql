-- 003_task_decisions.down.sql
--
-- Roll back 003_task_decisions.sql: restore the pre-003 materialized view
-- definitions (copied from 001 and 002), then drop the columns 003 added.
--
-- Narrow on purpose. The raw landing tables are untouched. The dropped columns
-- (route_decisions.tasks_json and the five task_tags columns) lose whatever 003
-- projected into them; rolling forward again re-projects new raw rows only.
--
-- Refuses to run while 004_tier_names is applied, because 004's views and columns
-- build on 003. Roll back 004 first. A refusal changes nothing.

SELECT throwIf(count() > 0, 'rollback 003 refused: 004_tier_names is applied, roll it back first')
FROM system.columns
WHERE database = 'llm_observability'
  AND name IN ('decided_by_tier_name', 'resolving_tier_name')
FORMAT Null;

DROP TABLE IF EXISTS llm_observability.route_decisions_mv;

CREATE MATERIALIZED VIEW llm_observability.route_decisions_mv
TO llm_observability.route_decisions AS
SELECT
    parseDateTime64BestEffort(JSONExtractString(line, 'timestamp'), 6, 'UTC')           AS timestamp,
    toUUID(JSONExtractString(line, 'request_id'))                                        AS request_id,

    JSONExtractString(line, 'hashes', 'tier1_toml')                                      AS tier1_toml_hash,
    JSONExtractString(line, 'hashes', 'tier2_bin')                                       AS tier2_bin_hash,

    JSONExtractString(line, 'prompt_raw')                                                AS prompt_raw,

    JSONExtractFloat(line, 'tier1_outcome', 'confidence')                                AS tier1_confidence,
    JSONExtractBool(line, 'tier1_outcome', 'decided')                                    AS tier1_decided,
    JSONExtractString(line, 'tier1_outcome', 'reason')                                   AS tier1_reason,

    JSONExtract(line, 'tier2_outcome', 'p_jev', 'Nullable(Float32)')                     AS tier2_p_jev,
    JSONExtract(line, 'tier2_outcome', 'p_llm', 'Nullable(Float32)')                     AS tier2_p_llm,
    JSONExtract(line, 'tier2_outcome', 'raw_probabilities', 'Array(Float32)')            AS tier2_raw_probabilities,
    JSONExtract(line, 'tier2_outcome', 'reason', 'Nullable(String)')                     AS tier2_reason,

    JSONExtract(line, 'tier3_outcome', 'score', 'Nullable(Float32)')                     AS tier3_score,
    JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)')                     AS tier3_reason,
    JSONExtract(line, 'tier4_outcome', 'score', 'Nullable(Float32)')                     AS tier4_score,
    JSONExtract(line, 'tier4_outcome', 'reason', 'Nullable(String)')                     AS tier4_reason,

    if(
        JSONHas(line, 'cascade_decision_tier'),
        toUInt8(JSONExtractUInt(line, 'cascade_decision_tier')),
        if(
        JSONHas(line, 'tier4_outcome'),
        multiIf(
            JSONExtract(line, 'tier4_outcome', 'reason', 'Nullable(String)') = 'resolved', 4,
            JSONExtract(line, 'tier4_outcome', 'reason', 'Nullable(String)') IS NOT NULL, 6,
            JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)') = 'resolved', 3,
            JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)') IS NOT NULL, 6,
            JSONExtract(line, 'tier2_outcome', 'reason', 'Nullable(String)') = 'resolved', 2,
            JSONExtract(line, 'tier2_outcome', 'reason', 'Nullable(String)') IS NOT NULL, 6,
            1
        ),
        multiIf(
            JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)') = 'resolved', 3,
            JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)') IS NOT NULL, 4,
            JSONExtract(line, 'tier2_outcome', 'reason', 'Nullable(String)') = 'resolved', 2,
            1
        )
        )
    )                                                                                    AS decided_by_tier,
    if(JSONExtractBool(line, 'is_failover'), 'jev', JSONExtractString(line, 'final_routing', 'backend'))
                                                                                         AS decided_backend,
    toUInt64(JSONExtractUInt(line, 'router_classification_latency_us'))                  AS router_classification_latency_us,

    JSONExtractString(line, 'final_routing', 'backend')                                  AS final_backend,
    JSONExtract(line, 'downstream_tier', 'Nullable(String)')                             AS downstream_tier,
    JSONExtract(line, 'downstream_outcome', 'Nullable(String)')                          AS downstream_outcome,
    JSONExtractFloat(line, 'downstream_invocation_latency_ms')                           AS downstream_invocation_latency_ms,
    JSONExtract(line, 'request_latency_primary_ms', 'Nullable(Float64)')                 AS request_latency_primary_ms,
    JSONExtract(line, 'request_latency_failover_ms', 'Nullable(Float64)')                AS request_latency_failover_ms,
    JSONExtractBool(line, 'is_failover')                                                 AS is_failover,
    JSONExtract(line, 'failure_reason', 'Nullable(String)')                              AS failure_reason,
    toUInt64(JSONExtractUInt(line, 'final_routing', 'latency_us'))                       AS total_latency_us,

    JSONExtractFloat(line, 'sampling', 'sample_prob')                                    AS sample_prob
FROM llm_observability.route_records_raw
-- Lines that fail these checks stay in route_records_raw and are skipped here,
-- so one malformed line cannot fail the insert block for the good lines around it.
WHERE isValidJSON(line)
  AND toUUIDOrNull(JSONExtractString(line, 'request_id')) IS NOT NULL
  AND parseDateTime64BestEffortOrNull(JSONExtractString(line, 'timestamp'), 6, 'UTC') IS NOT NULL;

DROP TABLE IF EXISTS llm_observability.task_tags_mv;

CREATE MATERIALIZED VIEW llm_observability.task_tags_mv
TO llm_observability.task_tags AS
SELECT
    parseDateTime64BestEffort(JSONExtractString(line, 'timestamp'), 6, 'UTC')                    AS timestamp,
    toUUID(JSONExtractString(line, 'request_id'))                                                 AS request_id,

    JSONExtractString(line, 'taxonomy_version')                                                  AS taxonomy_version,
    JSONExtractString(line, 'tagger_model')                                                      AS tagger_model,
    JSONExtractString(line, 'tagger_version')                                                    AS tagger_version,

    JSONExtractFloat(line, 'latency_ms')                                                         AS latency_ms,
    JSONExtractFloat(line, 'queue_wait_ms')                                                      AS queue_wait_ms,

    JSONExtractString(line, 'result', 'status')                                                  AS tag_status,
    JSONExtractString(line, 'result', 'primary', 'category_id')                                  AS primary_category,
    JSONExtract(line, 'result', 'primary', 'score', 'Nullable(Float32)')                        AS primary_score,
    JSONExtractString(line, 'result', 'nearest', 'category_id')                                 AS nearest_category,
    JSONExtract(line, 'result', 'nearest', 'score', 'Nullable(Float32)')                       AS nearest_score,
    JSONExtract(line, 'result', 'reason', 'Nullable(String)')                                   AS untagged_reason,
    JSONExtract(line, 'result', 'embedding', 'Array(Float32)')                                  AS embedding,

    JSONExtract(line, 'tagger_usage', 'input_tokens', 'Nullable(UInt64)')                       AS tagger_usage_input_tokens,
    JSONExtract(line, 'tagger_usage', 'output_tokens', 'Nullable(UInt64)')                      AS tagger_usage_output_tokens,
    JSONExtractBool(line, 'tagger_usage', 'estimated')                                           AS tagger_usage_estimated
FROM llm_observability.task_tags_raw
WHERE isValidJSON(line)
  AND toUUIDOrNull(JSONExtractString(line, 'request_id')) IS NOT NULL
  AND parseDateTime64BestEffortOrNull(JSONExtractString(line, 'timestamp'), 6, 'UTC') IS NOT NULL;

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
