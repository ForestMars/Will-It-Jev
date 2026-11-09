-- 006_tier_instance_identity.down.sql
--
-- Restore the 005 task-tag projection before dropping the two instance fields.

DROP TABLE IF EXISTS llm_observability.task_tags_mv;

CREATE MATERIALIZED VIEW llm_observability.task_tags_mv
TO llm_observability.task_tags AS
SELECT
    parseDateTime64BestEffort(JSONExtractString(line, 'timestamp'), 6, 'UTC') AS timestamp,
    toUUID(JSONExtractString(line, 'request_id')) AS request_id,
    JSONExtractString(line, 'taxonomy_version') AS taxonomy_version,
    JSONExtractString(line, 'tagger_model') AS tagger_model,
    JSONExtractString(line, 'tagger_version') AS tagger_version,

    JSONExtractFloat(line, 'latency_ms') AS latency_ms,
    JSONExtractFloat(line, 'queue_wait_ms') AS queue_wait_ms,

    JSONExtractString(line, 'result', 'status') AS tag_status,
    JSONExtractString(line, 'result', 'primary', 'category_id') AS primary_category,
    JSONExtract(line, 'result', 'primary', 'score', 'Nullable(Float32)') AS primary_score,
    JSONExtractString(line, 'result', 'nearest', 'category_id') AS nearest_category,
    JSONExtract(line, 'result', 'nearest', 'score', 'Nullable(Float32)') AS nearest_score,
    JSONExtract(line, 'result', 'reason', 'Nullable(String)') AS untagged_reason,
    JSONExtract(line, 'result', 'embedding', 'Array(Float32)') AS embedding,

    JSONExtract(line, 'tagger_usage', 'input_tokens', 'Nullable(UInt64)') AS tagger_usage_input_tokens,
    JSONExtract(line, 'tagger_usage', 'output_tokens', 'Nullable(UInt64)') AS tagger_usage_output_tokens,
    JSONExtractBool(line, 'tagger_usage', 'estimated') AS tagger_usage_estimated,

    if(
        empty(JSONExtractString(line, 'task_id')),
        'tag',
        JSONExtractString(line, 'task_id')
    ) AS task_id,
    nullIf(JSONExtractString(line, 'output_kind'), '') AS output_kind,
    JSONExtract(line, 'resolving_tier', 'Nullable(UInt8)') AS resolving_tier,
    nullIf(JSONExtractString(line, 'decision_model_id'), '') AS decision_model_id,
    if(JSONHas(line, 'decision_scores'), JSONExtractRaw(line, 'decision_scores'), '[]') AS decision_scores_json,
    nullIf(JSONExtractString(line, 'resolving_tier_name'), '') AS resolving_tier_name
FROM llm_observability.task_tags_raw
WHERE isValidJSON(line)
  AND toUUIDOrNull(JSONExtractString(line, 'request_id')) IS NOT NULL
  AND parseDateTime64BestEffortOrNull(JSONExtractString(line, 'timestamp'), 6, 'UTC') IS NOT NULL;

ALTER TABLE llm_observability.task_tags
    DROP COLUMN IF EXISTS tier_id;

ALTER TABLE llm_observability.task_tags
    DROP COLUMN IF EXISTS tier_instance_id;
