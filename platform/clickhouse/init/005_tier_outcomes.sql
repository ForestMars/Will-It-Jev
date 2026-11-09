-- 005_tier_outcomes.sql
--
-- Name-keyed per-tier outcomes on the route record.
--
--   tier_outcomes_json  every tier that ran for the route task, by name, in
--                       execution order (JSON array). Unlike tier1_outcome to
--                       tier4_outcome it covers tiers that have no legacy ordinal.
--   final_tier_name     the tier or downstream backend that served the request
--                       (a tier name, or jev / llm after dispatch).
--
-- Additive only. The numbered columns keep describing the original four tiers.
-- Raw landing tables stay append-only. The route view is recreated so new raw rows
-- project the new columns; rows written before this migration get '[]' and the
-- final backend. Run after 004_tier_names.sql.

ALTER TABLE llm_observability.route_decisions
    ADD COLUMN IF NOT EXISTS tier_outcomes_json String DEFAULT '[]';

ALTER TABLE llm_observability.route_decisions
    ADD COLUMN IF NOT EXISTS final_tier_name LowCardinality(String) DEFAULT '';

DROP TABLE IF EXISTS llm_observability.route_decisions_mv;

CREATE MATERIALIZED VIEW llm_observability.route_decisions_mv
TO llm_observability.route_decisions AS
SELECT
    parseDateTime64BestEffort(JSONExtractString(line, 'timestamp'), 6, 'UTC') AS timestamp,
    toUUID(JSONExtractString(line, 'request_id')) AS request_id,
    JSONExtractString(line, 'hashes', 'tier1_toml') AS tier1_toml_hash,
    JSONExtractString(line, 'hashes', 'tier2_bin') AS tier2_bin_hash,
    JSONExtractString(line, 'prompt_raw') AS prompt_raw,
    JSONExtractFloat(line, 'tier1_outcome', 'confidence') AS tier1_confidence,
    JSONExtractBool(line, 'tier1_outcome', 'decided') AS tier1_decided,
    JSONExtractString(line, 'tier1_outcome', 'reason') AS tier1_reason,
    JSONExtract(line, 'tier2_outcome', 'p_jev', 'Nullable(Float32)') AS tier2_p_jev,
    JSONExtract(line, 'tier2_outcome', 'p_llm', 'Nullable(Float32)') AS tier2_p_llm,
    JSONExtract(line, 'tier2_outcome', 'raw_probabilities', 'Array(Float32)') AS tier2_raw_probabilities,
    JSONExtract(line, 'tier2_outcome', 'reason', 'Nullable(String)') AS tier2_reason,
    JSONExtract(line, 'tier3_outcome', 'score', 'Nullable(Float32)') AS tier3_score,
    JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)') AS tier3_reason,
    JSONExtract(line, 'tier4_outcome', 'score', 'Nullable(Float32)') AS tier4_score,
    JSONExtract(line, 'tier4_outcome', 'reason', 'Nullable(String)') AS tier4_reason,
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
    ) AS decided_by_tier,
    if(JSONExtractBool(line, 'is_failover'), 'jev', JSONExtractString(line, 'final_routing', 'backend')) AS decided_backend,
    toUInt64(JSONExtractUInt(line, 'router_classification_latency_us')) AS router_classification_latency_us,
    JSONExtractString(line, 'final_routing', 'backend') AS final_backend,
    JSONExtract(line, 'downstream_tier', 'Nullable(String)') AS downstream_tier,
    JSONExtract(line, 'downstream_outcome', 'Nullable(String)') AS downstream_outcome,
    JSONExtractFloat(line, 'downstream_invocation_latency_ms') AS downstream_invocation_latency_ms,
    JSONExtract(line, 'request_latency_primary_ms', 'Nullable(Float64)') AS request_latency_primary_ms,
    JSONExtract(line, 'request_latency_failover_ms', 'Nullable(Float64)') AS request_latency_failover_ms,
    JSONExtractBool(line, 'is_failover') AS is_failover,
    JSONExtract(line, 'failure_reason', 'Nullable(String)') AS failure_reason,
    toUInt64(JSONExtractUInt(line, 'final_routing', 'latency_us')) AS total_latency_us,
    JSONExtractFloat(line, 'sampling', 'sample_prob') AS sample_prob,
    if(JSONHas(line, 'tasks'), JSONExtractRaw(line, 'tasks'), '[]') AS tasks_json,
    -- New rows carry the tier name. Rows written before 004 only have the
    -- numeric decided_by_tier, so derive the name of the four original tiers.
    if(
        empty(JSONExtractString(line, 'decided_by_tier_name')),
        multiIf(
            decided_by_tier = 1, 'rules',
            decided_by_tier = 2, 'fasttext',
            decided_by_tier = 3, 'modernbert',
            decided_by_tier = 4, 'harrier',
            decided_by_tier = 6, 'fallback',
            ''
        ),
        JSONExtractString(line, 'decided_by_tier_name')
    ) AS decided_by_tier_name,
    -- Every tier that ran for the route task, by name, in execution order.
    -- Rows written before 005 have no such list.
    if(JSONHas(line, 'tier_outcomes'), JSONExtractRaw(line, 'tier_outcomes'), '[]') AS tier_outcomes_json,
    -- Name of the tier or downstream backend that served the request. Rows
    -- written before 005 carry only the backend (jev or llm), which is the same
    -- value for a dispatched request.
    if(
        empty(JSONExtractString(line, 'final_routing', 'tier_name')),
        final_backend,
        JSONExtractString(line, 'final_routing', 'tier_name')
    ) AS final_tier_name
FROM llm_observability.route_records_raw
WHERE isValidJSON(line)
  AND toUUIDOrNull(JSONExtractString(line, 'request_id')) IS NOT NULL
  AND parseDateTime64BestEffortOrNull(JSONExtractString(line, 'timestamp'), 6, 'UTC') IS NOT NULL;

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
