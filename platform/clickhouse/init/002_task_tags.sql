-- 002_task_tags.sql
--
-- Task tagging observability for cascade-router. This migration creates the landing
-- table for raw task-tag JSONL rows and a typed projection table for downstream
-- analysis by request_id, category, and routing outcome.
--
-- The app's async tagger writes one JSON object per request to
-- logs/task_tags.YYYY-MM-DD.jsonl. Raw lines are intentionally kept immutable,
-- while task_tags is keyed by request_id and timestamp so replay safety is easy.
--
-- This migration assumes llm_observability already exists from 001_route_decisions.sql.
SELECT throwIf(count() > 0, 'migration 002 refused: task tag tables already exist')
FROM system.tables
WHERE database = 'llm_observability'
  AND name IN ('task_tags_raw', 'task_tags', 'task_tags_mv')
FORMAT Null;

CREATE TABLE llm_observability.task_tags_raw
(
    line        String CODEC(ZSTD(3)),
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY ingested_at;

CREATE TABLE llm_observability.task_tags
(
    timestamp                      DateTime64(6, 'UTC'),
    request_id                     UUID,

    taxonomy_version               LowCardinality(String),
    tagger_model                   LowCardinality(String),
    tagger_version                 LowCardinality(String),

    latency_ms                     Float64,
    queue_wait_ms                  Float64,

    tag_status                     LowCardinality(String),
    primary_category               LowCardinality(Nullable(String)),
    primary_score                  Nullable(Float32),
    nearest_category               LowCardinality(Nullable(String)),
    nearest_score                  Nullable(Float32),
    untagged_reason                LowCardinality(Nullable(String)),
    embedding                      Array(Float32),

    tagger_usage_input_tokens      Nullable(UInt64),
    tagger_usage_output_tokens     Nullable(UInt64),
    tagger_usage_estimated         Bool
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, request_id);

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
