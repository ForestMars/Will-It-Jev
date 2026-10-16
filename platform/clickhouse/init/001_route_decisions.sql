-- 001_route_decisions.sql
--
-- Decision observability for cascade-router (Will-It-Jev). Initial schema: creates the
-- llm_observability database and everything in it.
--
-- Up refuses to run if the database llm_observability already exists: the first statement is
-- a preflight that fails before anything is changed. If up fails after the preflight, everything
-- present was created by this run, so running the down migration restores the prior system
-- state exactly (database absent).
--
-- The router already emits one RouteRecord per request as a JSON line
-- (logs/routing_capture.jsonl). This schema ingests those lines unchanged:
--
--   route_records_raw   landing table, one row per JSON line, never altered
--   route_decisions     typed, queryable projection, populated by a materialized view
--
-- Ingest with no router change:
--   clickhouse-client --input_format_allow_errors_num=10 \
--       --query "INSERT INTO llm_observability.route_records_raw FORMAT JSONAsString" \
--       < logs/routing_capture.jsonl
-- The router appends to this file, so the last line can be mid-write. Without the flag,
-- one truncated line rejects the whole file; with it, up to 10 unparseable lines are skipped.
-- Replaying a file is safe: route_decisions is a ReplacingMergeTree keyed on
-- (timestamp, request_id), so replayed rows collapse on merge. Read with FINAL for exact
-- counts before a merge has run. route_records_raw keeps every replayed line.
--
-- Tier numbering note: cascade tiers 1-4 are classification tiers. dispatch.rs
-- reuses 4 and 5 for the downstream endpoints (jev, frontier). service.rs overwrites
-- final_routing.tier with the downstream value after dispatch, so the tier that
-- DECIDED the route is derived below as decided_by_tier from the per-tier outcomes.
--
-- Backend note: service.rs also overwrites final_routing.backend with the downstream backend,
-- so on failover (jev endpoint fails, request falls through to the frontier endpoint) the
-- record says llm although the cascade decided jev. decided_backend recovers the cascade's
-- decision: failover only happens from the jev path, so is_failover implies jev, and
-- otherwise final_backend is the decided backend.

SELECT throwIf(count() > 0, 'migration 001 refused: database llm_observability already exists')
FROM system.databases
WHERE name = 'llm_observability'
FORMAT Null;

CREATE DATABASE llm_observability;

CREATE TABLE llm_observability.route_records_raw
(
    line        String CODEC(ZSTD(3)),
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree
ORDER BY ingested_at;

CREATE TABLE llm_observability.route_decisions
(
    timestamp                       DateTime64(6, 'UTC'),
    request_id                      UUID,

    -- policy identity: the router hashes its tier1 config and tier2 model at startup
    tier1_toml_hash                 LowCardinality(String),
    tier2_bin_hash                  LowCardinality(String),
    policy_version                  LowCardinality(String)
        MATERIALIZED concat(substring(tier1_toml_hash, 1, 8), '-', substring(tier2_bin_hash, 1, 8)),

    prompt_raw                      String CODEC(ZSTD(3)),

    -- tier 1: heuristic
    tier1_confidence                Float32,
    tier1_decided                   Bool,
    tier1_reason                    LowCardinality(String),

    -- tier 2: fastText (null when tier 1 resolved)
    tier2_p_jev                     Nullable(Float32),
    tier2_p_llm                     Nullable(Float32),
    tier2_raw_probabilities         Array(Float32),
    tier2_reason                    LowCardinality(Nullable(String)),

    -- tier 3: harrier sidecar (null when tier 1 or 2 resolved)
    tier3_score                     Nullable(Float32),
    tier3_reason                    LowCardinality(Nullable(String)),

    -- the decision
    decided_by_tier                 UInt8,
    decided_backend                 LowCardinality(String),
    router_classification_latency_us UInt64,

    -- execution outcome (what happened after the decision)
    final_backend                   LowCardinality(String),
    downstream_tier                 LowCardinality(Nullable(String)),
    downstream_outcome              LowCardinality(Nullable(String)),
    downstream_invocation_latency_ms Float64,
    request_latency_primary_ms      Nullable(Float64),
    request_latency_failover_ms     Nullable(Float64),
    is_failover                     Bool,
    failure_reason                  LowCardinality(Nullable(String)),
    total_latency_us                UInt64,

    sample_prob                     Float32
)
ENGINE = ReplacingMergeTree
PARTITION BY toYYYYMM(timestamp)
ORDER BY (timestamp, request_id);

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

    multiIf(
        JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)') = 'resolved', 3,
        JSONExtract(line, 'tier3_outcome', 'reason', 'Nullable(String)') IS NOT NULL, 4,
        JSONExtract(line, 'tier2_outcome', 'reason', 'Nullable(String)') = 'resolved', 2,
        1
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
