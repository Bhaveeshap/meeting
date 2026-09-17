-- =============================================================================
-- LoopKeeper — The Meeting Accountability Engine
-- Database Schema (PostgreSQL 14+ with pgvector extension)
-- =============================================================================
-- Design notes:
--   * `assignees`      — canonical people, so name variants ("Bhaveesha",
--                         "Bhaveesha K.") resolve to one owner over time.
--   * `meetings`       — one row per processed meeting/transcript. Source of
--                         truth for chronology (everything is ordered by
--                         meeting_date, not by insert time).
--   * `action_items`   — the CURRENT state of each task. One row per logical
--                         task, even though it may have been re-phrased in
--                         five different meetings. This is what the dedup
--                         engine matches against.
--   * `task_history`   — the append-only audit trail. Every status change,
--                         deadline pushback, re-phrasing, or merge event is
--                         logged here, keyed to the meeting it was observed
--                         in. This is what lets you answer "what happened to
--                         this task over time?" and "how many times has this
--                         been pushed back?".
--
-- A SQLite-compatible fallback (no pgvector) is provided at the bottom for
-- local prototyping — see "SQLITE FALLBACK".
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram fuzzy text matching (assignee/action text)

-- -----------------------------------------------------------------------------
-- ENUM TYPES
-- -----------------------------------------------------------------------------

CREATE TYPE task_status AS ENUM ('pending', 'done', 'overdue', 'blocked', 'cancelled');

CREATE TYPE history_event_type AS ENUM (
    'created',            -- task first observed
    'status_change',      -- pending -> done, pending -> overdue, etc.
    'deadline_pushback',  -- deadline moved to a later date
    'reassigned',         -- owner changed
    'rephrased',          -- description updated but same underlying task
    'merged'               -- two action_items collapsed into one (dedup correction)
);

-- -----------------------------------------------------------------------------
-- ASSIGNEES
-- -----------------------------------------------------------------------------

CREATE TABLE assignees (
    id              BIGSERIAL PRIMARY KEY,
    canonical_name  TEXT NOT NULL UNIQUE,
    email           TEXT UNIQUE,
    aliases         TEXT[] NOT NULL DEFAULT '{}',   -- name variants seen in raw notes
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_assignees_name_trgm ON assignees USING gin (canonical_name gin_trgm_ops);

-- -----------------------------------------------------------------------------
-- MEETINGS
-- -----------------------------------------------------------------------------

CREATE TABLE meetings (
    id              BIGSERIAL PRIMARY KEY,
    meeting_date    DATE NOT NULL,
    title           TEXT,
    raw_notes       TEXT NOT NULL,           -- original unstructured transcript/notes
    attendees       TEXT[] NOT NULL DEFAULT '{}',
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_meetings_date ON meetings (meeting_date);

-- -----------------------------------------------------------------------------
-- ACTION ITEMS  (current state, one row per logical task)
-- -----------------------------------------------------------------------------

CREATE TABLE action_items (
    id                  BIGSERIAL PRIMARY KEY,

    -- Ownership
    assignee_id         BIGINT NOT NULL REFERENCES assignees(id),

    -- Content (current phrasing — history of prior phrasings lives in task_history)
    description         TEXT NOT NULL,
    action_core         TEXT NOT NULL,        -- normalized/lemmatized core phrase, used as a
                                               -- fast fuzzy pre-filter before vector search
    description_embedding vector(384),        -- embedding of `description`; dimension must match
                                               -- the embedding backend (384 = MiniLM-L6-v2 default)

    -- Deadlines & delay tracking
    original_deadline   DATE,                 -- deadline as first stated, never overwritten
    current_deadline    DATE,                 -- most recently stated deadline
    pushback_count      INTEGER NOT NULL DEFAULT 0,
    is_chronic_delay     BOOLEAN NOT NULL DEFAULT FALSE,  -- pushback_count >= CHRONIC_THRESHOLD

    -- Status
    status              task_status NOT NULL DEFAULT 'pending',

    -- Provenance
    first_meeting_id    BIGINT NOT NULL REFERENCES meetings(id),
    last_meeting_id     BIGINT NOT NULL REFERENCES meetings(id),
    is_active           BOOLEAN NOT NULL DEFAULT TRUE,    -- false once done/cancelled and archived

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_action_items_assignee_status ON action_items (assignee_id, status) WHERE is_active;
CREATE INDEX idx_action_items_deadline        ON action_items (current_deadline) WHERE status IN ('pending', 'overdue');
CREATE INDEX idx_action_items_core_trgm       ON action_items USING gin (action_core gin_trgm_ops);
-- Approximate nearest-neighbor index for semantic dedup search (cosine distance).
-- Build/tune `lists` once you have real data volume; IVFFlat needs ANALYZE after bulk load.
CREATE INDEX idx_action_items_embedding_ann ON action_items
    USING ivfflat (description_embedding vector_cosine_ops) WITH (lists = 100);

-- -----------------------------------------------------------------------------
-- TASK HISTORY (append-only audit trail)
-- -----------------------------------------------------------------------------

CREATE TABLE task_history (
    id              BIGSERIAL PRIMARY KEY,
    action_item_id  BIGINT NOT NULL REFERENCES action_items(id),
    meeting_id      BIGINT NOT NULL REFERENCES meetings(id),
    event_type      history_event_type NOT NULL,
    old_value       JSONB,          -- e.g. {"deadline": "2026-09-10"} or {"status": "pending"}
    new_value       JSONB,          -- e.g. {"deadline": "2026-09-24"} or {"status": "overdue"}
    note            TEXT,           -- free-text explanation extracted from the meeting notes
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_task_history_item    ON task_history (action_item_id, recorded_at);
CREATE INDEX idx_task_history_meeting ON task_history (meeting_id);
CREATE INDEX idx_task_history_type    ON task_history (event_type);

-- -----------------------------------------------------------------------------
-- Convenience view: current open workload per assignee
-- (the workload_analytics.py module reproduces this logic in Python so it can
--  run against the in-memory state store too; this view is the SQL-native
--  equivalent for direct BI/dashboard queries against Postgres.)
-- -----------------------------------------------------------------------------

CREATE VIEW v_assignee_workload AS
SELECT
    a.id                                                   AS assignee_id,
    a.canonical_name,
    COUNT(*) FILTER (WHERE ai.status = 'pending')           AS open_count,
    COUNT(*) FILTER (WHERE ai.status = 'overdue')           AS overdue_count,
    COUNT(*) FILTER (WHERE ai.status = 'done')              AS done_count,
    COUNT(*) FILTER (WHERE ai.is_chronic_delay)             AS chronic_delay_count,
    COUNT(*) FILTER (
        WHERE ai.status IN ('pending', 'overdue')
        AND ai.current_deadline BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
    )                                                        AS due_within_7d_count
FROM assignees a
LEFT JOIN action_items ai ON ai.assignee_id = a.id AND ai.is_active
GROUP BY a.id, a.canonical_name;

-- =============================================================================
-- SQLITE FALLBACK (no pgvector / pg_trgm) — for local prototyping without Postgres
-- =============================================================================
-- SQLite has no native vector or array types. Store the embedding as a JSON-
-- encoded float array (or a BLOB of packed float32s) and do the cosine-
-- similarity search in Python (see dedup_engine.py — this is exactly what its
-- `HashingTfidfBackend` / any sentence-transformers backend expects to plug
-- into). Enums become CHECK constraints; TEXT[] becomes a JSON TEXT column.
--
-- CREATE TABLE assignees (
--     id              INTEGER PRIMARY KEY AUTOINCREMENT,
--     canonical_name  TEXT NOT NULL UNIQUE,
--     email           TEXT UNIQUE,
--     aliases_json    TEXT NOT NULL DEFAULT '[]',
--     created_at      TEXT NOT NULL DEFAULT (datetime('now'))
-- );
--
-- CREATE TABLE meetings (
--     id              INTEGER PRIMARY KEY AUTOINCREMENT,
--     meeting_date    TEXT NOT NULL,
--     title           TEXT,
--     raw_notes       TEXT NOT NULL,
--     attendees_json  TEXT NOT NULL DEFAULT '[]',
--     ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
-- );
--
-- CREATE TABLE action_items (
--     id                      INTEGER PRIMARY KEY AUTOINCREMENT,
--     assignee_id             INTEGER NOT NULL REFERENCES assignees(id),
--     description             TEXT NOT NULL,
--     action_core             TEXT NOT NULL,
--     description_embedding_json TEXT,     -- JSON array of floats
--     original_deadline       TEXT,
--     current_deadline        TEXT,
--     pushback_count          INTEGER NOT NULL DEFAULT 0,
--     is_chronic_delay        INTEGER NOT NULL DEFAULT 0,  -- 0/1
--     status                  TEXT NOT NULL DEFAULT 'pending'
--                              CHECK (status IN ('pending','done','overdue','blocked','cancelled')),
--     first_meeting_id        INTEGER NOT NULL REFERENCES meetings(id),
--     last_meeting_id         INTEGER NOT NULL REFERENCES meetings(id),
--     is_active               INTEGER NOT NULL DEFAULT 1,
--     created_at               TEXT NOT NULL DEFAULT (datetime('now')),
--     updated_at               TEXT NOT NULL DEFAULT (datetime('now'))
-- );
--
-- CREATE TABLE task_history (
--     id              INTEGER PRIMARY KEY AUTOINCREMENT,
--     action_item_id  INTEGER NOT NULL REFERENCES action_items(id),
--     meeting_id      INTEGER NOT NULL REFERENCES meetings(id),
--     event_type      TEXT NOT NULL CHECK (event_type IN
--                      ('created','status_change','deadline_pushback','reassigned','rephrased','merged')),
--     old_value_json  TEXT,
--     new_value_json  TEXT,
--     note            TEXT,
--     recorded_at     TEXT NOT NULL DEFAULT (datetime('now'))
-- );
-- =============================================================================
