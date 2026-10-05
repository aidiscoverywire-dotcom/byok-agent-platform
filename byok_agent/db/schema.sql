-- BYOK Agent Platform — Phase 0 schema (PostgreSQL).
-- LangGraph checkpoints are managed by the checkpointer library itself;
-- do NOT hand-roll checkpoint tables here.

CREATE TABLE IF NOT EXISTS vault_keys (
    key_id                      TEXT PRIMARY KEY,
    label                       TEXT NOT NULL,
    provider                    TEXT NOT NULL,
    owner                       TEXT NOT NULL DEFAULT '',  -- tenant scope: which user owns this key
    encrypted_key               BYTEA NOT NULL,   -- Fernet token, never plaintext
    key_version                 INTEGER NOT NULL DEFAULT 1,
    status                      TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'revoked')),
    allow_cross_provider_fallback INTEGER NOT NULL DEFAULT 0,
    created_at                  TEXT NOT NULL     -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS ledger (
    id                  SERIAL PRIMARY KEY,
    key_id              TEXT NOT NULL,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cached_tokens       INTEGER NOT NULL DEFAULT 0,
    -- ESTIMATE only: litellm.completion_cost() approximation, never billing truth.
    estimated_cost_usd  DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,            -- ISO-8601 UTC
    status              TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok','failed','skipped')),
    error               TEXT                      -- sanitized reason for failed/skipped
);

CREATE INDEX IF NOT EXISTS idx_ledger_key_id ON ledger (key_id);

-- Phase 2: WhatsApp control plane --------------------------------------
-- One row per owner: which business number + whose keys drive it.
CREATE TABLE IF NOT EXISTS whatsapp_config (
    owner               TEXT PRIMARY KEY,   -- tenant scope
    waba_id             TEXT NOT NULL,
    phone_number_id     TEXT NOT NULL,
    -- Handshake token for Meta's GET verification. The real auth on the
    -- POST path is the HMAC app secret (stored as a vault record).
    verify_token        TEXT NOT NULL,
    -- New setups store the verify token as a vault record instead of
    -- plaintext (see verify_token_key_id). The old column stays for
    -- backward compatibility with existing installs.
    verify_token_key_id TEXT,
    access_token_key_id TEXT NOT NULL,      -- vault key_id (provider='whatsapp')
    app_secret_key_id   TEXT NOT NULL,      -- vault key_id (provider='whatsapp')
    agent_key_id        TEXT,               -- vault key_id driving the agent
    agent_model         TEXT NOT NULL DEFAULT 'openai/gpt-4o-mini',
    max_steps           INTEGER NOT NULL DEFAULT 15,
    max_tokens          INTEGER NOT NULL DEFAULT 30000,
    max_usd             DOUBLE PRECISION NOT NULL DEFAULT 1.0
);

-- Pairing state: which WhatsApp senders may drive the agent.
CREATE TABLE IF NOT EXISTS whatsapp_senders (
    sender_id   TEXT PRIMARY KEY,           -- WhatsApp msisdn of the sender
    owner       TEXT NOT NULL DEFAULT '',   -- paired owner ('' until paired)
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','paired','blocked')),
    paired_at   TEXT                        -- ISO-8601 UTC, set when paired
);

-- Single-use pairing codes issued by the owner via CLI.
CREATE TABLE IF NOT EXISTS pairing_codes (
    code        TEXT PRIMARY KEY,           -- 6 digits
    owner       TEXT NOT NULL,
    created_at  TEXT NOT NULL,              -- ISO-8601 UTC
    used        INTEGER NOT NULL DEFAULT 0
);

-- Webhook dedup: Meta retries deliveries; one row per message id.
CREATE TABLE IF NOT EXISTS whatsapp_dedup (
    message_id  TEXT PRIMARY KEY,
    seen_at     TEXT NOT NULL               -- ISO-8601 UTC
);

-- Per-sender inbound rate limiting (rolling 1h window).
CREATE TABLE IF NOT EXISTS whatsapp_rate (
    sender_id   TEXT NOT NULL,
    ts          TEXT NOT NULL               -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_wa_rate_sender ON whatsapp_rate (sender_id);

-- One polite notice per sender per hour, then silence.
CREATE TABLE IF NOT EXISTS whatsapp_rate_notice (
    sender_id   TEXT NOT NULL,
    ts          TEXT NOT NULL               -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_wa_rate_notice_sender ON whatsapp_rate_notice (sender_id);

-- HMAC-chained audit log (see channels/audit.py for the tamper model).
CREATE TABLE IF NOT EXISTS audit_log (
    seq         SERIAL PRIMARY KEY,
    ts          TEXT NOT NULL,              -- ISO-8601 UTC
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    args_hash   TEXT NOT NULL,              -- sha256 of canonical args, never secrets
    prev_hash   TEXT NOT NULL,
    hmac        TEXT NOT NULL
);

-- Phase 4: productization -------------------------------------------------
-- Persistent kill switch: single row (id=1). Budgets check this DB flag in
-- addition to the AGENT_KILL_SWITCH env var (cached <=5s).
CREATE TABLE IF NOT EXISTS kill_switch (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    engaged     INTEGER NOT NULL DEFAULT 0,
    engaged_by  TEXT,
    reason      TEXT,
    engaged_at  TEXT
);

-- Per-key daily USD budgets (estimates). `day` is UTC YYYY-MM-DD; caps reset
-- when the day rolls over.
CREATE TABLE IF NOT EXISTS key_budgets (
    key_id          TEXT PRIMARY KEY,
    daily_usd_cap   REAL NOT NULL,
    day             TEXT NOT NULL DEFAULT '',
    spent_today     REAL NOT NULL DEFAULT 0,
    alerted_80      INTEGER NOT NULL DEFAULT 0,
    alerted_100     INTEGER NOT NULL DEFAULT 0
);

-- Model routing rules: a key may only serve models matching >=1 pattern
-- (fnmatch, e.g. 'deepseek/*', 'ollama/*'). No rows => no restriction.
CREATE TABLE IF NOT EXISTS routing_rules (
    id            SERIAL PRIMARY KEY,
    key_id        TEXT NOT NULL,
    model_pattern TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routing_rules_key ON routing_rules (key_id);

-- Human approval queue (build promotions, deploy gates). interrupt() asks are
-- persisted here so the dashboard/CLI can approve/deny out of band.
CREATE TABLE IF NOT EXISTS approval_queue (
    id          SERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,              -- 'build-promote' | 'deploy' | ...
    summary     TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}', -- JSON: project_dir, staging_dir, ...
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','approved','denied')),
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    decided_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_queue (status);
