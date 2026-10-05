-- BYOK Agent Platform — Phase 0 schema (SQLite, local dev).
-- Same tables as schema.sql; AUTOINCREMENT instead of SERIAL.

CREATE TABLE IF NOT EXISTS vault_keys (
    key_id                      TEXT PRIMARY KEY,
    label                       TEXT NOT NULL,
    provider                    TEXT NOT NULL,
    owner                       TEXT NOT NULL DEFAULT '',  -- tenant scope: which user owns this key
    encrypted_key               BLOB NOT NULL,    -- Fernet token, never plaintext
    key_version                 INTEGER NOT NULL DEFAULT 1,
    status                      TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'revoked')),
    allow_cross_provider_fallback INTEGER NOT NULL DEFAULT 0,
    created_at                  TEXT NOT NULL     -- ISO-8601 UTC
);

CREATE TABLE IF NOT EXISTS ledger (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id              TEXT NOT NULL,
    provider            TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cached_tokens       INTEGER NOT NULL DEFAULT 0,
    -- ESTIMATE only: litellm.completion_cost() approximation, never billing truth.
    estimated_cost_usd  REAL NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,            -- ISO-8601 UTC
    status              TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok','failed','skipped')),
    error               TEXT                      -- sanitized reason for failed/skipped
);

CREATE INDEX IF NOT EXISTS idx_ledger_key_id ON ledger (key_id);

-- Phase 2: WhatsApp control plane --------------------------------------
CREATE TABLE IF NOT EXISTS whatsapp_config (
    owner               TEXT PRIMARY KEY,
    waba_id             TEXT NOT NULL,
    phone_number_id     TEXT NOT NULL,
    verify_token        TEXT NOT NULL,
    -- New setups store the verify token as a vault record instead of
    -- plaintext (see verify_token_key_id). The old column stays for
    -- backward compatibility with existing installs.
    verify_token_key_id TEXT,
    access_token_key_id TEXT NOT NULL,
    app_secret_key_id   TEXT NOT NULL,
    agent_key_id        TEXT,
    agent_model         TEXT NOT NULL DEFAULT 'openai/gpt-4o-mini',
    max_steps           INTEGER NOT NULL DEFAULT 15,
    max_tokens          INTEGER NOT NULL DEFAULT 30000,
    max_usd             REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS whatsapp_senders (
    sender_id   TEXT PRIMARY KEY,
    owner       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','paired','blocked')),
    paired_at   TEXT
);

CREATE TABLE IF NOT EXISTS pairing_codes (
    code        TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS whatsapp_dedup (
    message_id  TEXT PRIMARY KEY,
    seen_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS whatsapp_rate (
    sender_id   TEXT NOT NULL,
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wa_rate_sender ON whatsapp_rate (sender_id);

CREATE TABLE IF NOT EXISTS whatsapp_rate_notice (
    sender_id   TEXT NOT NULL,
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wa_rate_notice_sender ON whatsapp_rate_notice (sender_id);

CREATE TABLE IF NOT EXISTS audit_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    args_hash   TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    hmac        TEXT NOT NULL
);

-- Phase 4: productization -------------------------------------------------
CREATE TABLE IF NOT EXISTS kill_switch (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    engaged     INTEGER NOT NULL DEFAULT 0,
    engaged_by  TEXT,
    reason      TEXT,
    engaged_at  TEXT
);

CREATE TABLE IF NOT EXISTS key_budgets (
    key_id          TEXT PRIMARY KEY,
    daily_usd_cap   REAL NOT NULL,
    day             TEXT NOT NULL DEFAULT '',
    spent_today     REAL NOT NULL DEFAULT 0,
    alerted_80      INTEGER NOT NULL DEFAULT 0,
    alerted_100     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS routing_rules (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id        TEXT NOT NULL,
    model_pattern TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routing_rules_key ON routing_rules (key_id);

CREATE TABLE IF NOT EXISTS approval_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    summary     TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','approved','denied')),
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    decided_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_approval_status ON approval_queue (status);
