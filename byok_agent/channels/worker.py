"""WhatsApp worker: queue -> dedup -> pairing -> guard -> agent -> reply.

One worker coroutine drains the webhook's ``asyncio.Queue``. The pipeline
per message:

1. **Dedup** — ``whatsapp_dedup`` by ``message_id`` (Meta retries
   webhooks; without this the agent would run twice and double-spend).
2. **Rate limit** — max 20 inbound messages/hour per sender (anti-spam /
   anti-loop). Over-limit senders get ONE polite notice per hour, then
   silence.
3. **Pairing** — blocked/unknown/pending senders never reach the agent
   (see ``pairing.py``).
4. **Guard** — inbound injection scan before any agent execution.
5. **Agent** — the Phase 0 ``AgentRunner`` with a per-message
   ``BudgetEnforcer`` (sane WhatsApp defaults: 15 steps, $1.00 cap).
6. **Reply** — outbound DLP redaction, then send via the transport.

The worker NEVER replies to statuses or its own outbound traffic (the
webhook normalizer already drops statuses; the transport never enqueues
sends).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from .audit import AuditLog
from .guard import ToolGuard, install_tool_guard
from .pairing import NOT_PAIRED_TEXT, PairingStore
from .transport import ChannelTransport, IncomingMessage

log = logging.getLogger(__name__)

RATE_LIMIT_PER_HOUR = 20
RATE_NOTICE_TEXT = (
    "You're sending messages very quickly — I've paused for a bit to "
    "avoid runaway costs. Try again in a little while."
)
PAIRED_OK_TEXT = "Paired. You can now send instructions."
IMAGE_NO_VISION_TEXT = (
    "I can't see images yet — send text or a voice note instead."
)
NOT_CONFIGURED_TEXT = (
    "The WhatsApp agent isn't configured yet. The owner needs to run:\n"
    "byok whatsapp setup --user <owner>"
)
BLOCKED_INBOUND_TEXT = "That message was blocked by a safety filter."

# agent_runner_factory: (owner, thread_id, text, budget_kwargs) -> answer text
AgentRunnerFactory = Callable[[str, str, str, dict[str, Any]], Awaitable[str]]
# transport_factory: (owner) -> ChannelTransport
TransportFactory = Callable[[str], Awaitable[ChannelTransport]]


def _safe_tid(message_id: str) -> str:
    """Thread-safe task id derived from a channel message id."""
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", message_id or "nomid")[:48]
    return f"wa-{clean}"


def _is_conflict(exc: Exception) -> bool:
    """True if the exception is a unique/primary-key violation.

    sqlite3 raises IntegrityError, asyncpg raises UniqueViolationError —
    match on the class name and message so neither backend needs importing.
    """
    name = type(exc).__name__
    msg = str(exc).lower()
    return (
        name in ("IntegrityError", "UniqueViolationError")
        or "unique constraint" in msg
        or "duplicate key" in msg
        or "primary key" in msg
    )


class WhatsAppWorker:
    """Consumes normalized inbound messages and drives the agent."""

    def __init__(
        self,
        *,
        db: Any,
        vault: Any,
        ledger: Any,
        pairing: PairingStore,
        audit: AuditLog,
        guard: ToolGuard,
        transport_factory: TransportFactory,
        agent_runner_factory: AgentRunnerFactory,
        whatsapp_config: Callable[[str], Awaitable[dict[str, Any] | None]],
        rate_limit_per_hour: int = RATE_LIMIT_PER_HOUR,
    ) -> None:
        self._db = db
        self._vault = vault
        self._ledger = ledger
        self._pairing = pairing
        self._audit = audit
        self._guard = guard
        self._transport_factory = transport_factory
        self._agent_runner_factory = agent_runner_factory
        self._whatsapp_config = whatsapp_config
        self._rate_limit = rate_limit_per_hour
        self._running = False
        # Install the tool guard process-wide so the agent loop's tool
        # calls are gated even though AgentRunner is Phase 0 code.
        install_tool_guard(guard)

    # -- main loop -------------------------------------------------------
    async def run_forever(
        self, queue: "asyncio.Queue[tuple[str | None, IncomingMessage]]"
    ) -> None:
        self._running = True
        while self._running:
            owner_hint, message = await queue.get()
            try:
                await self.handle_message(owner_hint, message)
            except Exception as exc:  # one bad message must not kill the loop
                log.exception("worker: message %s failed: %s", message.message_id, exc)
                try:
                    await self._audit.append(
                        actor="worker",
                        action="message_error",
                        args={"message_id": message.message_id, "error": type(exc).__name__},
                    )
                except Exception as audit_exc:
                    # The audit trail must never take down the worker loop:
                    # log the logging failure and keep serving.
                    log.error(
                        "worker: audit append failed for message_error: %s",
                        type(audit_exc).__name__,
                    )
            finally:
                queue.task_done()

    def stop(self) -> None:
        self._running = False

    # -- per-message pipeline --------------------------------------------
    async def handle_message(
        self, owner_hint: str | None, message: IncomingMessage
    ) -> str:
        """Process one message. Returns a short outcome tag (for tests).

        ``owner_hint`` is the owner resolved by the webhook from the
        business ``phone_number_id`` (None if the payload didn't map to a
        configured number). The pairing table's owner wins for paired
        senders; otherwise the hint is used for replies.
        """
        sender = message.sender_id

        # 1. Dedup (Meta retries).
        if not await self._dedup_claim(message.message_id):
            await self._audit.append(
                actor="worker", action="dedup_skip",
                args={"message_id": message.message_id},
            )
            return "dedup_skip"

        # 2. Rate limit.
        allowed, notified = await self._rate_check(sender)
        if not allowed:
            await self._audit.append(
                actor="worker", action="rate_limited",
                args={"sender_id": sender},
            )
            if notified:
                transport = await self._transport_for_owner_hint(
                    owner_hint, sender
                )
                if transport:
                    await transport.send_text(sender, RATE_NOTICE_TEXT)
            return "rate_limited"

        # 3. Pairing gate — BEFORE any agent/tool execution.
        status = await self._pairing.sender_status(sender)
        if status == "blocked":
            await self._audit.append(
                actor="worker", action="blocked_sender",
                args={"sender_id": sender},
            )
            return "blocked"
        owner: str | None = None
        if status in ("unknown", "pending"):
            owner = await self._handle_unpaired(message, status, owner_hint)
            if owner is None:
                return "unpaired"
            # Redeemed a code just now -> fall through as paired.
        else:  # paired
            row = await self._db.fetchrow(
                "SELECT owner FROM whatsapp_senders WHERE sender_id=$1", sender
            )
            owner = str(row["owner"]) if row and str(row["owner"]) else owner_hint
        if not owner:
            await self._audit.append(
                actor="worker", action="no_owner",
                args={"sender_id": sender},
            )
            return "unpaired"

        # 4. Unknown kinds never reach the agent.
        if message.kind == "unknown":
            await self._audit.append(
                actor="worker", action="unknown_kind_skip",
                args={"message_id": message.message_id},
            )
            return "unknown_kind"

        # 5. Config present?
        config = await self._whatsapp_config(owner)
        if not config or not config.get("agent_key_id"):
            transport = await self._transport_factory(owner)
            await transport.send_text(sender, NOT_CONFIGURED_TEXT)
            return "not_configured"

        # 6. Materialize text (voice -> transcript).
        text = message.text or ""
        if message.kind == "audio":
            text = await self._transcribe(message, owner, config)
            if text is None:
                return "voice_failed"

        # 7. Guard: inbound injection scan.
        verdict = self._guard.check_inbound(text)
        if verdict.blocked:
            await self._audit.append(
                actor="worker", action="guard_blocked_inbound",
                args={"sender_id": sender, "reason": verdict.reason},
            )
            transport = await self._transport_factory(owner)
            await transport.send_text(sender, BLOCKED_INBOUND_TEXT)
            return "guard_blocked"

        if message.kind == "image" and not text.strip():
            transport = await self._transport_factory(owner)
            await transport.send_text(sender, IMAGE_NO_VISION_TEXT)
            return "image_no_caption"

        # 8. Agent run (per-message budget).
        await self._audit.append(
            actor="worker", action="agent_start",
            args={"sender_id": sender, "message_id": message.message_id},
        )
        thread_id = f"user:{owner}:task:{_safe_tid(message.message_id)}"
        budget = {
            "max_steps": int(config.get("max_steps", 15)),
            "max_tokens": int(config.get("max_tokens", 30_000)),
            "max_usd": float(config.get("max_usd", 1.0)),
        }
        try:
            answer = await self._agent_runner_factory(
                owner, thread_id, text, budget
            )
        except Exception as exc:
            await self._audit.append(
                actor="worker", action="agent_error",
                args={"sender_id": sender, "error": type(exc).__name__},
            )
            transport = await self._transport_factory(owner)
            await transport.send_text(
                sender, "Something went wrong running that — try again."
            )
            return "agent_error"

        # 9. Outbound DLP + reply.
        safe = self._guard.redact_outbound(answer or "(no answer produced)")
        transport = await self._transport_factory(owner)
        await transport.send_text(sender, safe)
        await self._audit.append(
            actor="worker", action="replied",
            args={"sender_id": sender, "message_id": message.message_id},
        )
        return "replied"

    # -- pipeline steps ----------------------------------------------------
    async def _dedup_claim(self, message_id: str) -> bool:
        """True if this message is new (claims the id), False if seen.

        Only a primary-key/unique conflict means "already seen". Any OTHER
        database error propagates — treating a DB outage as "duplicate"
        would silently drop user messages.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            await self._db.execute(
                "INSERT INTO whatsapp_dedup (message_id, seen_at) VALUES ($1,$2)",
                message_id,
                now,
            )
            return True
        except Exception as exc:
            if _is_conflict(exc):
                return False  # primary-key conflict => already processed
            raise

    async def _rate_check(self, sender_id: str) -> tuple[bool, bool]:
        """(allowed, should_notify). One notice per hour, then silence."""
        now = datetime.now(timezone.utc)
        hour_ago = (now - timedelta(hours=1)).isoformat()
        await self._db.execute(
            "DELETE FROM whatsapp_rate WHERE ts < $1", hour_ago
        )
        rows = await self._db.fetch(
            "SELECT ts FROM whatsapp_rate WHERE sender_id=$1 AND ts >= $2",
            sender_id,
            hour_ago,
        )
        if len(rows) >= self._rate_limit:
            # Already notified this hour?
            notice = await self._db.fetchrow(
                "SELECT ts FROM whatsapp_rate_notice WHERE sender_id=$1 AND ts >= $2",
                sender_id,
                hour_ago,
            )
            if notice:
                return False, False
            await self._db.execute(
                "INSERT INTO whatsapp_rate_notice (sender_id, ts) VALUES ($1,$2)",
                sender_id,
                now.isoformat(),
            )
            return False, True
        await self._db.execute(
            "INSERT INTO whatsapp_rate (sender_id, ts) VALUES ($1,$2)",
            sender_id,
            now.isoformat(),
        )
        return True, False

    async def _handle_unpaired(
        self, message: IncomingMessage, status: str, owner_hint: str | None
    ) -> str | None:
        """Pairing flow. Returns owner if now paired, else None."""
        sender = message.sender_id
        # Maybe they're answering with a code?
        text = (message.text or "").strip()
        if text and re.fullmatch(r"\d{6}", text):
            owner = await self._pairing.redeem_code(sender, text)
            if owner:
                await self._audit.append(
                    actor="worker", action="paired",
                    args={"sender_id": sender, "owner": owner},
                )
                transport = await self._transport_factory(owner)
                await transport.send_text(sender, PAIRED_OK_TEXT)
                return owner
            # Wrong/expired code: stay silent-ish (they may retry).
            await self._audit.append(
                actor="worker", action="pair_code_failed",
                args={"sender_id": sender},
            )
            return None
        if status == "unknown":
            await self._pairing.mark_pending(sender)
            # Reply ONCE with pairing instructions from the business
            # number they messaged (owner_hint). The unknown->pending
            # transition guarantees this fires exactly once per sender.
            if owner_hint:
                try:
                    transport = await self._transport_factory(owner_hint)
                    await transport.send_text(sender, NOT_PAIRED_TEXT)
                except Exception as exc:
                    log.warning("worker: not-paired reply failed: %s", exc)
            await self._audit.append(
                actor="worker", action="first_contact",
                args={"sender_id": sender},
            )
        return None

    async def _transport_for_owner_hint(
        self, owner_hint: str | None, sender: str
    ) -> Any | None:
        """Transport for rate-limit notices: pairing owner first, then hint."""
        row = await self._db.fetchrow(
            "SELECT owner FROM whatsapp_senders WHERE sender_id=$1", sender
        )
        owner = str(row["owner"]) if row and str(row["owner"]) else owner_hint
        if not owner:
            return None
        return await self._transport_factory(owner)

    async def _transcribe(
        self, message: IncomingMessage, owner: str, config: dict[str, Any]
    ) -> str | None:
        """Download audio and transcribe. None => failure (already notified)."""
        from . import voice as voicemod
        from .whatsapp_cloud import WhatsAppCloudTransport

        transport = await self._transport_factory(owner)
        if not isinstance(transport, WhatsAppCloudTransport):
            sender = message.sender_id
            await transport.send_text(
                sender, "Voice notes need the WhatsApp Cloud transport."
            )
            return None
        try:
            audio, mime = await transport.download_media(message.media_id)
        except Exception as exc:
            log.warning("worker: media download failed: %s", type(exc).__name__)
            await transport.send_text(
                message.sender_id, "Couldn't download that voice note."
            )
            return None
        try:
            return await voicemod.transcribe_audio(
                audio=audio,
                mime_type=mime,
                key_id=str(config["agent_key_id"]),
                get_key=self._vault.get_key,
                owner=owner,
                ledger=self._ledger,
            )
        except Exception as exc:
            log.warning("worker: transcription failed: %s", type(exc).__name__)
            await transport.send_text(
                message.sender_id, "Couldn't transcribe that voice note."
            )
            await self._audit.append(
                actor="worker", action="voice_failed",
                args={"sender_id": message.sender_id},
            )
            return None
