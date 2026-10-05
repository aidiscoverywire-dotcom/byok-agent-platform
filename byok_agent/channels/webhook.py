"""Meta webhook endpoint (FastAPI).

Two responsibilities ONLY:

1. ``GET /webhook`` — Meta's verification handshake.
2. ``POST /webhook`` — verify ``X-Hub-Signature-256`` (HMAC-SHA256 with
   the app secret, constant-time compare; 401 on mismatch), normalize the
   payload into :class:`IncomingMessage`, push it onto the queue, and
   return 200 IMMEDIATELY.

The agent is NEVER run inside the webhook handler. A slow model call in
the request path would make Meta retry the webhook (duplicate messages)
and eventually disable it. The worker pool (``worker.py``) consumes the
queue separately. There is a test asserting the handler returns without
calling the agent.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from .transport import IncomingMessage

log = logging.getLogger(__name__)


def _to_bytes(s: str) -> bytes:
    """UTF-8 encode for compare_digest.

    ``hmac.compare_digest(str, str)`` raises TypeError on non-ASCII input —
    an unauthenticated request with a non-ASCII token would 500 instead of
    401/403. Comparing bytes never raises (utf-8 encodes any str).
    """
    return s.encode("utf-8")


def verify_signature(raw_body: bytes, signature_header: str, app_secret: str) -> bool:
    """Constant-time HMAC-SHA256 check of ``X-Hub-Signature-256``."""
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(_to_bytes("sha256=" + expected), _to_bytes(signature_header))


def normalize_webhook(payload: dict[str, Any]) -> list[IncomingMessage]:
    """Flatten Meta's webhook JSON into normalized inbound messages.

    - Text messages -> kind="text"
    - audio/voice   -> kind="audio" (media_id set)
    - image         -> kind="image" (media_id set, caption as text)
    - statuses (sent/delivered/read) -> skipped entirely (never replied to)
    - anything else -> kind="unknown" (worker skips gracefully)
    """
    out: list[IncomingMessage] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                sender = str(msg.get("from", ""))
                mid = str(msg.get("id", ""))
                ts = msg.get("timestamp", "")
                try:
                    iso_ts = datetime.fromtimestamp(
                        int(ts), tz=timezone.utc
                    ).isoformat()
                except (ValueError, TypeError):
                    iso_ts = ""
                mtype = msg.get("type", "")
                if mtype == "text":
                    out.append(
                        IncomingMessage(
                            channel="whatsapp",
                            sender_id=sender,
                            message_id=mid,
                            kind="text",
                            text=str(msg.get("text", {}).get("body", "")),
                            timestamp=iso_ts,
                            raw=msg,
                        )
                    )
                elif mtype in ("audio", "voice"):
                    audio = msg.get(mtype, {})
                    out.append(
                        IncomingMessage(
                            channel="whatsapp",
                            sender_id=sender,
                            message_id=mid,
                            kind="audio",
                            media_id=str(audio.get("id", "")),
                            timestamp=iso_ts,
                            raw=msg,
                        )
                    )
                elif mtype == "image":
                    image = msg.get("image", {})
                    out.append(
                        IncomingMessage(
                            channel="whatsapp",
                            sender_id=sender,
                            message_id=mid,
                            kind="image",
                            text=str(image.get("caption", "")),
                            media_id=str(image.get("id", "")),
                            timestamp=iso_ts,
                            raw=msg,
                        )
                    )
                else:
                    # stickers, reactions, locations, contacts, ... — the
                    # worker skips these without replying.
                    out.append(
                        IncomingMessage(
                            channel="whatsapp",
                            sender_id=sender,
                            message_id=mid,
                            kind="unknown",
                            timestamp=iso_ts,
                            raw=msg,
                        )
                    )
            # NOTE: value["statuses"] (sent/delivered/read receipts) are
            # intentionally ignored — we never reply to our own traffic.
    return out


def create_app(
    *,
    verify_tokens: dict[str, str],
    app_secrets: dict[str, str],
    phone_owner: dict[str, str],
    queue: "asyncio.Queue[tuple[str | None, IncomingMessage]]",
) -> FastAPI:
    """Build the webhook app with injected config (test-friendly).

    Args:
        verify_tokens: owner -> verify token (GET handshake; accepted if it
            matches ANY configured owner).
        app_secrets: owner -> app secret (POST HMAC; tried per owner).
        phone_owner: business phone_number_id -> owner (resolves which
            owner's number was messaged).
        queue: receives ``(owner_or_None, IncomingMessage)`` tuples.
    """
    app = FastAPI(title="byok-agent WhatsApp webhook")

    def _owner_for_secret(sig: str, raw: bytes) -> str | None:
        for owner, secret in app_secrets.items():
            if verify_signature(raw, sig, secret):
                return owner
        return None

    @app.get("/webhook")
    async def verify(request: Request) -> Response:
        params = request.query_params
        mode = params.get("hub.mode", "")
        token = params.get("hub.verify_token", "")
        challenge = params.get("hub.challenge", "")
        if mode == "subscribe" and token:
            for expected in verify_tokens.values():
                # Bytes comparison: str compare_digest raises TypeError on
                # non-ASCII input (500 on an unauthenticated request).
                if hmac.compare_digest(_to_bytes(token), _to_bytes(expected)):
                    return PlainTextResponse(challenge)
        return JSONResponse({"error": "verification failed"}, status_code=403)

    @app.post("/webhook")
    async def receive(request: Request) -> Response:
        raw = await request.body()
        sig = request.headers.get("X-Hub-Signature-256", "")
        # The signature doesn't name the owner: try each configured secret.
        # (Few owners in practice; each check is constant-time.)
        owner = _owner_for_secret(sig, raw)
        if owner is None:
            log.warning("webhook: bad signature; rejecting")
            return JSONResponse({"error": "bad signature"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "bad json"}, status_code=400)
        for message in normalize_webhook(payload):
            # Prefer the business number actually messaged, fall back to
            # the signature owner.
            pn_owner = _phone_owner(payload, phone_owner) or owner
            try:
                queue.put_nowait((pn_owner, message))
            except asyncio.QueueFull:
                log.error("webhook: queue full; dropping message %s", message.message_id)
        return JSONResponse({"ok": True})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _phone_owner(
    payload: dict[str, Any], phone_owner: dict[str, str]
) -> str | None:
    """Resolve the owner from the business phone_number_id in the payload."""
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            pn_id = str(change.get("value", {}).get("metadata", {}).get("phone_number_id", ""))
            if pn_id and pn_id in phone_owner:
                return phone_owner[pn_id]
    return None
