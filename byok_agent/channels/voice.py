"""Voice notes: WhatsApp audio -> text via transcription.

Flow: download the audio bytes with the Cloud API transport, then
transcribe with litellm's ``atranscription`` (``whisper-1``) using the
CLIENT's key — metered in the ledger like any other model call.

Limits: audio longer than 5 minutes is rejected (Whisper's practical
limit and a cost guard in one).
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

MAX_AUDIO_SECONDS = 5 * 60
TRANSCRIBE_MODEL = "whisper-1"
# Whisper is billed per minute, not per token — this constant is the
# documented rate; VERIFY against the provider's current pricing at deploy.
WHISPER_USD_PER_MINUTE = 0.006
# WhatsApp voice notes are opus at roughly 6-12 KB/s; 8 KB/s is the
# midpoint used ONLY to estimate minutes for cost metering.
_AUDIO_BYTES_PER_MINUTE = 8 * 1024 * 60


class VoiceError(Exception):
    """Transcription failed (message already sanitized)."""


def _estimate_minutes(num_bytes: int) -> float:
    return max(1.0 / 60.0, num_bytes / _AUDIO_BYTES_PER_MINUTE)


async def transcribe_audio(
    *,
    audio: bytes,
    mime_type: str,
    key_id: str,
    get_key,  # async (key_id, owner) -> plaintext, from the vault
    owner: str,
    ledger=None,
    filename: str = "voice.ogg",
) -> str:
    """Transcribe audio bytes to text. Returns the transcript.

    Metered: a ledger row is written with the whisper-1 estimate.
    """
    from ..llm import sanitize_secret_text
    from ..llm.client import LiteLLMClient

    if not audio:
        raise VoiceError("empty audio")
    # Rough duration guard: assume >= 8 kbps floor; anything clearly over
    # the cap is rejected before spending on transcription.
    est_seconds = len(audio) / 1024  # ~1KB/s very rough floor for opus
    if est_seconds > MAX_AUDIO_SECONDS * 8:  # generous upper bound
        raise VoiceError(
            f"audio too long (over ~{MAX_AUDIO_SECONDS // 60} min cap)"
        )
    try:
        plaintext = await get_key(key_id, owner)
    except Exception as exc:
        raise VoiceError(
            f"vault: {sanitize_secret_text(str(exc))}"
        ) from exc

    client = LiteLLMClient()
    buf = io.BytesIO(audio)
    buf.name = filename
    try:
        text = await client.atranscribe(
            model=TRANSCRIBE_MODEL,
            audio_file=buf,
            api_key=plaintext,
        )
    except Exception as exc:
        raise VoiceError(
            f"transcription failed: {sanitize_secret_text(str(exc))}"
        ) from exc
    finally:
        # Best-effort memory hygiene for the transient secret.
        plaintext = "x" * len(plaintext)

    transcript = (text or "").strip()
    if ledger is not None:
        try:
            minutes = _estimate_minutes(len(audio))
            cost = round(minutes * WHISPER_USD_PER_MINUTE, 6)
            await ledger.record(
                key_id=key_id,
                provider="openai",
                model=TRANSCRIBE_MODEL,
                input_tokens=0,
                output_tokens=max(1, len(transcript) // 4),
                estimated_cost_usd=cost,
                status="ok",
            )
        except Exception as exc:  # metering must never break the reply
            log.warning("voice: ledger record failed: %s", exc)
    return transcript
