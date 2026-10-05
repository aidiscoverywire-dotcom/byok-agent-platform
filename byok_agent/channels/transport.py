"""Channel abstraction (wappajs pattern).

The worker and pairing flows only speak this protocol — never the
WhatsApp Cloud API directly. That is what lets the transport be swapped
(Baileys, Cloud API, Telegram, ...) without touching agent logic.

An :class:`IncomingMessage` is the NORMALIZED view of one inbound chat
message, whatever the underlying channel's webhook format is. Normalizing
at the transport boundary keeps prompt-injection surface in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class IncomingMessage:
    """One inbound message, channel-normalized."""

    channel: str  # e.g. "whatsapp"
    sender_id: str  # channel-native sender id (WhatsApp: the sender's msisdn)
    message_id: str  # channel-native message id (for dedup)
    kind: str  # "text" | "audio" | "image" | "unknown"
    text: str = ""  # message text / caption ("" for non-text)
    media_id: str = ""  # channel-native media id (audio/image)
    timestamp: str = ""  # ISO-8601 UTC as reported by the channel
    raw: dict[str, Any] = field(default_factory=dict)  # original payload


@dataclass(frozen=True)
class OutgoingMessage:
    """One outbound message."""

    to: str  # channel-native recipient id
    text: str = ""
    media_url: str = ""
    media_caption: str = ""


# Middleware: (transport, outgoing) -> outgoing (may transform or drop by
# raising). Runs on every outbound send, e.g. DLP / logging.
Middleware = Callable[["ChannelTransport", OutgoingMessage], "Awaitable[OutgoingMessage]"]


@runtime_checkable
class ChannelTransport(Protocol):
    """Minimal send/receive contract every channel implements."""

    channel_name: str

    async def send_text(self, to: str, text: str) -> str:
        """Send a text message. Returns the channel's message id."""
        ...

    async def send_media(
        self, to: str, media_url: str, caption: str = ""
    ) -> str:
        """Send media by URL. Returns the channel's message id."""
        ...

    def add_middleware(self, mw: Middleware) -> None:
        """Register an outbound middleware hook."""
        ...


class MockTransport:
    """In-memory transport for tests and offline development.

    Records everything sent; never touches the network.
    """

    channel_name = "mock"

    def __init__(self) -> None:
        self.sent: list[OutgoingMessage] = []
        self._middlewares: list[Middleware] = []
        self._counter = 0

    def add_middleware(self, mw: Middleware) -> None:
        self._middlewares.append(mw)

    async def _apply(self, msg: OutgoingMessage) -> OutgoingMessage:
        for mw in self._middlewares:
            msg = await mw(self, msg)
        return msg

    async def send_text(self, to: str, text: str) -> str:
        msg = await self._apply(OutgoingMessage(to=to, text=text))
        self._counter += 1
        self.sent.append(msg)
        return f"mock-msg-{self._counter}"

    async def send_media(self, to: str, media_url: str, caption: str = "") -> str:
        msg = await self._apply(
            OutgoingMessage(to=to, text=caption, media_url=media_url, media_caption=caption)
        )
        self._counter += 1
        self.sent.append(msg)
        return f"mock-msg-{self._counter}"
