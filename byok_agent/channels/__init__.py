"""WhatsApp (Cloud API) control plane: transports, webhook, worker, pairing,
audit, guard, and voice transcription.

The channel layer is deliberately transport-agnostic: the worker only
speaks the :class:`ChannelTransport` protocol, so a Baileys-style
transport could be plugged in later without touching agent logic.
"""
from .transport import (
    ChannelTransport,
    IncomingMessage,
    MockTransport,
    OutgoingMessage,
)

__all__ = [
    "ChannelTransport",
    "IncomingMessage",
    "MockTransport",
    "OutgoingMessage",
]
