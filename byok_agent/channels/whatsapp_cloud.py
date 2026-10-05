"""Meta WhatsApp Cloud API transport.

Sends through ``https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_number_id}/messages``
with the access token as a Bearer header. The token is read from the
vault at the call boundary and NEVER logged (see the test asserting
``caplog`` stays token-free).

``GRAPH_API_VERSION`` is pinned and documented as *verify at deploy*:
Meta deprecates Graph versions on a rolling schedule, so check the
current version in Meta's changelog before pointing production at this.

Pricing reality (verify Meta's current rate card before billing anyone):
replies inside the 24h customer-service window are free; utility and
marketing template messages are billed per message.
"""
from __future__ import annotations

import logging
from typing import Any

from .transport import ChannelTransport, Middleware, OutgoingMessage

log = logging.getLogger(__name__)

# PINNED Graph API version — verify against Meta's changelog at deploy time.
# Meta retires old Graph versions; a pinned version that Meta has sunset
# will fail every call with a version error.
GRAPH_API_VERSION = "v21.0"

GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


class WhatsAppCloudTransport:
    """Cloud API implementation of :class:`ChannelTransport`.

    Args:
        phone_number_id: the business phone-number id from Meta.
        access_token: the system-user access token — from the vault at the
            call boundary, never stored on the instance longer than needed
            and never logged.
        http_client: injectable async HTTP client (httpx-like) for tests.
    """

    channel_name = "whatsapp"

    def __init__(
        self,
        *,
        phone_number_id: str,
        access_token: str,
        http_client: Any | None = None,
    ) -> None:
        if not phone_number_id or not access_token:
            raise ValueError("phone_number_id and access_token are required")
        self._phone_number_id = phone_number_id
        # Kept only for the lifetime of this transport; callers should
        # construct it per send-batch from the vault, not hold it forever.
        self._access_token = access_token
        self._http = http_client
        self._middlewares: list[Middleware] = []

    # -- ChannelTransport ------------------------------------------------
    def add_middleware(self, mw: Middleware) -> None:
        self._middlewares.append(mw)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        from ..llm import sanitize_secret_text

        if self._http is None:
            import httpx  # lazy: keeps import cost off the CLI path

            client = httpx.AsyncClient(timeout=30.0)
            close_after = True
        else:
            client = self._http
            close_after = False
        url = f"{GRAPH_BASE}/{self._phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            # Sanitize: provider error bodies can echo request fragments.
            # The token itself only ever travels in the header, but the
            # phone-number id and message text are still scrubbed.
            raise WhatsAppAPIError(
                f"Cloud API call failed: {sanitize_secret_text(str(exc))}"
            ) from exc
        finally:
            if close_after:
                await client.aclose()
        return data

    async def _apply(self, msg: OutgoingMessage) -> OutgoingMessage:
        for mw in self._middlewares:
            msg = await mw(self, msg)
        return msg

    async def send_text(self, to: str, text: str) -> str:
        msg = await self._apply(OutgoingMessage(to=to, text=text))
        data = await self._post(
            {
                "messaging_product": "whatsapp",
                "to": msg.to,
                "type": "text",
                "text": {"body": msg.text, "preview_url": False},
            }
        )
        return _first_message_id(data)

    async def send_media(self, to: str, media_url: str, caption: str = "") -> str:
        msg = await self._apply(
            OutgoingMessage(to=to, text=caption, media_url=media_url)
        )
        data = await self._post(
            {
                "messaging_product": "whatsapp",
                "to": msg.to,
                "type": "image",
                "image": {"link": msg.media_url, "caption": msg.text or None},
            }
        )
        return _first_message_id(data)

    async def send_template(
        self,
        to: str,
        template_name: str,
        language: str = "en_US",
        components: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send an approved utility/marketing template (billed per message)."""
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": language},
            },
        }
        if components:
            payload["template"]["components"] = components
        data = await self._post(payload)
        return _first_message_id(data)

    # -- media -----------------------------------------------------------
    async def download_media(self, media_id: str) -> tuple[bytes, str]:
        """Download media bytes: media-id -> URL -> bytes.

        Returns (bytes, mime_type). The media URL itself needs the Bearer
        token; it is never logged.
        """
        from ..llm import sanitize_secret_text

        if self._http is None:
            import httpx

            client = httpx.AsyncClient(timeout=60.0)
            close_after = True
        else:
            client = self._http
            close_after = False
        headers = {"Authorization": f"Bearer {self._access_token}"}
        try:
            meta = await client.get(f"{GRAPH_BASE}/{media_id}", headers=headers)
            meta.raise_for_status()
            media_url = meta.json().get("url", "")
            mime = meta.json().get("mime_type", "application/octet-stream")
            if not media_url:
                raise WhatsAppAPIError("media metadata had no url")
            blob = await client.get(media_url, headers=headers)
            blob.raise_for_status()
            return blob.content, mime
        except Exception as exc:
            raise WhatsAppAPIError(
                f"media download failed: {sanitize_secret_text(str(exc))}"
            ) from exc
        finally:
            if close_after:
                await client.aclose()


def _first_message_id(data: dict[str, Any]) -> str:
    try:
        return str(data["messages"][0]["id"])
    except (KeyError, IndexError, TypeError):
        return ""


class WhatsAppAPIError(Exception):
    """A Cloud API call failed (message already sanitized)."""
