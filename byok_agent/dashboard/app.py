"""Operator dashboard (FastAPI + server-rendered Jinja2).

Built on OUR OWN vault + ledger — deliberately NOT the LiteLLM proxy /ui,
which would create a second key vault contradicting the single-vault design
(user audit decision).

Auth: bearer token from env ``DASHBOARD_TOKEN`` (``Authorization: Bearer``
header only — the token is never accepted via URL query parameters, which
would leak it into server logs, browser history, and Referer headers).
Run behind localhost or TLS; the token is an operator credential, treat it
like one.

Every mutating action is audit-logged (actor ``dashboard``).
"""
from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.templating import Jinja2Templates
from pathlib import Path

from byok_agent.channels.audit import AuditLog
from byok_agent.llm import sanitize_secret_text
from byok_agent.ops.approvals import ApprovalStore
from byok_agent.ops.budgets import KeyBudgetStore
from byok_agent.ops.killswitch import KillSwitch

TEMPLATES_DIR = Path(__file__).parent / "templates"

_bearer = HTTPBearer(auto_error=False)


def _dashboard_token() -> str:
    token = os.environ.get("DASHBOARD_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "DASHBOARD_TOKEN env var is required to run the dashboard"
        )
    return token


async def _require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    expected = _dashboard_token()
    presented = ""
    if credentials is not None:
        presented = credentials.credentials or ""
    # Bytes comparison: str compare_digest raises TypeError on non-ASCII
    # input — an unauthenticated request must 401, never 500.
    if not presented or not hmac.compare_digest(
        presented.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="unauthorized")


class Deps:
    """All dashboard collaborators, wired once at startup."""

    def __init__(
        self,
        *,
        db: Any,
        vault: Any,
        ledger: Any,
        rule_store: Any,
        operator: str = "dashboard",
    ) -> None:
        self.db = db
        self.vault = vault
        self.ledger = ledger
        self.rule_store = rule_store
        self.kill = KillSwitch(db)
        self.approvals = ApprovalStore(db)
        self.key_budgets = KeyBudgetStore(db)
        self.audit = AuditLog(db, _audit_hmac_key())
        self.operator = operator


def _audit_hmac_key() -> bytes:
    master = os.environ.get("APP_VAULT_KEY", "")
    if not master:
        raise RuntimeError("APP_VAULT_KEY is required for the audit log")
    return hashlib.sha256(("byok-audit-v1:" + master).encode()).digest()


def _usd(x: float) -> str:
    return f"${x:.4f}"


def create_app(deps: Deps) -> FastAPI:
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app = FastAPI(title="BYOK Agent — Operator Dashboard", docs_url=None,
                  redoc_url=None)

    async def _log(action: str, args: Any = None) -> None:
        await deps.audit.append(actor=deps.operator, action=action, args=args)

    # -- keys -----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def _root() -> RedirectResponse:
        return RedirectResponse(url="/keys")

    @app.get("/keys", response_class=HTMLResponse)
    async def keys_page(
        request: Request, owner: str = "", _: None = Depends(_require_auth)
    ):
        keys = await deps.vault.list_keys(owner=owner or None)
        return templates.TemplateResponse(
            "keys.html",
            {"request": request, "keys": keys, "owner_filter": owner},
        )

    @app.post("/keys/add")
    async def keys_add(
        owner: str = Form(...),
        provider: str = Form(...),
        label: str = Form(...),
        secret: str = Form(...),
        _: None = Depends(_require_auth),
    ):
        rec = await deps.vault.add_key(
            label=label.strip(),
            provider=provider.strip().lower(),
            plaintext_key=secret,
            owner=owner.strip(),
        )
        await _log("key.add", {"key_id": rec.key_id, "provider": rec.provider,
                               "owner": owner})
        return RedirectResponse(url="/keys", status_code=303)

    @app.post("/keys/{key_id}/rotate")
    async def keys_rotate(
        key_id: str, secret: str = Form(...), _: None = Depends(_require_auth)
    ):
        rec = await deps.vault.rotate_key(key_id, secret)
        await _log("key.rotate", {"key_id": key_id,
                                  "version": rec.key_version})
        return RedirectResponse(url="/keys", status_code=303)

    @app.post("/keys/{key_id}/revoke")
    async def keys_revoke(key_id: str, _: None = Depends(_require_auth)):
        await deps.vault.revoke_key(key_id)
        await _log("key.revoke", {"key_id": key_id})
        return RedirectResponse(url="/keys", status_code=303)

    # -- spend ----------------------------------------------------------
    @app.get("/spend", response_class=HTMLResponse)
    async def spend_page(request: Request, _: None = Depends(_require_auth)):
        per_key = await deps.db.fetch(
            "SELECT key_id, provider, COUNT(*) AS calls,"
            " SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok,"
            " SUM(estimated_cost_usd) AS usd"
            " FROM ledger GROUP BY key_id, provider ORDER BY usd DESC"
        )
        per_provider = await deps.db.fetch(
            "SELECT provider, COUNT(*) AS calls, SUM(estimated_cost_usd) AS usd"
            " FROM ledger GROUP BY provider ORDER BY usd DESC"
        )
        per_day = await deps.db.fetch(
            "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS calls,"
            " SUM(estimated_cost_usd) AS usd FROM ledger"
            " GROUP BY day ORDER BY day DESC LIMIT 30"
        )
        caps = await deps.key_budgets.list_caps()
        cap_map = {c["key_id"]: c for c in caps}
        return templates.TemplateResponse(
            "spend.html",
            {
                "request": request,
                "per_key": [dict(r) for r in per_key],
                "per_provider": [dict(r) for r in per_provider],
                "per_day": [dict(r) for r in per_day],
                "cap_map": cap_map,
                "usd": _usd,
            },
        )

    @app.post("/budgets/set")
    async def budgets_set(
        key_id: str = Form(...),
        daily_usd_cap: float = Form(...),
        _: None = Depends(_require_auth),
    ):
        await deps.key_budgets.set_cap(key_id, daily_usd_cap)
        await _log("budget.set", {"key_id": key_id, "daily_usd_cap": daily_usd_cap})
        return RedirectResponse(url="/spend", status_code=303)

    @app.post("/budgets/remove")
    async def budgets_remove(
        key_id: str = Form(...), _: None = Depends(_require_auth)
    ):
        await deps.key_budgets.remove_cap(key_id)
        await _log("budget.remove", {"key_id": key_id})
        return RedirectResponse(url="/spend", status_code=303)

    # -- runs -----------------------------------------------------------
    @app.get("/runs", response_class=HTMLResponse)
    async def runs_page(request: Request, _: None = Depends(_require_auth)):
        rows = await deps.db.fetch(
            "SELECT id, created_at, key_id, provider, model, input_tokens,"
            " output_tokens, estimated_cost_usd, status, task_id, user_id"
            " FROM ledger ORDER BY id DESC LIMIT 100"
        )
        audit_rows = await deps.db.fetch(
            "SELECT seq, ts, actor, action FROM audit_log"
            " ORDER BY seq DESC LIMIT 50"
        )
        return templates.TemplateResponse(
            "runs.html",
            {
                "request": request,
                "ledger": [dict(r) for r in rows],
                "audit": [dict(r) for r in audit_rows],
                "usd": _usd,
            },
        )

    # -- kill switch ----------------------------------------------------
    @app.get("/kill", response_class=HTMLResponse)
    async def kill_page(request: Request, _: None = Depends(_require_auth)):
        state = await deps.kill.state()
        return templates.TemplateResponse(
            "kill.html", {"request": request, "state": state}
        )

    @app.post("/kill/toggle")
    async def kill_toggle(
        action: str = Form(...),
        reason: str = Form(""),
        by: str = Form("operator"),
        _: None = Depends(_require_auth),
    ):
        if action == "on":
            await deps.kill.engage(by=by, reason=reason)
            await _log("kill.engage", {"by": by, "reason": reason})
        else:
            await deps.kill.disengage(by=by)
            await _log("kill.disengage", {"by": by})
        return RedirectResponse(url="/kill", status_code=303)

    # -- routing rules --------------------------------------------------
    @app.get("/routing", response_class=HTMLResponse)
    async def routing_page(request: Request, _: None = Depends(_require_auth)):
        rules = await deps.rule_store.list_rules()
        return templates.TemplateResponse(
            "routing.html", {"request": request, "rules": rules}
        )

    @app.post("/routing/add")
    async def routing_add(
        key_id: str = Form(...),
        model_pattern: str = Form(...),
        _: None = Depends(_require_auth),
    ):
        rule_id = await deps.rule_store.add_rule(key_id, model_pattern)
        await _log("routing.add", {"rule_id": rule_id, "key_id": key_id,
                                   "pattern": model_pattern})
        return RedirectResponse(url="/routing", status_code=303)

    @app.post("/routing/{rule_id}/delete")
    async def routing_delete(rule_id: int, _: None = Depends(_require_auth)):
        await deps.rule_store.remove_rule(rule_id)
        await _log("routing.delete", {"rule_id": rule_id})
        return RedirectResponse(url="/routing", status_code=303)

    # -- approvals ------------------------------------------------------
    @app.get("/approvals", response_class=HTMLResponse)
    async def approvals_page(request: Request, _: None = Depends(_require_auth)):
        pending = await deps.approvals.list_pending()
        return templates.TemplateResponse(
            "approvals.html", {"request": request, "pending": pending}
        )

    @app.post("/approvals/{approval_id}/decide")
    async def approvals_decide(
        approval_id: int,
        approved: str = Form(...),
        by: str = Form("operator"),
        _: None = Depends(_require_auth),
    ):
        from byok_agent.agent.builder_graph import apply_approval

        rec = await deps.approvals.get(approval_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="approval not found")
        decision = approved.lower() in ("true", "1", "yes", "on")
        if rec.kind == "build-promote":
            # Applies staged files to the real project dir on approve.
            await apply_approval(deps.db, approval_id, approved=decision,
                                 decided_by=by)
        else:
            await deps.approvals.decide(approval_id, approved=decision,
                                        decided_by=by)
        await _log(
            "approval.decide",
            {"approval_id": approval_id, "kind": rec.kind,
             "approved": decision, "by": by},
        )
        return RedirectResponse(url="/approvals", status_code=303)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"ok": True}

    return app
