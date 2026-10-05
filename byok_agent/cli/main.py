"""BYOK Agent Platform — Phase 4 CLI (productization).

Usage (after ``pip install -e .``)::

    byok add-key --user u1 --provider openai --label personal
    byok list-keys --user u1
    byok run --user u1 --task t1 --key-id key_openai_ab12 --model openai/gpt-4o-mini "what time is it?"
    byok sandbox exec --lang python -- "python -c 'print(1+1)'"
    byok codemap ./myproj
    byok build --user u1 --task t1 --dir ./myproj --key-id key_openai_ab12 "add a /health endpoint"
    byok build --user u1 --task t1 --dir ./fresh --key-id key_openai_ab12 --template fastapi-min "add auth"
    byok revoke-key key_openai_ab12
    byok rotate-key key_openai_ab12
    byok dashboard --port 8080                       # operator web UI
    byok kill-switch on --reason "incident"          # or: off
    byok template list
    byok deploy --platform auto --dir ./myproj --user u1 --token key_vercel_x=VERCEL_TOKEN
    byok approval list
    byok approval decide 3 --approve --by ops

Environment (see .env.example):
    APP_VAULT_KEY          Fernet master key (required)
    APP_VAULT_KEY_PREVIOUS previous master key (rotation only)
    DATABASE_URL           Postgres DSN; if unset, local SQLite is used
    BYOK_SQLITE_PATH       SQLite file (default ./byok_phase1.db)
    AGENT_KILL_SWITCH      "1" stops all new work immediately
    AGENT_WORKSPACE        sandbox dir for read_file (default ./workspace)
    DEFAULT_MODEL          default model string for `run` / `build`

Tenant model: every key belongs to exactly one ``--user`` (owner). Users
can only list, run with, or manage their own keys — cross-tenant key ids
behave as if they don't exist.

Phase 1 notes: `sandbox exec` needs a Docker daemon; without one it fails
with a clear error (see README). `build` runs the planner -> coder ->
tester -> reviewer graph inside a sandbox and asks for approval before
promoting changes to the real project directory.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _open_db():
    """Return (Database, backend_name)."""
    from byok_agent.db.postgres import PostgresDB
    from byok_agent.db.sqlite import SqliteDB

    dsn = os.environ.get("DATABASE_URL", "").strip()
    if dsn:
        return PostgresDB(dsn), "postgres"
    path = os.environ.get("BYOK_SQLITE_PATH", "./byok_phase1.db")
    return SqliteDB(path), "sqlite"


async def _open_vault_and_ledger():
    """Open one DB connection shared by the vault and the ledger.

    Applies the schema (idempotent, incl. migrations) first.
    Returns (db, vault, ledger, backend). Caller must ``await db.close()``.

    The ledger is wired to a KeyBudgetStore so every recorded call accrues
    into the key's daily budget and the enforcer's gate enforces the
    dashboard-managed daily caps.
    """
    from byok_agent.db import apply_schema
    from byok_agent.ledger import Ledger
    from byok_agent.ops.budgets import KeyBudgetStore
    from byok_agent.vault import FernetVault

    db, backend = _open_db()
    await apply_schema(db, backend)
    return db, FernetVault(db), Ledger(db, key_budgets=KeyBudgetStore(db)), backend


def _run(coro):
    return asyncio.run(coro)


def _kill_checker(db):
    """DB kill-switch checker for BudgetEnforcer gates (cached <=5s)."""
    from byok_agent.ops.killswitch import cached_checker

    return cached_checker(db)


def _wire_router(router, db):
    """Attach dashboard-managed routing rules to a Router."""
    from byok_agent.llm import RoutingRuleStore

    router.set_rule_store(RoutingRuleStore(db))
    return router


def _fail(msg: str) -> int:
    from byok_agent.llm import sanitize_secret_text

    print(f"ERROR: {sanitize_secret_text(msg)}", file=sys.stderr)
    return 1


# -- subcommands ------------------------------------------------------------
async def cmd_add_key(args: argparse.Namespace) -> int:
    from byok_agent.vault import VaultNotConfiguredError

    secret = getpass.getpass(f"API key for provider '{args.provider}': ").strip()
    if not secret:
        print("empty key — aborted.", file=sys.stderr)
        return 2
    try:
        db, vault, _ledger, _backend = await _open_vault_and_ledger()
    except VaultNotConfiguredError as exc:
        return _fail(str(exc))
    try:
        record = await vault.add_key(
            label=args.label,
            provider=args.provider,
            plaintext_key=secret,
            owner=args.user,
            allow_cross_provider_fallback=args.allow_cross_provider_fallback,
        )
    finally:
        await db.close()
    print(f"stored: key_id={record.key_id} provider={record.provider} "
          f"owner={record.owner} label={record.label} version={record.key_version}")
    print("The plaintext was encrypted at rest; it is never shown again.")
    return 0


async def cmd_list_keys(args: argparse.Namespace) -> int:
    db, vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        records = await vault.list_keys(owner=args.user)
    finally:
        await db.close()
    if not records:
        print(f"no keys stored for user {args.user!r}.")
        return 0
    print(f"{'key_id':40} {'provider':12} {'label':16} {'status':8} "
          f"{'ver':3} fallback")
    for r in records:
        print(f"{r.key_id:40} {r.provider:12} {r.label:16} {r.status.value:8} "
              f"{r.key_version:<3} {r.allow_cross_provider_fallback}")
    print("\nSecrets are never printed. Use get_key only at the call boundary.")
    return 0


async def cmd_revoke_key(args: argparse.Namespace) -> int:
    from byok_agent.vault import KeyNotFoundError

    db, vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        await vault.revoke_key(args.key_id)
    except KeyNotFoundError as exc:
        return _fail(str(exc))
    finally:
        await db.close()
    print(f"revoked {args.key_id} — immediate, no draining; it can never be used again.")
    return 0


async def cmd_rotate_key(args: argparse.Namespace) -> int:
    from byok_agent.vault import ConcurrentModificationError, KeyNotFoundError, RevokedKeyError

    secret = getpass.getpass("New API key: ").strip()
    if not secret:
        print("empty key — aborted.", file=sys.stderr)
        return 2
    db, vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        record = await vault.rotate_key(args.key_id, secret)
    except (KeyNotFoundError, RevokedKeyError, ConcurrentModificationError) as exc:
        return _fail(str(exc))
    finally:
        await db.close()
    print(f"rotated {args.key_id} -> version {record.key_version}")
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    from byok_agent.agent import AgentRunner, TaskBudget, make_checkpointer
    from byok_agent.agent.budgets import (
        BudgetEnforcer,
        BudgetExceeded,
        KillSwitchEngaged,
    )
    from byok_agent.llm import (
        AllTargetsFailed,
        FallbackNotConsented,
        LiteLLMClient,
        LLMError,
        ProviderMismatchError,
        RouteTarget,
        Router,
    )
    from byok_agent.vault import KeyNotFoundError, RevokedKeyError, VaultError

    db, vault, ledger, _backend = await _open_vault_and_ledger()
    try:
        # Fail fast on unknown/revoked/foreign keys before spending anything.
        try:
            record = await vault.get_record(args.key_id, owner=args.user)
        except (KeyNotFoundError, RevokedKeyError) as exc:
            return _fail(str(exc))
        if record.status.value != "active":
            return _fail(f"key {args.key_id} is {record.status.value}")

        model = args.model or os.environ.get("DEFAULT_MODEL", "openai/gpt-4o-mini")
        router = _wire_router(Router(LiteLLMClient(), vault), db)
        primary = RouteTarget(key_id=args.key_id, model=model)
        budget = TaskBudget(
            max_steps=args.max_steps, max_tokens=args.max_tokens, max_usd=args.max_usd
        )

        def make_enforcer(b: TaskBudget) -> BudgetEnforcer:
            return BudgetEnforcer(
                b,
                ledger=ledger,
                key_id=args.key_id,
                key_max_usd=args.key_max_usd,
                kill_switch_checker=_kill_checker(db),
            )

        dsn = os.environ.get("DATABASE_URL", "").strip() or None
        try:
            async with make_checkpointer(dsn) as checkpointer:
                runner = AgentRunner(
                    router=router,
                    ledger=ledger,
                    make_enforcer=make_enforcer,
                    primary=primary,
                    checkpointer=checkpointer,
                )
                result = await runner.run(
                    uid=args.user, tid=args.task, user_message=args.message, budget=budget
                )
        except (
            BudgetExceeded,
            KillSwitchEngaged,
            LLMError,  # covers AllTargetsFailed + ProviderMismatchError
            FallbackNotConsented,
            VaultError,
        ) as exc:
            return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        await db.close()
    print(f"--- thread {result['thread_id']} ---")
    print(result.get("answer") or "(no ANSWER produced)")
    b = result["budget"]
    print(f"[budget] steps={b['steps']} tokens={b['tokens']} "
          f"usd_est=${b['usd']:.4f} (estimate, not billing truth)")
    return 0


# -- Phase 1 subcommands ----------------------------------------------------
async def cmd_sandbox_exec(args: argparse.Namespace) -> int:
    """Ad-hoc command inside a throwaway sandbox container."""
    from byok_agent.sandbox import DockerSandbox, is_docker_available

    if not is_docker_available():
        return _fail(
            "Docker daemon not reachable. `sandbox exec` needs Docker "
            "(see README Phase 1 setup)."
        )
    sbx = DockerSandbox(
        language=args.lang,
        install_network=args.install_network,
    )
    await sbx.start()
    try:
        result = await sbx.exec(args.command, timeout_s=args.timeout)
    finally:
        await sbx.stop()
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n",
              file=sys.stderr)
    print(f"[exit={result.exit_code}]")
    return result.exit_code


async def cmd_codemap(args: argparse.Namespace) -> int:
    from byok_agent.codemap import RepoMap

    m = RepoMap(token_budget=args.budget).build(args.dir, changed_files=args.changed)
    print(m)
    return 0


async def _prompt_approver(diff: str) -> bool:
    print("=" * 70)
    print("PROPOSED CHANGES (sandbox -> real project dir):")
    print("=" * 70)
    print(diff or "(no diff)")
    print("=" * 70)
    try:
        ans = await asyncio.to_thread(
            input, "Apply these changes to the REAL project dir? [y/N]: "
        )
    except EOFError:
        return False
    return ans.strip().lower() in ("y", "yes")


async def cmd_build(args: argparse.Namespace) -> int:
    from byok_agent.agent import TaskBudget
    from byok_agent.agent.budgets import BudgetEnforcer, BudgetExceeded, KillSwitchEngaged
    from byok_agent.agent.builder_graph import (
        ApprovalNeeded,
        BuildError,
        BuilderRunner,
    )
    from byok_agent.llm import (
        AllTargetsFailed,
        FallbackNotConsented,
        LiteLLMClient,
        LLMError,
        RouteTarget,
        Router,
    )
    from byok_agent.sandbox import DockerSandbox, is_docker_available
    from byok_agent.vault import KeyNotFoundError, RevokedKeyError, VaultError

    if not is_docker_available():
        return _fail(
            "Docker daemon not reachable. `build` needs Docker "
            "(see README Phase 1 setup)."
        )

    db, vault, ledger, _backend = await _open_vault_and_ledger()
    try:
        # Optional: scaffold a starter template into the project dir first.
        if getattr(args, "template", None):
            from byok_agent.templates import scaffold

            scaffold(args.template, args.dir)
            print(f"scaffolded template {args.template!r} into {args.dir}")
        try:
            record = await vault.get_record(args.key_id, owner=args.user)
        except (KeyNotFoundError, RevokedKeyError) as exc:
            return _fail(str(exc))
        if record.status.value != "active":
            return _fail(f"key {args.key_id} is {record.status.value}")

        model = args.model or os.environ.get("DEFAULT_MODEL", "openai/gpt-4o-mini")
        router = _wire_router(Router(LiteLLMClient(), vault), db)
        primary = RouteTarget(key_id=args.key_id, model=model)
        budget = TaskBudget(
            max_steps=args.max_steps, max_tokens=args.max_tokens, max_usd=args.max_usd
        )

        def make_enforcer(b: TaskBudget) -> BudgetEnforcer:
            return BudgetEnforcer(
                b,
                ledger=ledger,
                key_id=args.key_id,
                key_max_usd=args.key_max_usd,
                kill_switch_checker=_kill_checker(db),
            )

        from byok_agent.ops.approvals import ApprovalStore

        approval_store = ApprovalStore(db)

        dsn = os.environ.get("DATABASE_URL", "").strip() or None
        try:
            from byok_agent.agent import make_checkpointer

            async with make_checkpointer(dsn) as checkpointer:
                runner = BuilderRunner(
                    router=router,
                    ledger=ledger,
                    make_enforcer=make_enforcer,
                    primary=primary,
                    checkpointer=checkpointer,
                    auto_approve=args.auto_approve,
                    approver=None if args.auto_approve else _prompt_approver,
                    approval_store=approval_store,
                )
                result = await runner.run(
                    uid=args.user,
                    tid=args.task,
                    project_dir=args.dir,
                    task_instruction=args.message,
                    language=args.lang,
                    test_command=args.test_command,
                    budget=budget,
                )
        except ApprovalNeeded as exc:
            if exc.approval_id is not None:
                print(
                    f"approval #{exc.approval_id} recorded."
                    " Decide with: byok approval decide"
                    f" {exc.approval_id} --approve|--deny --by <name>",
                    file=sys.stderr,
                )
            else:
                print("approval required but no approver configured.", file=sys.stderr)
                print(exc.diff)
            return _fail("aborted at approval gate")
        except (
            BudgetExceeded,
            KillSwitchEngaged,
            BuildError,
            LLMError,
            FallbackNotConsented,
            VaultError,
        ) as exc:
            return _fail(f"{type(exc).__name__}: {exc}")
    finally:
        await db.close()
    print(f"--- thread {result['thread_id']} ---")
    print(result.get("answer") or "(no answer)")
    b = result["budget"]
    print(f"[budget] steps={b['steps']} tokens={b['tokens']} "
          f"usd_est=${b['usd']:.4f} (estimate, not billing truth)")
    return 0


    print(f"[budget] steps={b['steps']} tokens={b['tokens']} "
          f"usd_est=${b['usd']:.4f} (estimate, not billing truth)")
    return 0


# -- Phase 2 subcommands: WhatsApp control plane ---------------------------
def _audit_hmac_key() -> bytes:
    """Audit-log HMAC key: derived from the vault master key material.

    Same-host limitation documented in channels/audit.py: this makes the
    log tamper-evident against row edits, not full-chain forgery.
    """
    import hashlib
    import os

    master = os.environ.get("APP_VAULT_KEY", "")
    if not master:
        raise RuntimeError("APP_VAULT_KEY is required for the audit log")
    return hashlib.sha256(("byok-audit-v1:" + master).encode()).digest()


async def _load_wa_config(db, owner: str) -> dict | None:
    row = await db.fetchrow(
        "SELECT owner, waba_id, phone_number_id, verify_token,"
        " access_token_key_id, app_secret_key_id, agent_key_id, agent_model,"
        " max_steps, max_tokens, max_usd"
        " FROM whatsapp_config WHERE owner=$1",
        owner,
    )
    return dict(row) if row else None


async def _make_wa_transport(db, vault, owner: str, http_client=None):
    """Build a WhatsAppCloudTransport for owner (token from vault)."""
    from byok_agent.channels.whatsapp_cloud import WhatsAppCloudTransport

    config = await _load_wa_config(db, owner)
    if not config:
        raise RuntimeError(f"no WhatsApp config for user {owner!r} (run whatsapp setup)")
    token = await vault.get_key(config["access_token_key_id"], owner=owner)
    try:
        return WhatsAppCloudTransport(
            phone_number_id=config["phone_number_id"],
            access_token=token,
            http_client=http_client,
        )
    finally:
        token = "x" * 8  # best-effort: don't hold the secret


async def cmd_whatsapp_setup(args: argparse.Namespace) -> int:
    from byok_agent.vault import VaultNotConfiguredError

    print("WhatsApp Cloud API setup — values are stored encrypted in the vault.")
    print("(The Meta-side steps are printed at the end; the agent cannot do those.)")
    waba_id = input("WABA ID: ").strip()
    phone_number_id = input("Phone number ID: ").strip()
    verify_token = getpass.getpass("Verify token (for Meta webhook handshake): ").strip()
    app_secret = getpass.getpass("App secret: ").strip()
    access_token = getpass.getpass("Access token (system user): ").strip()
    if not all([waba_id, phone_number_id, verify_token, app_secret, access_token]):
        return _fail("all fields are required — aborted")
    try:
        db, vault, _ledger, _backend = await _open_vault_and_ledger()
    except VaultNotConfiguredError as exc:
        return _fail(str(exc))
    try:
        token_rec = await vault.add_key(
            label="whatsapp-access-token", provider="whatsapp",
            plaintext_key=access_token, owner=args.user,
        )
        secret_rec = await vault.add_key(
            label="whatsapp-app-secret", provider="whatsapp",
            plaintext_key=app_secret, owner=args.user,
        )
        # The verify token is a shared secret too: vault, not plaintext.
        verify_rec = await vault.add_key(
            label="whatsapp-verify-token", provider="whatsapp",
            plaintext_key=verify_token, owner=args.user,
        )
        existing = await _load_wa_config(db, args.user)
        if existing is None:
            await db.execute(
                "INSERT INTO whatsapp_config (owner, waba_id, phone_number_id,"
                " verify_token, verify_token_key_id, access_token_key_id,"
                " app_secret_key_id, agent_key_id, agent_model)"
                " VALUES ($1,$2,$3,'',$4,$5,$6,$7,$8)",
                args.user, waba_id, phone_number_id,
                verify_rec.key_id, token_rec.key_id, secret_rec.key_id,
                args.agent_key_id, args.agent_model,
            )
        else:
            await db.execute(
                "UPDATE whatsapp_config SET waba_id=$1, phone_number_id=$2,"
                " verify_token='', verify_token_key_id=$3, access_token_key_id=$4,"
                " app_secret_key_id=$5,"
                " agent_key_id=COALESCE($6, agent_key_id), agent_model=$7"
                " WHERE owner=$8",
                waba_id, phone_number_id,
                verify_rec.key_id, token_rec.key_id, secret_rec.key_id,
                args.agent_key_id, args.agent_model, args.user,
            )
    finally:
        await db.close()
    print("\nStored. NOW DO THE META-SIDE STEPS YOURSELF (the agent cannot):")
    print("  1. developers.facebook.com -> create/choose a Meta app, add the")
    print("     WhatsApp product.")
    print("  2. Create or select your WhatsApp Business Account (WABA); complete")
    print("     business verification if Meta requires it.")
    print("  3. Add your BUSINESS phone number to the WABA. (Limitation: a number")
    print("     on the Cloud API cannot also run the WhatsApp mobile app, and the")
    print("     API only sees messages sent TO this business number — never your")
    print("     personal chats.)")
    print("  4. Create a system user, grant whatsapp_business_messaging, copy the")
    print("     access token (stored above).")
    print("  5. In the app's WhatsApp > Configuration, set the webhook:")
    print("       Callback URL: https://<your-public-host>/webhook")
    print("       Verify token: the value you just entered")
    print("     (For testing, expose local port with ngrok; production needs a")
    print("     real domain + TLS.)")
    print("  6. Subscribe the webhook to the 'messages' field, then send a test")
    print("     message to the business number.")
    print("  7. Start the server: byok whatsapp serve --port 8000")
    if not args.agent_key_id:
        print("\nNOTE: no --agent-key-id given. Add one later with:")
        print("  byok add-key --user <you> --provider openai --label agent")
        print("  then re-run setup with --agent-key-id <key-id>.")
    return 0


async def cmd_whatsapp_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from byok_agent.channels.webhook import create_app

    db, vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        rows = await db.fetch(
            "SELECT owner, phone_number_id, verify_token, verify_token_key_id,"
            " app_secret_key_id FROM whatsapp_config"
        )
        if not rows:
            return _fail("no WhatsApp configs — run 'byok whatsapp setup' first")
        verify_tokens: dict[str, str] = {}
        app_secrets: dict[str, str] = {}
        phone_owner: dict[str, str] = {}
        for row in rows:
            owner = str(row["owner"])
            # Verify token: prefer the vault record; fall back to the legacy
            # plaintext column (pre-fix setups) with a loud warning.
            vt_key_id = str(row.get("verify_token_key_id") or "")
            if vt_key_id:
                verify_tokens[owner] = await vault.get_key(vt_key_id, owner=owner)
            elif str(row.get("verify_token") or ""):
                print(
                    f"WARNING: owner {owner!r} has a plaintext verify_token"
                    " (legacy). Re-run 'byok whatsapp setup' to move it into"
                    " the vault.",
                    file=sys.stderr,
                )
                verify_tokens[owner] = str(row["verify_token"])
            else:
                await db.close()
                return _fail(
                    f"owner {owner!r} has no verify token — re-run"
                    " 'byok whatsapp setup'"
                )
            secret = await vault.get_key(str(row["app_secret_key_id"]), owner=owner)
            app_secrets[owner] = secret
            secret = "x" * 8
            phone_owner[str(row["phone_number_id"])] = owner
    finally:
        await db.close()

    import asyncio

    from byok_agent.channels import IncomingMessage

    queue: asyncio.Queue[tuple[str | None, IncomingMessage]] = asyncio.Queue(
        maxsize=1000
    )
    app = create_app(
        verify_tokens=verify_tokens,
        app_secrets=app_secrets,
        phone_owner=phone_owner,
        queue=queue,
    )

    async def _serve() -> None:
        # The worker needs its own DB/vault handles inside the task.
        from byok_agent.channels.audit import AuditLog
        from byok_agent.channels.guard import ToolGuard
        from byok_agent.channels.pairing import PairingStore
        from byok_agent.channels.worker import WhatsAppWorker

        db2, vault2, ledger2, _b2 = await _open_vault_and_ledger()
        try:
            pairing = PairingStore(db2)
            audit = AuditLog(db2, _audit_hmac_key())
            guard = ToolGuard()

            async def transport_factory(owner: str):
                return await _make_wa_transport(db2, vault2, owner)

            async def whatsapp_config(owner: str):
                return await _load_wa_config(db2, owner)

            async def agent_runner_factory(owner, thread_id, text, budget_kwargs):
                # Reuse the Phase 0 AgentRunner per message.
                from byok_agent.agent import AgentRunner, TaskBudget, make_checkpointer
                from byok_agent.agent.budgets import BudgetEnforcer
                from byok_agent.llm import LiteLLMClient, RouteTarget, Router

                cfg = await whatsapp_config(owner)
                if not cfg or not cfg.get("agent_key_id"):
                    raise RuntimeError("agent key not configured")
                router = Router(LiteLLMClient(), vault2)
                primary = RouteTarget(
                    key_id=str(cfg["agent_key_id"]),
                    model=str(cfg.get("agent_model") or "openai/gpt-4o-mini"),
                )
                b = TaskBudget(
                    max_steps=int(budget_kwargs.get("max_steps", 15)),
                    max_tokens=int(budget_kwargs.get("max_tokens", 30_000)),
                    max_usd=float(budget_kwargs.get("max_usd", 1.0)),
                )

                def make_enforcer(bb: TaskBudget) -> BudgetEnforcer:
                    return BudgetEnforcer(
                        bb, ledger=ledger2, key_id=primary.key_id,
                        key_max_usd=None,
                        kill_switch_checker=_kill_checker(db2),
                    )

                dsn = os.environ.get("DATABASE_URL", "").strip() or None
                async with make_checkpointer(dsn) as checkpointer:
                    runner = AgentRunner(
                        router=router, ledger=ledger2,
                        make_enforcer=make_enforcer, primary=primary,
                        checkpointer=checkpointer,
                    )
                    uid = thread_id.split(":")[1]
                    tid = ":".join(thread_id.split(":")[3:])
                    result = await runner.run(
                        uid=uid, tid=tid, user_message=text, budget=b
                    )
                return str(result.get("answer") or "")

            worker = WhatsAppWorker(
                db=db2, vault=vault2, ledger=ledger2, pairing=pairing,
                audit=audit, guard=guard,
                transport_factory=transport_factory,
                agent_runner_factory=agent_runner_factory,
                whatsapp_config=whatsapp_config,
            )
            worker_task = asyncio.create_task(worker.run_forever(queue))
            config = uvicorn.Config(
                app, host=args.host, port=args.port, log_level="info"
            )
            server = uvicorn.Server(config)
            try:
                await server.serve()
            finally:
                worker.stop()
                worker_task.cancel()
        finally:
            await db2.close()

    print(f"webhook listening on http://{args.host}:{args.port} "
          f"(Meta callback path: /webhook)")
    try:
        await _serve()  # already inside the event loop (no nested asyncio.run)
    except KeyboardInterrupt:
        print("\nstopped.", file=sys.stderr)
    return 0


async def cmd_whatsapp_pair_code(args: argparse.Namespace) -> int:
    from byok_agent.channels.pairing import PAIRING_CODE_TTL_MINUTES, PairingStore

    db, _vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        code = await PairingStore(db).issue_code(args.user)
    finally:
        await db.close()
    print(f"pairing code for --user {args.user}: {code}")
    print(f"single-use, expires in {PAIRING_CODE_TTL_MINUTES} minutes.")
    print("The sender must message the business number with this code.")
    return 0


async def cmd_whatsapp_send(args: argparse.Namespace) -> int:
    db, vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        transport = await _make_wa_transport(db, vault, args.user)
        mid = await transport.send_text(args.to, args.text)
    except Exception as exc:
        return _fail(f"send failed: {exc}")
    finally:
        await db.close()
    print(f"sent (message id: {mid or 'n/a'})")
    return 0


async def cmd_whatsapp_senders(args: argparse.Namespace) -> int:
    db, _vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        rows = await db.fetch(
            "SELECT sender_id, owner, status, paired_at FROM whatsapp_senders"
            " ORDER BY status, sender_id"
        )
    finally:
        await db.close()
    if not rows:
        print("(no senders yet)")
        return 0
    for row in rows:
        print(f"{row['sender_id']}\t{row['status']}\t"
              f"owner={row['owner'] or '-'}\tpaired_at={row['paired_at'] or '-'}")
    return 0


async def cmd_whatsapp_audit(args: argparse.Namespace) -> int:
    from byok_agent.channels.audit import AuditLog

    db, _vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        ok, detail = await AuditLog(db, _audit_hmac_key()).verify()
    finally:
        await db.close()
    print(f"audit chain: {'OK' if ok else 'BROKEN'} — {detail}")
    if not ok:
        print("See channels/audit.py for the tamper model: row edits and")
        print("mid-chain deletions are detected; full-chain rebuild is not")
        print("(use an external anchor for that).")
        return 1
    return 0


# -- Phase 3 subcommands: research & articles -----------------------------------

async def _make_research_pipeline(args: argparse.Namespace, *, max_usd: float):
    """Open DB/vault/ledger and wire a ResearchPipeline. Returns (pipeline, db).

    Caller must ``await db.close()``. Raises clean errors for bad keys.
    """
    from byok_agent.agent import TaskBudget
    from byok_agent.agent.budgets import BudgetEnforcer
    from byok_agent.llm import LiteLLMClient, RouteTarget, Router
    from byok_agent.research import ResearchPipeline
    from byok_agent.vault import KeyNotFoundError, RevokedKeyError

    db, vault, ledger, _backend = await _open_vault_and_ledger()
    try:
        record = await vault.get_record(args.key_id, owner=args.user)
    except (KeyNotFoundError, RevokedKeyError) as exc:
        await db.close()
        raise _CLIError(str(exc))
    if record.status.value != "active":
        await db.close()
        raise _CLIError(f"key {args.key_id} is {record.status.value}")

    model = args.model or os.environ.get("DEFAULT_MODEL", "openai/gpt-4o-mini")
    router = _wire_router(Router(LiteLLMClient(), vault), db)
    primary = RouteTarget(key_id=args.key_id, model=model)
    budget = TaskBudget(max_steps=200, max_tokens=300_000, max_usd=max_usd)
    enforcer = BudgetEnforcer(
        budget,
        ledger=ledger,
        key_id=args.key_id,
        key_max_usd=args.key_max_usd,
        kill_switch_checker=_kill_checker(db),
    )
    pipeline = ResearchPipeline(
        router=router, primary=primary, owner=args.user, vault=vault,
        ledger=ledger, enforcer=enforcer, model=model,
    )
    return pipeline, db


class _CLIError(Exception):
    """Clean, user-facing CLI failure (no traceback)."""


def _run_slug(topic: str) -> str:
    import re
    from datetime import datetime, timezone

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40] or "topic"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{slug}-{ts}"


async def cmd_research(args: argparse.Namespace) -> int:
    from byok_agent.research.search import SEARCH_PROVIDER_NAMES

    if args.topic.strip().lower() == "providers":
        db, vault, _ledger, _backend = await _open_vault_and_ledger()
        try:
            records = await vault.list_keys(owner=args.user)
            found = [r for r in records if r.provider in SEARCH_PROVIDER_NAMES]
            if not found:
                print("no search providers configured. Add one with:")
                print("  byok add-key --provider tavily --label <label> --user "
                      + args.user)
                return 0
            for r in found:
                print(f"{r.provider:8} {r.key_id}  label={r.label}  "
                      f"status={r.status.value}  (secret never shown)")
            return 0
        finally:
            await db.close()

    # `byok research "topic"` — plan + gather + outline, save the brief.
    if not args.key_id:
        return _fail("--key-id is required for topic research "
                     "(not needed for 'research providers')")
    max_usd = args.max_usd
    try:
        pipeline, db = await _make_research_pipeline(args, max_usd=max_usd)
    except _CLIError as exc:
        return _fail(str(exc))
    try:
        from byok_agent.research import SearchNotConfigured
        try:
            brief = await pipeline.research_brief(
                args.topic, depth=args.depth, max_usd=max_usd,
                primary_keyword=args.keyword or "",
            )
        except SearchNotConfigured as exc:
            return _fail(str(exc))
        outdir = os.path.join("research_runs", _run_slug(args.topic))
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "brief.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(brief.brief_md)
    finally:
        await db.close()
    print(f"research brief saved: {path}")
    print(f"queries={len(brief.queries)} sources={len(brief.sources)} "
          f"sections={len(brief.outline)}")
    return 0


async def cmd_article(args: argparse.Namespace) -> int:
    from byok_agent.agent.budgets import BudgetExceeded, KillSwitchEngaged
    from byok_agent.llm import AllTargetsFailed, LLMError
    from byok_agent.research import SearchNotConfigured, VerificationFailed

    max_usd = args.max_usd
    try:
        pipeline, db = await _make_research_pipeline(args, max_usd=max_usd)
    except _CLIError as exc:
        return _fail(str(exc))
    try:
        try:
            result = await pipeline.run_article(
                args.topic, depth=args.depth, policy=args.policy,
                primary_keyword=args.keyword or "", max_usd=max_usd,
            )
        except (SearchNotConfigured, VerificationFailed) as exc:
            return _fail(f"{type(exc).__name__}: {exc}")
        except (BudgetExceeded, KillSwitchEngaged, LLMError,
                AllTargetsFailed) as exc:
            return _fail(f"{type(exc).__name__}: {exc}")

        outdir = os.path.join("research_runs", _run_slug(args.topic))
        os.makedirs(outdir, exist_ok=True)
        article_path = os.path.join(outdir, "article.md")
        sources_path = os.path.join(outdir, "sources.md")
        verify_path = os.path.join(outdir, "verification_report.md")
        with open(article_path, "w", encoding="utf-8") as fh:
            fh.write(result.article_md)
        with open(sources_path, "w", encoding="utf-8") as fh:
            fh.write(result.brief_md + "\n\n" + result.sources_md)
        with open(verify_path, "w", encoding="utf-8") as fh:
            for rep in result.verification_reports:
                fh.write(f"## {rep.section_title}: "
                         f"{'PASS' if rep.passed else 'FAIL'} "
                         f"({rep.rounds} rounds)\n")
                for ch in rep.checks:
                    fh.write(f"- [{ch.verdict}] {ch.claim}\n")
                fh.write("\n")
    finally:
        await db.close()
    s = result.seo
    words = len(result.article_md.split())
    print(f"article saved: {article_path} ({words} words)")
    print(f"sources: {sources_path} | verification: {verify_path}")
    print(f"seo: title={s['title_len']}ch meta={s['meta_len']}ch "
          f"density={s['keyword_density_pct']}% first100={s['keyword_in_first_100']}")
    print(f"spend_est=${result.spend_usd_est:.4f} (estimate, not billing truth)")
    return 0


# -- Phase 4 subcommands: productization -------------------------------------
async def cmd_dashboard(args: argparse.Namespace) -> int:
    """Run the operator dashboard web UI."""
    import uvicorn

    from byok_agent.dashboard.app import Deps, create_app
    from byok_agent.llm import RoutingRuleStore

    if not os.environ.get("DASHBOARD_TOKEN", "").strip():
        return _fail(
            "DASHBOARD_TOKEN env var is required (operator bearer token)."
            " Example: DASHBOARD_TOKEN=$(openssl rand -hex 32) byok dashboard"
        )
    db, vault, ledger, _backend = await _open_vault_and_ledger()
    try:
        deps = Deps(
            db=db,
            vault=vault,
            ledger=ledger,
            rule_store=RoutingRuleStore(db),
        )
        app = create_app(deps)
        print(f"dashboard on http://{args.host}:{args.port} (token required)")
        print("WARNING: no TLS here — bind localhost or put it behind a reverse proxy.")
        await asyncio.to_thread(
            uvicorn.run, app, host=args.host, port=args.port, log_level="warning"
        )
        return 0
    finally:
        await db.close()


async def cmd_kill_switch(args: argparse.Namespace) -> int:
    """Engage/disengage the persistent kill switch."""
    from byok_agent.channels.audit import AuditLog
    from byok_agent.ops.killswitch import KillSwitch

    if args.action == "on" and not args.reason.strip():
        return _fail("--reason is required when engaging the kill switch")
    from byok_agent.db import apply_schema

    db_obj, backend = _open_db()
    await apply_schema(db_obj, backend)
    try:
        ks = KillSwitch(db_obj)
        audit = AuditLog(db_obj, _audit_hmac_key())
        if args.action == "on":
            await ks.engage(by=args.by, reason=args.reason)
            await audit.append(
                actor=f"cli:{args.by}", action="kill.engage",
                args={"reason": args.reason},
            )
            print("kill switch ENGAGED — all new agent work will refuse to start.")
        else:
            await ks.disengage(by=args.by)
            await audit.append(actor=f"cli:{args.by}", action="kill.disengage")
            print("kill switch disengaged.")
        return 0
    finally:
        await db_obj.close()


async def cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy a project dir. Tokens come from the vault, injected at the
    subprocess boundary only — never into the sandbox, never into files."""
    from pathlib import Path

    from byok_agent.deploy import DeployApprovalNeeded, autodetect, get_backend
    from byok_agent.llm import sanitize_secret_text
    from byok_agent.vault import KeyNotFoundError, RevokedKeyError

    project_dir = Path(args.dir).resolve()
    if not project_dir.is_dir():
        return _fail(f"project dir not found: {project_dir}")

    db, vault, _ledger, _backend = await _open_vault_and_ledger()
    try:
        backend = (
            autodetect(project_dir)
            if args.platform == "auto"
            else get_backend(args.platform)
        )
        if backend is None:
            return _fail(
                "no deploy backend detected for this project dir"
                " (need vercel.json/package.json, fly.toml, or railway.json)"
            )
        problems = backend.prechecks(project_dir)
        if problems:
            return _fail("prechecks failed:\n- " + "\n- ".join(problems))

        # Resolve deploy tokens from the vault ON THE HOST.
        secret_env: dict[str, str] = {}
        for spec in args.token:
            if "=" not in spec:
                return _fail(
                    f"bad --token {spec!r}: expected key_id=ENV_VAR"
                )
            key_id, env_var = spec.split("=", 1)
            try:
                secret_env[env_var] = await vault.get_key(
                    key_id.strip(), owner=args.user
                )
            except (KeyNotFoundError, RevokedKeyError) as exc:
                return _fail(str(exc))

        approved = args.auto_approve
        if args.auto_approve:
            print(
                "WARNING: --auto-approve skips the human approval gate."
                " Use only in CI you trust.",
                file=sys.stderr,
            )
        if not approved:
            try:
                ans = await asyncio.to_thread(
                    input,
                    f"Deploy {project_dir.name} to {backend.name}"
                    f"{' (PROD)' if args.prod else ''}? [y/N]: ",
                )
            except EOFError:
                ans = "n"
            approved = ans.strip().lower() in ("y", "yes")

        try:
            result = await backend.deploy(
                project_dir,
                env_extra=secret_env,
                prod=args.prod,
                approved=approved,
            )
        except DeployApprovalNeeded as exc:
            return _fail(str(exc))
        print(sanitize_secret_text(result.output[-2000:]))
        if result.url:
            print(f"deployed: {result.url}")
        return 0 if result.ok else 1
    finally:
        await db.close()


async def cmd_template_list(args: argparse.Namespace) -> int:
    from byok_agent.templates import list_templates

    for t in list_templates():
        print(f"{t['name']:15} {t['description']}")
    return 0


async def cmd_approval_list(args: argparse.Namespace) -> int:
    from byok_agent.ops.approvals import ApprovalStore

    db, backend = _open_db()
    from byok_agent.db import apply_schema

    await apply_schema(db, backend)
    try:
        pending = await ApprovalStore(db).list_pending()
        if not pending:
            print("no pending approvals")
            return 0
        for a in pending:
            print(f"#{a.id} [{a.kind}] {a.thread_id}")
            print(f"  {a.summary}")
            print(f"  created: {a.created_at}")
        return 0
    finally:
        await db.close()


async def cmd_approval_decide(args: argparse.Namespace) -> int:
    """Approve/deny a persisted approval (build-promote applies staged files)."""
    from byok_agent.agent.builder_graph import apply_approval
    from byok_agent.channels.audit import AuditLog

    if bool(args.approve) == bool(args.deny):
        return _fail("pass exactly one of --approve or --deny")
    db, backend = _open_db()
    from byok_agent.db import apply_schema

    await apply_schema(db, backend)
    try:
        result = await apply_approval(
            db, args.approval_id, approved=bool(args.approve), decided_by=args.by
        )
        audit = AuditLog(db, _audit_hmac_key())
        await audit.append(
            actor=f"cli:{args.by}",
            action="approval.decide",
            args={"approval_id": args.approval_id, "approved": bool(args.approve)},
        )
        if result.get("applied"):
            print(f"applied {len(result['applied'])} file(s):")
            for p in result["applied"]:
                print(f"  {p}")
        else:
            print("denied — nothing applied." if args.deny else "approved.")
        return 0
    except (KeyError, ValueError) as exc:
        return _fail(str(exc))
    finally:
        await db.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="byok", description="BYOK Agent Platform — Phase 3")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add-key", help="store a provider API key (encrypted at rest)")
    a.add_argument("--user", required=True, help="owner of this key (tenant scope)")
    a.add_argument("--provider", required=True, help="e.g. openai, anthropic, deepseek")
    a.add_argument("--label", required=True, help="human label, e.g. personal")
    a.add_argument("--allow-cross-provider-fallback", action="store_true",
                   help="consent: this key may serve cross-provider fallbacks")
    a.set_defaults(func=cmd_add_key)

    l = sub.add_parser("list-keys", help="list YOUR key envelopes (never secrets)")
    l.add_argument("--user", required=True, help="only keys owned by this user are shown")
    l.set_defaults(func=cmd_list_keys)

    r = sub.add_parser("revoke-key", help="revoke a key immediately (no draining)")
    r.add_argument("key_id")
    r.set_defaults(func=cmd_revoke_key)

    ro = sub.add_parser("rotate-key", help="replace a key's secret (bumps version)")
    ro.add_argument("key_id")
    ro.set_defaults(func=cmd_rotate_key)

    run = sub.add_parser("run", help="run one agent task")
    run.add_argument("--user", required=True, help="tenant scope: key must belong to this user")
    run.add_argument("--task", required=True)
    run.add_argument("--key-id", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("message", help="the task instruction")
    run.add_argument("--max-steps", type=int, default=25)
    run.add_argument("--max-tokens", type=int, default=100_000)
    run.add_argument("--max-usd", type=float, default=5.0)
    run.add_argument("--key-max-usd", type=float, default=None,
                     help="optional cumulative cap on the key's estimated spend")
    run.set_defaults(func=cmd_run)

    sbx = sub.add_parser("sandbox", help="sandbox utilities")
    sbx_sub = sbx.add_subparsers(dest="sandbox_command", required=True)
    se = sbx_sub.add_parser("exec", help="run one command in a throwaway sandbox")
    se.add_argument("--lang", default="python",
                    help="python | node | java | go | rust")
    se.add_argument("--timeout", type=float, default=120.0)
    se.add_argument("--install-network", default=None,
                    help="named Docker network for installs (default: no network)")
    se.add_argument("command", help="shell command to run in the sandbox")
    se.set_defaults(func=cmd_sandbox_exec)

    cm = sub.add_parser("codemap", help="print the repo map for a directory")
    cm.add_argument("dir", help="project directory")
    cm.add_argument("--changed", action="append", default=[],
                    help="repo-relative changed file (repeatable; ranked first)")
    cm.add_argument("--budget", type=int, default=1024,
                    help="token budget for the map")
    cm.set_defaults(func=cmd_codemap)

    bld = sub.add_parser(
        "build",
        help="planner -> coder -> tester -> reviewer against a project dir "
             "(promotes to the real dir only on approval)",
    )
    bld.add_argument("--user", required=True)
    bld.add_argument("--task", required=True)
    bld.add_argument("--dir", required=True, help="REAL project directory")
    bld.add_argument("--key-id", required=True)
    bld.add_argument("--model", default=None)
    bld.add_argument("--lang", default="python",
                     help="python | node | java | go | rust")
    bld.add_argument("--test-command", default=None,
                     help="override the test command for --lang")
    bld.add_argument("message", help="the build instruction")
    bld.add_argument("--max-steps", type=int, default=40)
    bld.add_argument("--max-tokens", type=int, default=200_000)
    bld.add_argument("--max-usd", type=float, default=10.0)
    bld.add_argument("--key-max-usd", type=float, default=None)
    bld.add_argument("--auto-approve", action="store_true",
                     help="skip the promotion approval gate (scripting/tests)")
    bld.add_argument("--template", default=None,
                     help="scaffold a starter template into --dir first"
                     " (see: byok template list)")
    bld.set_defaults(func=cmd_build)

    # -- Phase 2: WhatsApp control plane -----------------------------------
    wa = sub.add_parser("whatsapp", help="WhatsApp Cloud API control plane")
    wa_sub = wa.add_subparsers(dest="whatsapp_command", required=True)

    wsetup = wa_sub.add_parser(
        "setup", help="store WhatsApp Cloud API credentials (interactive)")
    wsetup.add_argument("--user", required=True, help="owner (tenant scope)")
    wsetup.add_argument("--agent-key-id", default=None,
                        help="vault key id driving the agent (optional now)")
    wsetup.add_argument("--agent-model", default="openai/gpt-4o-mini")
    wsetup.set_defaults(func=cmd_whatsapp_setup)

    wserve = wa_sub.add_parser("serve", help="run the webhook server")
    wserve.add_argument("--port", type=int, default=8000)
    wserve.add_argument("--host", default="127.0.0.1")
    wserve.set_defaults(func=cmd_whatsapp_serve)

    wpc = wa_sub.add_parser(
        "pair-code", help="print a single-use sender pairing code")
    wpc.add_argument("--user", required=True)
    wpc.set_defaults(func=cmd_whatsapp_pair_code)

    wsend = wa_sub.add_parser("send", help="manually send a WhatsApp message")
    wsend.add_argument("--user", required=True)
    wsend.add_argument("--to", required=True, help="recipient msisdn")
    wsend.add_argument("--text", required=True)
    wsend.set_defaults(func=cmd_whatsapp_send)

    wsenders = wa_sub.add_parser("senders", help="list sender pairing states")
    wsenders.add_argument("--user", required=True)
    wsenders.set_defaults(func=cmd_whatsapp_senders)

    waudit = wa_sub.add_parser("audit", help="verify the audit chain")
    waudit.add_argument("--user", required=True)
    waudit.set_defaults(func=cmd_whatsapp_audit)

    # -- Phase 3: research & articles --------------------------------------
    # NOTE: `research` takes the topic positionally; the literal topic
    # "providers" lists configured search keys instead (documented below).
    # A real subparser was tried and rejected: argparse cannot mix an
    # optional positional with subcommands without misrouting topics.
    rs = sub.add_parser("research", help="research a topic (plan+gather+outline); "
                                        "'providers' lists search keys")
    rs.add_argument("topic", help='topic to research, or "providers"')
    rs.add_argument("--user", required=True, help="tenant scope")
    rs.add_argument("--key-id", required=False, default=None,
                    help="LLM vault key id (not needed for 'providers')")
    rs.add_argument("--model", default=None)
    rs.add_argument("--keyword", default="",
                    help="primary keyword (defaults to topic)")
    rs.add_argument("--depth", default="standard",
                    choices=["quick", "standard", "deep"])
    rs.add_argument("--max-usd", type=float, default=5.0,
                    help="cap for the whole research run (estimate)")
    rs.add_argument("--key-max-usd", type=float, default=None)
    rs.set_defaults(func=cmd_research)

    art = sub.add_parser("article", help="full pipeline: research -> article")
    art.add_argument("topic", help="article topic")
    art.add_argument("--user", required=True, help="tenant scope")
    art.add_argument("--key-id", required=True, help="LLM vault key id")
    art.add_argument("--model", default=None)
    art.add_argument("--keyword", default="",
                     help="primary keyword (defaults to topic)")
    art.add_argument("--depth", default="standard",
                     choices=["quick", "standard", "deep"])
    art.add_argument("--policy", default="strict",
                     choices=["strict", "lenient"],
                     help="verification policy")
    art.add_argument("--max-usd", type=float, default=10.0,
                     help="cap for the whole article run (estimate)")
    art.add_argument("--key-max-usd", type=float, default=None)
    art.set_defaults(func=cmd_article)

    # -- Phase 4 subcommands: productization ---------------------------------
    dash = sub.add_parser("dashboard", help="run the operator dashboard (web UI)")
    dash.add_argument("--port", type=int, default=8080)
    dash.add_argument("--host", default="127.0.0.1")
    dash.set_defaults(func=cmd_dashboard)

    ks = sub.add_parser("kill-switch", help="engage/disengage the persistent kill switch")
    ks.add_argument("action", choices=["on", "off"])
    ks.add_argument("--reason", default="", help="reason (required when engaging)")
    ks.add_argument("--by", default="cli", help="operator name for the audit log")
    ks.set_defaults(func=cmd_kill_switch)

    dep = sub.add_parser("deploy", help="deploy a project dir to a hosting platform")
    dep.add_argument("--platform", default="auto",
                     choices=["auto", "vercel", "fly", "railway"])
    dep.add_argument("--dir", required=True, help="project directory")
    dep.add_argument("--user", required=True, help="tenant scope")
    dep.add_argument("--token", action="append", default=[],
                     help="vault key -> env var, e.g. --token key_vercel_x=VERCEL_TOKEN"
                     " (repeatable; secrets stay in the vault, injected at the"
                     " subprocess boundary only)")
    dep.add_argument("--prod", action="store_true",
                     help="production deploy (needs approval)")
    dep.add_argument("--auto-approve", action="store_true",
                     help="DANGER: skip the human approval gate (CI only)")
    dep.set_defaults(func=cmd_deploy)

    tmpl = sub.add_parser("template", help="app starter templates")
    tmpl_sub = tmpl.add_subparsers(dest="template_cmd", required=True)
    tmpl_list = tmpl_sub.add_parser("list", help="list starter templates")
    tmpl_list.set_defaults(func=cmd_template_list)

    appr = sub.add_parser("approval", help="decide persisted interrupt() approvals")
    appr_sub = appr.add_subparsers(dest="approval_cmd", required=True)
    appr_dec = appr_sub.add_parser("decide", help="approve/deny a pending approval")
    appr_dec.add_argument("approval_id", type=int)
    appr_dec.add_argument("--approve", action="store_true")
    appr_dec.add_argument("--deny", action="store_true")
    appr_dec.add_argument("--by", default="cli")
    appr_dec.set_defaults(func=cmd_approval_decide)
    appr_ls = appr_sub.add_parser("list", help="list pending approvals")
    appr_ls.set_defaults(func=cmd_approval_list)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
