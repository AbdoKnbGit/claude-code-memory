"""FastAPI route handlers — with MCP Memory middleware."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re as _re
import sqlite3 as _sqlite3
import time as _time
import traceback
import uuid
from collections import OrderedDict as _OD
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from providers.base import BaseProvider
from providers.common import get_user_facing_error_message
from providers.exceptions import InvalidRequestError, ProviderError
from providers.logging_utils import build_request_summary, log_request_compact

from .dependencies import get_provider_for_request, get_settings
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import MessagesResponse, TokenCountResponse, Usage
from .optimization_handlers import try_optimizations
from .request_utils import get_token_count

try:
    from api.hot_cache import hot_cache as _hot_cache
    from api.hot_cache import notifications as _notifs_api
except ImportError:
    _hot_cache = None
    _notifs_api = None

PING_BEFORE_EXPIRE = 3300
SESSION_TIMEOUT    = 3600
DEDUP_WINDOW       = float(os.getenv("DEDUP_WINDOW", "2.0"))
MEMORY_DB_PATH     = os.getenv("MEMORY_DB_PATH", "./data/memory.db")

_MAX_CACHED_SESSIONS = 500

_keepalive_tasks:      _OD[str, asyncio.Task] = _OD()
_session_last_msg:     _OD[str, float]        = _OD()
_session_last_user_msg: _OD[str, float]       = _OD()
_last_injection_hash:  _OD[str, str]          = _OD()
_last_injection_block: _OD[str, str]          = _OD()
_session_boot_done:    _OD[str, bool]         = _OD()
_session_boot_injected: _OD[str, float]       = _OD()

_session_seen_memory_ids: _OD[str, set]   = _OD()
_last_query_embedding:    _OD[str, list]  = _OD()
_last_injection_data:     _OD[str, dict]  = _OD()

DEDUP_MIN_OUT_TOKENS = int(os.getenv("DEDUP_MIN_OUT_TOKENS", "15"))
DEDUP_TTL_SECONDS    = float(os.getenv("DEDUP_TTL_SECONDS", "10.0"))

_dedup_cache: _OD[str, dict] = _OD()
_MAX_DEDUP_ENTRIES = 25

_inflight_requests: _OD[str, asyncio.Future] = _OD()
_MAX_INFLIGHT_ENTRIES = 200

INHERITED_CONTEXT_THRESHOLD = int(os.getenv("INHERITED_CONTEXT_THRESHOLD", "5750"))
_HISTORY_TOKEN_THRESHOLD = int(os.getenv("HISTORY_COMPRESS_THRESHOLD", "4600"))
_HISTORY_KEEP_RECENT = int(os.getenv("HISTORY_KEEP_RECENT", "8"))
_LARGE_OUTPUT_CHAR_CAP = int(os.getenv("LARGE_OUTPUT_CAP", "3000"))
_TOOL_RESULT_SUMMARIZE_THRESHOLD = int(os.getenv("TOOL_RESULT_SUMMARIZE_THRESHOLD", "1500"))
_TOOL_RESULT_SUMMARIZE_MAX_TOKENS = 150

logger.info(f"THRESHOLDS: HISTORY_COMPRESS={_HISTORY_TOKEN_THRESHOLD} INHERITED_CONTEXT={INHERITED_CONTEXT_THRESHOLD} TOOL_SUMMARIZE_MAX={_TOOL_RESULT_SUMMARIZE_MAX_TOKENS} OUTPUT_CAP={_LARGE_OUTPUT_CHAR_CAP}")

_session_compressed: _OD[str, bool] = _OD()

_project_warmed_sessions: _OD[str, set] = _OD()
_project_last_mem_hash:   _OD[str, str] = _OD()


_file_access: dict[str, dict[str, dict]] = {}
_FILE_ACCESS_MAX_PER_PROJECT = 200

_session_tokens: _OD[str, list] = _OD()
_session_turn_count: _OD[str, int] = _OD()
_session_cost: _OD[str, float] = _OD()
_session_model: _OD[str, str] = _OD()

_session_compact_count: _OD[str, int] = _OD()
_total_requests: int = 0

_session_last_cache_read: _OD[str, int] = _OD()
_session_cache_baseline: _OD[str, int] = _OD()
_session_compact_delay: _OD[str, int] = _OD()
_session_compact_reset_ctx: _OD[str, int] = _OD()
_session_compact_last_turn: _OD[str, int] = _OD()

_session_last_messages: _OD[str, list] = _OD()
_session_last_project: _OD[str, str] = _OD()
_session_last_output_tokens: _OD[str, int] = _OD()

_cerebras_fail_count: _OD[str, int] = _OD()
_cerebras_disabled: _OD[str, bool] = _OD()

SMART_COMPACT_THRESHOLD_PCT = float(os.getenv("SMART_COMPACT_THRESHOLD_PCT", "0.30"))
SMART_COMPACT_GROWTH_TRIGGER = int(os.getenv("SMART_COMPACT_GROWTH_TRIGGER", "2000"))
SMART_COMPACT_FLOOR_TOKENS = int(os.getenv("SMART_COMPACT_FLOOR_TOKENS", "15000"))
SMART_COMPACT_MIN_TURNS = int(os.getenv("SMART_COMPACT_MIN_TURNS", "8"))
SMART_COMPACT_MARKER = "[SMART_COMPACT_SUMMARY]"
COMPACT_MODEL = os.getenv("COMPACT_MODEL", "")
SESSION_END_MIN_TURNS = int(os.getenv("SESSION_END_MIN_TURNS", "4"))
SMART_COMPACT_MIN_REGROWTH_PCT = float(os.getenv("SMART_COMPACT_MIN_REGROWTH_PCT", "0.50"))
SMART_COMPACT_REGROWTH_ESCALATION = float(os.getenv("SMART_COMPACT_REGROWTH_ESCALATION", "0.10"))
SMART_COMPACT_REGROWTH_CAP = float(os.getenv("SMART_COMPACT_REGROWTH_CAP", "0.85"))
SMART_COMPACT_COOLDOWN_TURNS = int(os.getenv("SMART_COMPACT_COOLDOWN_TURNS", "4"))

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5": 200_000,
    "claude-opus-4-1": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 200_000

_INJECTION_BUDGET = int(os.getenv("INJECTION_BUDGET", "3000"))
_INJECTION_TTL_DAYS = int(os.getenv("INJECTION_TTL_DAYS", "14"))

_project_live_context: _OD[str, dict] = _OD()
_SUBAGENT_CTX_WINDOW_SEC = float(os.getenv("SUBAGENT_CTX_WINDOW_SEC", "1800"))
_SUBAGENT_CTX_MAX_INPUT = 3000
_SUBAGENT_CTX_MAX_OUTPUT = 500
_session_ctx_injected: _OD[str, bool] = _OD()
_project_last_token: _OD[str, str] = _OD()
_project_last_model: _OD[str, str] = _OD()
_main_event_loop: asyncio.AbstractEventLoop | None = None


def _init_project_meta_table():
    """Create _project_meta table and load cached values into memory."""
    try:
        conn = _sqlite3.connect(MEMORY_DB_PATH, timeout=2)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _project_meta "
            "(project_id TEXT PRIMARY KEY, last_model TEXT DEFAULT '', last_token TEXT DEFAULT '')"
        )
        rows = conn.execute("SELECT project_id, last_model, last_token FROM _project_meta").fetchall()
        conn.close()
        for pid, model, token in rows:
            if model:
                _project_last_model[pid] = model
            if token:
                _project_last_token[pid] = token
        if rows:
            logger.info(f"PROJECT_META: loaded {len(rows)} project(s) from SQLite")
    except Exception as e:
        logger.debug(f"PROJECT_META: init failed (non-fatal): {e}")


def _persist_project_meta(project_id: str, model: str = "", token: str = ""):
    """Write last model/token to SQLite so prewarm works after container restart."""
    try:
        conn = _sqlite3.connect(MEMORY_DB_PATH, timeout=2)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _project_meta "
            "(project_id TEXT PRIMARY KEY, last_model TEXT DEFAULT '', last_token TEXT DEFAULT '')"
        )
        existing = conn.execute(
            "SELECT last_model, last_token FROM _project_meta WHERE project_id=?",
            (project_id,),
        ).fetchone()
        final_model = model or (existing[0] if existing else "")
        final_token = token or (existing[1] if existing else "")
        conn.execute(
            "INSERT OR REPLACE INTO _project_meta VALUES (?,?,?)",
            (project_id, final_model, final_token),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


_init_project_meta_table()


def _od_set(d: _OD, key: str, val) -> None:
    """Set key in OrderedDict, evicting oldest entry when cap is exceeded."""
    d[key] = val
    if len(d) > _MAX_CACHED_SESSIONS:
        d.popitem(last=False)


def _od_set_dedup(d: _OD, key: str, val) -> None:
    """Set key in dedup cache, evicting oldest entry when tighter cap is exceeded."""
    d[key] = val
    if len(d) > _MAX_DEDUP_ENTRIES:
        d.popitem(last=False)


async def _prewarm_cache_for_session(
    session_id: str,
    project_id: str,
    injection_block: str,
    base_system: str,
    model: str,
    tools: list,
) -> None:
    """Fire a silent minimal API call to warm Anthropic's KV cache for a new session.

    ROOT CAUSE THIS FIXES:
      Sub-agents (Task tool) and new Claude Code windows open a fresh session
      with no cache history. Their first turn cold-writes the entire system
      prompt (15k–30k tokens at $1/MTok cache_write).  If the parent session
      already has the same prefix cached at Anthropic, a pre-warm call with
      an identical system prompt triggers a cache_read ($0.08/MTok) instead,
      saving ~10× on that first write.

    HOW:
      Called as a fire-and-forget asyncio task immediately after we detect
      a new session_id for a project whose injection_block we already know.
      Sends a single minimal message ("." ) with the full system prompt so
      Anthropic caches the prefix.  The response is discarded.

    COST:
      ~1 output token ($0.000004) + cache_read of existing prefix (already
      paid by parent session).  Net cost: effectively zero.
    """
    try:
        import httpx

        api_key  = os.getenv("ANTHROPIC_API_KEY", "")
        direct_url = "https://api.anthropic.com/v1/messages"

        if not api_key:
            api_key = _project_last_token.get(project_id, "")
        if not api_key:
            logger.debug(f"PREWARM: no token available for project={project_id} — skipping")
            return

        if base_system:
            system_blocks = [
                {"type": "text", "text": base_system, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": injection_block},
            ]
        else:
            system_blocks = [
                {"type": "text", "text": injection_block, "cache_control": {"type": "ephemeral"}},
            ]

        payload = {
            "model": model or _project_last_model.get(project_id, "") or "claude-haiku-4-5-20251001",
            "max_tokens": 1,
            "system": system_blocks,
            "messages": [{"role": "user", "content": "."}],
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ",".join([
                "prompt-caching-2024-07-31",
                "context-management-2025-06-27",
                "compact-2026-01-12",
                "token-efficient-tools-2025-02-19",
            ]),
            "content-type": "application/json",
        }

        _fallback_client = None
        try:
            from api.app import get_prewarm_client
            client = get_prewarm_client()
        except (ImportError, RuntimeError):
            import httpx as _httpx_fallback
            _fallback_client = _httpx_fallback.AsyncClient(timeout=10.0)
            client = _fallback_client

        try:
            resp = await client.post(direct_url, json=payload, headers=headers)
            _resp_json = resp.json() if resp.status_code == 200 else {}
            usage = _resp_json.get("usage", {})
            cw = usage.get("cache_creation_input_tokens", 0)
            cr = usage.get("cache_read_input_tokens", 0)
            _prewarm_model = _resp_json.get("model", payload["model"])
            logger.info(
                f"PREWARM: session={session_id} project={project_id} "
                f"model={_prewarm_model} "
                f"status={resp.status_code} cache_write={cw:,} cache_read={cr:,} "
                f"({'HIT' if cr > cw else 'MISS — wrote fresh'})"
            )
        except Exception as exc:
            logger.debug(f"PREWARM: failed for session={session_id}: {exc}")
        finally:
            if _fallback_client:
                await _fallback_client.aclose()
    except Exception as e:
        logger.debug(f"PREWARM: outer handler failed: {e}")


def _fire_prewarm_after_save(project_id: str) -> None:
    """Fire a prewarm for active sessions of a project after a memory save.

    Works with ANTHROPIC_API_KEY or stored OAuth token from passthrough mode.
    Falls back to hot cache update if no token is available.

    BUG 6 FIX (complete): After a memory save, the injection block WILL change
    because new entries exist. The old block in _last_injection_block is stale.
    We must:
      1. Build the NEW block from current (post-save) memory state
      2. Store it in hot cache under the new memory hash — so ALL subsequent
         requests hit the hot cache and get this exact block (bypassing
         session-specific query differences that cause block divergence)
      3. Update _last_injection_block for all active sessions — so keepalive
         pings warm the correct prefix
      4. Fire prewarm with the new block — so Anthropic caches the new prefix
         BEFORE the next real request arrives

    This guarantees ONE cold write (the prewarm) instead of multiple cold writes
    as sessions individually rebuild different blocks from different queries.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = _project_last_token.get(project_id, "")
    if not api_key:
        try:
            from memory import _search_with_tiers, _build_memory_block
            hits = _search_with_tiers("context", project_id, f"prewarm_after_save_{project_id}")
            if not hits:
                return
            block = _build_memory_block(hits, project_id=project_id, session_id="prewarm_save")
            if not block:
                return
            new_hash = _get_memory_hash(project_id)
            if _hot_cache and new_hash:
                _hot_cache.set(project_id, new_hash, block)
            _block_hash = hashlib.sha256(block.encode()).hexdigest()[:16]
            for sid, pid in list(_session_last_project.items()):
                if pid == project_id:
                    _od_set(_last_injection_block, sid, block)
                    _od_set(_last_injection_hash, sid, _block_hash)
            logger.debug(f"PREWARM_AFTER_SAVE: passthrough mode — hot cache + _last_injection_block updated for project={project_id}")
        except Exception as e:
            logger.debug(f"PREWARM_AFTER_SAVE: passthrough hot cache update failed: {e}")
        return

    try:
        from memory import _search_with_tiers, _build_memory_block

        hits = _search_with_tiers("context", project_id, f"prewarm_after_save_{project_id}")
        if not hits:
            return
        block = _build_memory_block(hits, project_id=project_id, session_id="prewarm_save")
        if not block:
            return

        new_hash = _get_memory_hash(project_id)
        if _hot_cache and new_hash:
            _hot_cache.set(project_id, new_hash, block)

        _block_hash = hashlib.sha256(block.encode()).hexdigest()[:16]
        sessions_to_warm = []
        for sid, pid in list(_session_last_project.items()):
            if pid == project_id:
                _od_set(_last_injection_block, sid, block)
                _od_set(_last_injection_hash, sid, _block_hash)
                sessions_to_warm.append(sid)

        logger.debug(
            f"PREWARM_AFTER_SAVE: hot cache + _last_injection_block updated "
            f"for {len(sessions_to_warm)} sessions in project={project_id}"
        )
    except Exception as e:
        logger.debug(f"PREWARM_AFTER_SAVE: failed for project={project_id}: {e}")


def _count_real_user_messages(messages: list) -> int:
    """Count user messages that are real text — not tool_result turns."""
    count = 0
    for m in messages:
        if hasattr(m, "model_dump"):
            md = m.model_dump()
            role = md.get("role", "")
            content = md.get("content", "")
        elif isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
        else:
            continue

        if role != "user":
            continue

        if isinstance(content, str) and content.strip():
            count += 1
        elif isinstance(content, list):
            if any(b.get("type") == "text" for b in content if isinstance(b, dict)):
                count += 1
    return count


_memory_hash_cache: dict[str, tuple[str, float]] = {}

def _invalidate_hash_cache(project_id: str) -> None:
    """Call after any memory save/delete to force next request to re-query."""
    _memory_hash_cache.pop(project_id, None)

def _get_memory_hash(project_id: str) -> str:
    """Stable content hash of the current injection-relevant memory state.

    Uses a 3-second TTL cache to avoid SQLite round-trip on every request.
    Hash is based on actual text + pinned status of active entries,
    sorted by id (stable order).
    """
    import time as _t
    cached = _memory_hash_cache.get(project_id)
    if cached and (_t.time() - cached[1]) < 3.0:
        return cached[0]

    try:
        conn = _sqlite3.connect(MEMORY_DB_PATH, timeout=2)
        rows = conn.execute(
            "SELECT id, text, pinned FROM memories "
            "WHERE project_id = ? AND deleted = 0 AND superseded = 0 "
            "ORDER BY id ASC",
            (project_id,),
        ).fetchall()
        conn.close()
        if not rows:
            h = hashlib.sha256(f"{project_id}:empty".encode()).hexdigest()[:16]
        else:
            fingerprint = "|".join(f"{r[0]}:{r[2]}:{r[1][:80]}" for r in rows)
            h = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        _memory_hash_cache[project_id] = (h, _t.time())
        return h
    except Exception:
        return ""


def _get_persisted_hash(project_id: str) -> str:
    """Read last-known memory hash from SQLite (survives restarts)."""
    try:
        conn = _sqlite3.connect(MEMORY_DB_PATH, timeout=2)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _injection_hashes "
            "(project_id TEXT PRIMARY KEY, hash TEXT)"
        )
        row = conn.execute(
            "SELECT hash FROM _injection_hashes WHERE project_id=?",
            (project_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception:
        return ""


def _set_persisted_hash(project_id: str, hash_val: str):
    """Write memory hash to SQLite so it survives container restarts."""
    try:
        conn = _sqlite3.connect(MEMORY_DB_PATH, timeout=2)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _injection_hashes "
            "(project_id TEXT PRIMARY KEY, hash TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO _injection_hashes VALUES (?,?)",
            (project_id, hash_val),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _db_acquire_session_lock(session_id: str) -> bool:
    """Exactly one cache write per session across all workers.

    Uses BEGIN IMMEDIATE to grab the SQLite write-lock *before* any reads,
    then INSERT OR IGNORE + rowcount to detect the true first inserter.
    No timing tricks, no race window — works under any concurrency.
    """
    import time
    now = time.time()
    try:
        conn = _sqlite3.connect(MEMORY_DB_PATH, timeout=5, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _session_locks (
                session_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                last_seen  REAL NOT NULL
            )
        """)

        conn.execute("BEGIN IMMEDIATE")

        conn.execute(
            "DELETE FROM _session_locks WHERE last_seen < ?",
            (now - SESSION_TIMEOUT,)
        )
        cursor = conn.execute(
            "INSERT OR IGNORE INTO _session_locks (session_id, created_at, last_seen) VALUES (?,?,?)",
            (session_id, now, now)
        )
        is_first = cursor.rowcount == 1

        if not is_first:
            conn.execute(
                "UPDATE _session_locks SET last_seen=? WHERE session_id=?",
                (now, session_id)
            )

        conn.commit()
        conn.close()

        logger.info(
            f"SESSION_LOCK: session={session_id} is_first={is_first}"
        )
        return is_first

    except Exception as e:
        logger.warning(f"SESSION_LOCK: db error {e} — allowing request through")
        return True


def _is_real_message(request_data) -> bool:
    """Real Claude Code requests always have a system prompt.
    Hook scripts (PreToolUse/PostToolUse/UserPromptSubmit) never send one."""
    if getattr(request_data, "system", None):
        return True
    if len(getattr(request_data, "messages", [])) >= 2:
        return True
    if getattr(request_data, "tools", None):
        return True
    return False


async def _keepalive_loop(session_id: str, provider, request_data):
    """Idle-aware ping: fires at 4m30s idle to refresh Anthropic's 5min cache TTL.

    Uses the LATEST injected block from _last_injection_block (not the stale
    request_data.system snapshot) so the ping warms the correct cache entry
    even if memory changed since session start.
    """
    import time
    import types

    from api.models.anthropic import Message as _Msg

    _model = getattr(request_data, "model", "claude-haiku-4-5-20251001")
    _tools = getattr(request_data, "tools", None) or []

    import re as _re_sys
    _raw_sys = getattr(request_data, "system", None)
    if isinstance(_raw_sys, list):
        _base_sys = "\n".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in _raw_sys
        )
    else:
        _base_sys = _raw_sys or ""
    _base_sys = _re_sys.sub(r"<memory_context[^>]*>.*?</memory_context_end>\n?", "", _base_sys, flags=_re_sys.DOTALL).strip()

    _self_task = None

    while True:
        await asyncio.sleep(1)

        if _self_task is None:
            _self_task = asyncio.current_task()
        if _keepalive_tasks.get(session_id) is not _self_task:
            logger.debug(f"KEEPALIVE: orphaned task detected session={session_id} — stopping")
            break

        now  = time.time()

        last = _session_last_msg.get(session_id, None)
        if last is None:
            logger.debug(f"KEEPALIVE: session={session_id} evicted from memory — stopping")
            await asyncio.get_running_loop().run_in_executor(
                None, _save_session_end, session_id,
            )
            _keepalive_tasks.pop(session_id, None)
            break

        idle = now - last

        last_user = _session_last_user_msg.get(session_id, last)
        user_idle = now - last_user
        if user_idle > SESSION_TIMEOUT:
            logger.info(f"KEEPALIVE: session={session_id} idle {user_idle/3600:.1f}h — stopped")
            await asyncio.get_running_loop().run_in_executor(
                None, _save_session_end, session_id,
            )
            _keepalive_tasks.pop(session_id, None)
            _session_last_msg.pop(session_id, None)
            _session_last_user_msg.pop(session_id, None)
            break

        if idle < PING_BEFORE_EXPIRE:
            continue

        try:
            latest_block = _last_injection_block.get(session_id, "")
            if latest_block and _base_sys:
                ping_system = [
                    {"type": "text", "text": _base_sys, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": latest_block},
                ]
            elif latest_block:
                ping_system = [{"type": "text", "text": latest_block, "cache_control": {"type": "ephemeral"}}]
            elif _base_sys:
                ping_system = [{"type": "text", "text": _base_sys, "cache_control": {"type": "ephemeral"}}]
            else:
                ping_system = None

            ping_req = types.SimpleNamespace(
                model=_model,
                max_tokens=1,
                messages=[_Msg(role="user", content=".")],
                system=ping_system,
                tools=_tools,
                temperature=None, top_p=None, metadata=None,
            )
            async for _ in provider.stream_response(ping_req, 0):
                pass
            _od_set(_session_last_msg, session_id, time.time())
            logger.info(f"KEEPALIVE: ping sent session={session_id} (idle={idle:.0f}s)")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"KEEPALIVE: ping failed session={session_id}: {e}")


def _increment_real_msg_count(session_id: str, project_id: str) -> None:
    """Increment the real user message counter for this session in SQLite.

    Only called from the _is_real_message() gate — so pings, hook calls,
    health checks, and tool-result-only turns never increment this counter.
    """
    try:
        conn = _sqlite3.connect(MEMORY_DB_PATH, timeout=2, isolation_level=None)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _session_msg_counts (
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                msg_count  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (session_id, project_id)
            )
        """)
        conn.execute("""
            INSERT INTO _session_msg_counts (session_id, project_id, msg_count)
            VALUES (?, ?, 1)
            ON CONFLICT(session_id, project_id) DO UPDATE SET msg_count = msg_count + 1
        """, (session_id, project_id))
        conn.close()
    except Exception as e:
        logger.debug(f"MSG_COUNT: failed to increment: {e}")


def _touch_keepalive(session_id: str, provider, request_data):
    """Record real-message activity. Start loop once per session."""
    import time
    _od_set(_session_last_msg, session_id, time.time())
    _od_set(_session_last_user_msg, session_id, time.time())

    existing = _keepalive_tasks.get(session_id)
    if existing and not existing.done():
        return

    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_keepalive_loop(session_id, provider, request_data))
        _od_set(_keepalive_tasks, session_id, task)
        logger.info(f"KEEPALIVE: started session={session_id}")
    except Exception as e:
        logger.warning(f"KEEPALIVE: could not start: {e}")


def _stop_keepalive(session_id: str):
    """Cancel keepalive immediately (client disconnected)."""
    _session_last_msg.pop(session_id, None)
    _session_last_user_msg.pop(session_id, None)
    _last_injection_hash.pop(session_id, None)
    _last_injection_block.pop(session_id, None)
    _session_boot_done.pop(session_id, None)
    _session_boot_injected.pop(session_id, None)
    _session_seen_memory_ids.pop(session_id, None)
    _last_query_embedding.pop(session_id, None)
    _last_injection_data.pop(session_id, None)
    _dedup_cache.pop(session_id, None)
    _session_compressed.pop(session_id, None)
    _session_tokens.pop(session_id, None)
    _session_turn_count.pop(session_id, None)
    _session_cost.pop(session_id, None)
    _session_model.pop(session_id, None)
    _session_compact_count.pop(session_id, None)
    _session_last_cache_read.pop(session_id, None)
    _session_cache_baseline.pop(session_id, None)
    _session_compact_delay.pop(session_id, None)
    _session_compact_reset_ctx.pop(session_id, None)
    _session_compact_last_turn.pop(session_id, None)
    _session_last_messages.pop(session_id, None)
    _session_last_output_tokens.pop(session_id, None)
    _session_last_project.pop(session_id, None)
    _cerebras_fail_count.pop(session_id, None)
    _cerebras_disabled.pop(session_id, None)
    _session_ctx_injected.pop(session_id, None)
    _tool_result_hashes.pop(session_id, None)

    old = _keepalive_tasks.pop(session_id, None)
    if old and not old.done():
        old.cancel()


def _get_context_window(model: str) -> int:
    """Return context window size for the given model."""
    model_lower = (model or "").lower()
    for key, window in _MODEL_CONTEXT_WINDOWS.items():
        if key in model_lower:
            return window
    return _DEFAULT_CONTEXT_WINDOW


def _compute_session_value_score(session_id: str) -> float:
    """Compute a 0-1 score representing how much this session is worth optimizing.

    Combines cost-per-turn, growth rate, and session age. Cheap short sessions
    score low. Expensive long sessions score high. Every optimization decision
    uses this as a proportionality gate.
    """
    turn_count = _session_turn_count.get(session_id, 0)
    if turn_count < 2:
        return 0.0

    cost = _session_cost.get(session_id, 0.0)
    token_history = _session_tokens.get(session_id, [])

    cost_per_turn = cost / max(turn_count, 1)
    cost_signal = min(1.0, cost_per_turn / 0.05)

    recent = token_history[-10:] if len(token_history) >= 2 else token_history
    if len(recent) >= 2:
        growth = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        growth_signal = min(1.0, max(0.0, growth / 3000))
    else:
        growth_signal = 0.0

    age_signal = min(1.0, max(0.0, (turn_count - 2) / 28))

    score = 0.4 * cost_signal + 0.35 * growth_signal + 0.25 * age_signal
    return round(score, 3)


def _update_file_access(project_id: str, messages: list, turn_count: int) -> None:
    """Parse tool_use blocks in messages to track file access patterns.

    Updates the module-level _file_access tracker. Zero cost — pure Python
    dict operations, no API calls.
    """
    if project_id not in _file_access:
        if len(_file_access) > _MAX_CACHED_SESSIONS:
            _file_access.pop(next(iter(_file_access)), None)
        _file_access[project_id] = {}
    tracker = _file_access[project_id]

    if len(tracker) > _FILE_ACCESS_MAX_PER_PROJECT:
        sorted_files = sorted(tracker.items(), key=lambda x: x[1]["last_turn"])
        for fp, _ in sorted_files[:len(tracker) - _FILE_ACCESS_MAX_PER_PROJECT]:
            del tracker[fp]

    for msg in messages:
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
        elif isinstance(msg, dict):
            md = msg
        else:
            continue

        role = md.get("role", "")
        content = md.get("content", "")
        if role != "assistant" or not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            name = block.get("name", "")
            inp = block.get("input", {})
            if not isinstance(inp, dict):
                continue

            file_path = (
                inp.get("file_path")
                or inp.get("path")
                or inp.get("filename")
                or ""
            )
            if not file_path or len(file_path) < 3:
                continue

            if file_path not in tracker:
                tracker[file_path] = {"reads": 0, "edits": 0, "last_turn": 0, "tokens": 0}

            entry = tracker[file_path]
            entry["last_turn"] = turn_count

            if name in ("Read", "read_file", "View", "cat"):
                entry["reads"] += 1
                content_text = inp.get("content", "")
                if content_text:
                    entry["tokens"] += len(str(content_text)) // 3
            elif name in ("Edit", "Write", "write_file", "edit_file", "Insert", "Replace"):
                entry["edits"] += 1


def _get_compression_strategy(
    project_id: str, file_path: str, current_turn: int
) -> str:
    """Return compression strategy for a file: 'skip', 'light', or 'aggressive'.

    Skip: file read frequently in recent turns — compressing causes re-read churn.
    Light: file has been edited — actionable content deserves reasonable preservation.
    Aggressive: file read but never edited, not referenced recently — dead weight.
    """
    tracker = _file_access.get(project_id, {})
    if not file_path or file_path not in tracker:
        return "light"

    entry = tracker[file_path]
    turns_since_access = current_turn - entry.get("last_turn", 0)

    if entry["reads"] >= 2 and turns_since_access <= 3:
        return "skip"

    if entry["edits"] > 0:
        return "light"

    if entry["reads"] > 0 and entry["edits"] == 0 and turns_since_access > 5:
        return "aggressive"

    return "light"


def _l1_call_llm_safe(session_id: str, **kwargs):
    """Layer 1 graceful degradation wrapper around _memory_call_llm.

    Level 1: retry once after 200ms on failure.
    Level 2: pass through uncompressed on retry failure.
    Level 3: disable Layer 1 after 5 consecutive failures.
    """
    if not _memory_call_llm:
        return None

    _task = kwargs.get('task', 'unknown')

    if _cerebras_disabled.get(session_id):
        logger.info(
            f"LLM_TRACE: L1_DISABLED session={session_id} task={_task} — "
            f"skipping Cerebras (disabled after 5 failures). Passing through uncompressed."
        )
        return None

    fail_count = _cerebras_fail_count.get(session_id, 0)

    try:
        result = _memory_call_llm(**kwargs)
        if result:
            if fail_count > 0:
                _od_set(_cerebras_fail_count, session_id, 0)
                logger.info(
                    f"LLM_TRACE: L1_RECOVERED session={session_id} task={_task} — "
                    f"Cerebras back online after {fail_count} failures"
                )
            return result
    except Exception:
        pass

    import time as _retry_time
    _retry_time.sleep(0.2)
    try:
        result = _memory_call_llm(**kwargs)
        if result:
            _od_set(_cerebras_fail_count, session_id, 0)
            return result
    except Exception:
        pass

    fail_count += 1
    _od_set(_cerebras_fail_count, session_id, fail_count)

    if fail_count >= 5:
        _od_set(_cerebras_disabled, session_id, True)
        logger.warning(
            f"LLM_TRACE: L1_DISABLED_NOW session={session_id} task={_task} — "
            f"Cerebras failed {fail_count} consecutive calls. Layer 1 disabled for this session. "
            f" ALL COMPRESSION BYPASSED — cost will increase!"
        )
    else:
        logger.warning(
            f"LLM_TRACE: L1_FAIL session={session_id} task={_task} — "
            f"Cerebras call failed ({fail_count}/5). Passing through uncompressed."
        )

    return None


def _content_density_check(messages: list) -> bool:
    """Check if messages contain enough real decisions to justify compaction.

    Counts unique file paths, function names, decision markers, and error types.
    Returns True if density is sufficient for a meaningful summary.
    """
    import re as _density_re

    text_parts = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
        elif isinstance(msg, dict):
            md = msg
        else:
            continue
        content = md.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        text_parts.append(inner[:300])

    full_text = " ".join(text_parts)
    if not full_text:
        return False

    file_paths = set(_density_re.findall(r'[/\\][\w./\\-]+\.\w{1,6}', full_text))
    func_names = set(_density_re.findall(r'(?:def |function |fn |class )\w+', full_text))
    decisions = len(_density_re.findall(
        r'(?:decided|chose|using|switched to|changed to|fixed|resolved|implemented|created|added)',
        full_text, _density_re.IGNORECASE
    ))
    errors = set(_density_re.findall(r'\w+Error|\w+Exception|FAIL|FAILED|bug|issue', full_text))

    density_score = len(file_paths) + len(func_names) + decisions + len(errors)
    return density_score >= 4


def _select_compact_model(context_tokens: int, session_value: float) -> str:
    """Select the cheapest Anthropic model sufficient for the context size.

    Small context + low value → Haiku.
    Medium context → Sonnet.
    Large context + high value → best available.
    COMPACT_MODEL env var always overrides.
    """
    if COMPACT_MODEL:
        return COMPACT_MODEL

    if context_tokens < 20_000 or session_value < 0.3:
        return "claude-haiku-4-5-20251001"
    elif context_tokens < 80_000:
        return "claude-sonnet-4-20250514"
    else:
        return "claude-opus-4-5-20250514"


async def _smart_compact(
    session_id: str,
    project_id: str,
    messages: list,
    model: str,
) -> list | None:
    """Layer 2 smart compact — no hard cap; escalating regrowth gate controls frequency.

    Each subsequent fire requires progressively more regrowth and a longer
    turn cooldown, making re-fires increasingly rare but never impossible
    when the context genuinely needs it.

    Returns compacted messages list on success, None on skip/failure.
    """
    fire_count = _session_compact_count.get(session_id, 0)
    turn_count = _session_turn_count.get(session_id, 0)

    if turn_count < SMART_COMPACT_MIN_TURNS:
        return None

    if fire_count > 0:
        last_fire_turn = _session_compact_last_turn.get(session_id, 0)
        cooldown = SMART_COMPACT_MIN_TURNS + fire_count * SMART_COMPACT_COOLDOWN_TURNS
        if turn_count - last_fire_turn < cooldown:
            return None

    delay_until = _session_compact_delay.get(session_id, 0)
    if delay_until > 0 and turn_count < delay_until:
        return None

    total_tokens = sum(_estimate_msg_tokens(m) for m in messages)
    context_window = _get_context_window(model)
    usage_pct = total_tokens / context_window

    pct_triggered = usage_pct >= SMART_COMPACT_THRESHOLD_PCT

    token_history = _session_tokens.get(session_id, [])
    growth_triggered = False
    if len(token_history) >= 2 and total_tokens >= SMART_COMPACT_FLOOR_TOKENS:
        recent = token_history[-10:] if len(token_history) >= 10 else token_history
        growth_per_turn = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        growth_triggered = growth_per_turn >= SMART_COMPACT_GROWTH_TRIGGER

    _COST_CEILING = int(os.getenv("SMART_COMPACT_COST_CEILING", "60000"))
    cost_triggered = False
    if total_tokens >= _COST_CEILING:
        _cr_current = _session_last_cache_read.get(session_id, 0)
        _cr_baseline = _session_cache_baseline.get(session_id, 0)
        _cr_growth = _cr_current - _cr_baseline if _cr_current > _cr_baseline else 0
        _effective_size = max(total_tokens, _cr_growth)
        cost_triggered = _effective_size >= _COST_CEILING

    if not pct_triggered and not growth_triggered and not cost_triggered:
        return None

    if fire_count > 0:
        reset_ctx = _session_compact_reset_ctx.get(session_id, 0)
        threshold_tokens = context_window * SMART_COMPACT_THRESHOLD_PCT
        _REGROWTH_CAP_TOKENS = _COST_CEILING * 2
        threshold_tokens = min(threshold_tokens, _REGROWTH_CAP_TOKENS)
        escalated_pct = min(
            SMART_COMPACT_MIN_REGROWTH_PCT + SMART_COMPACT_REGROWTH_ESCALATION * (fire_count - 1),
            SMART_COMPACT_REGROWTH_CAP,
        )
        min_regrowth = reset_ctx + (threshold_tokens - reset_ctx) * escalated_pct
        if total_tokens < min_regrowth:
            return None

    if not _content_density_check(messages):
        _od_set(_session_compact_delay, session_id, turn_count + 10)
        logger.info(
            f"SMART_COMPACT: session={session_id} density too low at turn {turn_count} — "
            f"delaying 10 turns"
        )
        return None

    _prev_last_turn = _session_compact_last_turn.get(session_id, 0)
    _od_set(_session_compact_count, session_id, fire_count + 1)
    _od_set(_session_compact_last_turn, session_id, turn_count)

    logger.info(
        f"SMART_COMPACT: firing #{fire_count + 1} "
        f"session={session_id} turn={turn_count} "
        f"tokens={total_tokens} usage={usage_pct:.1%} "
        f"trigger={'pct' if pct_triggered else ('cost' if cost_triggered else 'growth')}"
    )

    keep_count = min(_HISTORY_KEEP_RECENT, len(messages) - 1)
    target_cutoff = max(1, len(messages) - keep_count)
    cutoff = 0
    for i in range(target_cutoff, 0, -1):
        msg = messages[i]
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
            role = md.get("role", "")
            content = md.get("content", "")
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            continue
        if role == "user":
            is_tool_result = False
            if isinstance(content, list):
                for block in content:
                    b = block if isinstance(block, dict) else (
                        block.model_dump() if hasattr(block, "model_dump") else {}
                    )
                    if b.get("type") == "tool_result":
                        is_tool_result = True
                        break
            if not is_tool_result:
                cutoff = i
                break

    if cutoff <= 0:
        for i in range(target_cutoff, 0, -1):
            msg = messages[i]
            if hasattr(msg, "model_dump"):
                role = msg.model_dump().get("role", "")
            elif isinstance(msg, dict):
                role = msg.get("role", "")
            else:
                continue
            if role == "user":
                cutoff = i
                break

    if cutoff <= 0 and target_cutoff > 0:
        cutoff = target_cutoff

    if cutoff <= 0:
        _od_set(_session_compact_count, session_id, fire_count)
        _od_set(_session_compact_last_turn, session_id, _prev_last_turn)
        return None

    to_compact = messages[:cutoff]
    to_keep = messages[cutoff:]

    turns = []
    for m in to_compact:
        if hasattr(m, "model_dump"):
            md = m.model_dump()
            role = md.get("role", "?")
            content = md.get("content", "")
        elif isinstance(m, dict):
            role = m.get("role", "?")
            content = m.get("content", "")
        else:
            continue
        if isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("text"):
                    parts.append(b["text"])
                inner = b.get("content", "")
                if isinstance(inner, str) and inner:
                    parts.append(inner[:300])
            content = " ".join(parts)
        if content and len(str(content)) > 10:
            turns.append(f"{role.upper()}: {str(content)[:600]}")

    if not turns:
        _od_set(_session_compact_count, session_id, fire_count)
        _od_set(_session_compact_last_turn, session_id, _prev_last_turn)
        return None

    turns_text = "\n---\n".join(turns)

    session_value = _compute_session_value_score(session_id)
    compact_tokens = sum(_estimate_msg_tokens(m) for m in to_compact)
    compact_model = _select_compact_model(compact_tokens, session_value)

    try:
        import httpx

        compact_system = (
            "You are a technical session summarizer. Produce a dense, precise summary "
            "of the conversation. PRESERVE: all file paths with line numbers, function "
            "and class names, bug types and error messages, decisions made and WHY, "
            "current task state (what is done, what comes next). STRIP: greetings, "
            "filler, repeated information, tool output noise. Output only the summary."
        )

        api_key = os.getenv("ANTHROPIC_API_KEY", "")

        if not api_key:
            if not _memory_call_llm:
                _od_set(_session_compact_count, session_id, fire_count)
                _od_set(_session_compact_last_turn, session_id, _prev_last_turn)
                return None

            _loop = asyncio.get_running_loop()
            summary_text = await _loop.run_in_executor(
                None, _memory_call_llm,
                compact_system,
                f"Summarize this session:\n{turns_text[:6000]}",
                500, 8.0, "smart_compact",
            )
            _responded_model = os.getenv("MEMORY_MODEL", "cerebras/unknown")

            if not summary_text:
                logger.warning(f"SMART_COMPACT: Cerebras fallback returned empty for session={session_id}")
                _od_set(_session_compact_count, session_id, fire_count)
                _od_set(_session_compact_last_turn, session_id, _prev_last_turn)
                return None

            logger.info(
                f"SMART_COMPACT: success (cerebras fallback) session={session_id} "
                f"model={_responded_model} "
                f"compacted={compact_tokens} tokens → ~{len(summary_text)//3} tokens"
            )
        else:
            payload = {
                "model": compact_model,
                "max_tokens": 1024,
                "system": compact_system,
                "messages": [{"role": "user", "content": f"Summarize this session:\n{turns_text[:12000]}"}],
            }

            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

            _fallback_client = None
            try:
                from api.app import get_prewarm_client
                client = get_prewarm_client()
            except (ImportError, RuntimeError):
                _fallback_client = httpx.AsyncClient(timeout=30.0)
                client = _fallback_client

            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers,
            )

            if _fallback_client:
                await _fallback_client.aclose()

            if resp.status_code != 200:
                logger.warning(f"SMART_COMPACT: API call failed status={resp.status_code}")
                _od_set(_session_compact_count, session_id, fire_count)
                _od_set(_session_compact_last_turn, session_id, _prev_last_turn)
                return None

            resp_data = resp.json()
            _responded_model = resp_data.get("model", compact_model)
            summary_text = ""
            for block in resp_data.get("content", []):
                if block.get("type") == "text":
                    summary_text += block.get("text", "")

            if not summary_text:
                _od_set(_session_compact_count, session_id, fire_count)
                _od_set(_session_compact_last_turn, session_id, _prev_last_turn)
                return None

            usage = resp_data.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            _cache_r = usage.get("cache_read_input_tokens", 0)
            _cache_w = usage.get("cache_creation_input_tokens", 0)
            logger.info(
                f"SMART_COMPACT: success session={session_id} "
                f"requested={compact_model} responded={_responded_model} "
                f"input={in_tok:,} output={out_tok:,} "
                f"cache_read={_cache_r:,} cache_write={_cache_w:,} "
            f"compacted={compact_tokens} tokens → ~{len(summary_text)//3} tokens"
        )

        try:
            from memory import _save
            _save(
                project_id, session_id,
                f"[Smart compact turns 1-{cutoff}] {summary_text}",
                source="smart_compact",
                check_dedup=False,
                is_manual=False,
                trigger=f"turn_{turn_count}",
                component="layer2",
            )
            if _hot_cache:
                _hot_cache.invalidate(project_id)
            _invalidate_hash_cache(project_id)
            await asyncio.get_running_loop().run_in_executor(None, _fire_prewarm_after_save, project_id)
        except Exception as e:
            logger.warning(f"SMART_COMPACT: save to memory failed: {e}")

        try:
            from api.models.anthropic import Message as _Msg
            summary_msg = _Msg(
                role="user",
                content=f"{SMART_COMPACT_MARKER} {summary_text}",
            )
        except Exception:
            import types
            summary_msg = types.SimpleNamespace(
                role="user",
                content=f"{SMART_COMPACT_MARKER} {summary_text}",
            )

        reset_tokens = sum(_estimate_msg_tokens(m) for m in [summary_msg, *to_keep])
        _od_set(_session_compact_reset_ctx, session_id, reset_tokens)

        logger.info(
            f"SMART_COMPACT: complete #{fire_count + 1} session={session_id} "
            f"context_before={total_tokens:,} context_after={reset_tokens:,} "
            f"model={compact_model}"
        )

        return [summary_msg, *to_keep]

    except Exception as e:
        logger.warning(f"SMART_COMPACT: failed session={session_id}: {e}")
        _od_set(_session_compact_count, session_id, fire_count)
        _od_set(_session_compact_last_turn, session_id, _prev_last_turn)
        return None


def _save_session_end(session_id: str) -> None:
    """Save session tail to Layer 3 when keepalive detects session death.

    Uses Cerebras (zero cost). If no smart compact fired, uses tiered chunking:
    split into 3000-token chunks, summarize each, then merge.
    """
    messages = _session_last_messages.pop(session_id, None)
    project_id = _session_last_project.pop(session_id, None)

    if not messages or not project_id:
        return

    if not _memory_call_llm:
        _fire_prewarm_after_save(project_id)
        return

    post_compact_start = 0
    for i, msg in enumerate(messages):
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
        elif isinstance(msg, dict):
            md = msg
        else:
            continue
        content = md.get("content", "")
        if isinstance(content, str) and SMART_COMPACT_MARKER in content:
            post_compact_start = i + 1

    save_messages = messages[post_compact_start:]

    assistant_turns = 0
    content_parts = []
    for msg in save_messages:
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
        elif isinstance(msg, dict):
            md = msg
        else:
            continue
        if md.get("role") == "assistant":
            assistant_turns += 1
            content = md.get("content", "")
            if isinstance(content, str):
                content_parts.append(content[:600])
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content_parts.append(block.get("text", "")[:600])

    if assistant_turns < SESSION_END_MIN_TURNS:
        _fire_prewarm_after_save(project_id)
        return

    save_text = "\n---\n".join(content_parts)
    if not save_text.strip():
        _fire_prewarm_after_save(project_id)
        return

    if post_compact_start == 0 and len(save_text) > 3000:
        chunks = [save_text[i:i+3000] for i in range(0, len(save_text), 3000)]
        chunk_summaries = []
        for chunk in chunks[:5]:
            summary = _memory_call_llm(
                system="Summarize this session fragment. Keep: file paths, decisions, errors, task state. Max 60 words.",
                user=chunk,
                max_tokens=100,
                timeout=4.0,
                task="session_end_chunk",
            )
            if summary:
                chunk_summaries.append(summary.strip())

        if chunk_summaries:
            merged_input = "\n".join(chunk_summaries)
            save_text = _memory_call_llm(
                system="Merge these summaries into one coherent session summary. Keep all technical details. Max 80 words.",
                user=merged_input,
                max_tokens=120,
                timeout=4.0,
                task="session_end_merge",
            ) or merged_input[:500]
    else:
        save_text = _memory_call_llm(
            system="Summarize this session tail. Keep: file paths, decisions, errors, current state. Max 80 words.",
            user=save_text[:3000],
            max_tokens=120,
            timeout=4.0,
            task="session_end_save",
        ) or save_text[:300]

    if not save_text or not save_text.strip():
        _fire_prewarm_after_save(project_id)
        return

    try:
        from memory import _save
        turn_count = _session_turn_count.get(session_id, 0)
        _save(
            project_id, session_id,
            f"[Session end] {save_text.strip()}",
            source="session_end",
            check_dedup=True,
            is_manual=False,
            trigger=f"session_death_turn_{turn_count}",
            component="layer2",
        )
        if _hot_cache:
            _hot_cache.invalidate(project_id)
        _invalidate_hash_cache(project_id)
        logger.info(
            f"SESSION_END_SAVE: session={session_id} project={project_id} "
            f"turns={assistant_turns} saved ~{len(save_text)//3} tokens"
        )
    except Exception as e:
        logger.warning(f"SESSION_END_SAVE: failed session={session_id}: {e}")
    finally:
        _fire_prewarm_after_save(project_id)


def _update_live_context_async(session_id: str, project_id: str, messages: list) -> None:
    """Fire-and-forget: compress last 10 assistant turns and update _project_live_context.

    Uses the existing free Cerebras path. Never blocks the main request.
    """
    if not _memory_call_llm or not project_id:
        if project_id:
            _fire_prewarm_after_save(project_id)
        return

    assistant_parts = []
    for msg in reversed(messages):
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
        elif isinstance(msg, dict):
            md = msg
        else:
            continue
        if md.get("role") != "assistant":
            continue
        content = md.get("content", "")
        if isinstance(content, str) and content.strip():
            assistant_parts.append(content[:400])
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        assistant_parts.append(text[:400])
                        break
        if len(assistant_parts) >= 10:
            break

    if not assistant_parts:
        return

    assistant_parts.reverse()
    raw_text = "\n---\n".join(assistant_parts)
    raw_text = raw_text[:12000]

    summary = _memory_call_llm(
        system=(
            "Summarize this session state for a sub-agent that will continue the work. "
            "PRESERVE: current task, files being worked on, decisions made, what is done, "
            "what comes next. Max 120 words. Dense, no filler."
        ),
        user=raw_text,
        max_tokens=_SUBAGENT_CTX_MAX_OUTPUT,
        timeout=6.0,
        task="subagent_ctx",
    )

    if not summary or not summary.strip():
        return

    file_entries = _file_access.get(project_id, {})
    active_files = sorted(
        file_entries.keys(),
        key=lambda f: file_entries[f].get("last_turn", 0),
        reverse=True,
    )[:15]

    _od_set(_project_live_context, project_id, {
        "summary": summary.strip(),
        "active_files": active_files,
        "session_id": session_id,
        "timestamp": _time.time(),
    })
    logger.debug(
        f"SUBAGENT_CTX: updated project={project_id} session={session_id} "
        f"files={len(active_files)} summary_len={len(summary)}"
    )


def _get_parent_context_block(project_id: str, session_id: str) -> str | None:
    """Return a formatted context block if a live parent context exists for this project.

    Returns None if no context, stale (>30min), or same session.
    """
    ctx = _project_live_context.get(project_id)
    if not ctx:
        return None

    if ctx["session_id"] == session_id:
        return None

    age = _time.time() - ctx["timestamp"]
    if age > _SUBAGENT_CTX_WINDOW_SEC:
        return None

    files_str = ", ".join(ctx["active_files"][:10]) if ctx["active_files"] else "none tracked"
    block = (
        f"\n<live_parent_session_context>\n"
        f"Parent session {ctx['session_id'][:12]}... was working on this project "
        f"{int(age)}s ago.\n\n"
        f"Active files: {files_str}\n\n"
        f"Session state:\n{ctx['summary']}\n"
        f"</live_parent_session_context>\n"
    )
    return block


def _is_topic_switch(session_id: str, current_query: str, threshold: float = 0.40) -> bool:
    """
    Returns True if current query is semantically far from the last query.
    Uses cosine similarity on embeddings from memory._encode().
    Falls back to False (no rebuild) on any error — never blocks the request.
    """
    try:
        from memory import _encode
        curr_emb = _encode(current_query)
        prev_emb = _last_query_embedding.get(session_id)
        _od_set(_last_query_embedding, session_id, curr_emb)

        if prev_emb is None:
            return True

        dot = sum(a * b for a, b in zip(curr_emb, prev_emb))
        mag_a = sum(x * x for x in curr_emb) ** 0.5
        mag_b = sum(x * x for x in prev_emb) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return True
        cos = dot / (mag_a * mag_b)
        return cos < threshold

    except Exception as e:
        logger.debug(f"[SmartInject] Topic switch check failed: {e}")
        return False


try:
    from memory import (
        COMMAND_RE as _MEMORY_COMMAND_RE,
    )
    from memory import _call_llm as _memory_call_llm
    from memory import (
        _extract_last_user_text,
        _inject_system,
        process_memory,
    )

    MEMORY_ENABLED = True
    logger.info("MCP Memory middleware loaded")
except ImportError as e:
    MEMORY_ENABLED = False
    _memory_call_llm = None
    logger.warning(f"Memory middleware not available: {e}")


_COMPRESS_SYSTEM = (
    "Summarize this conversation preserving ALL of the following without exception: "
    "variable names, function names, class names, file paths, line numbers, "
    "error types, error messages, error codes, import statements, stack trace structure, "
    "test names, and any decision or conclusion reached. "
    "Summarize prose context aggressively. Never paraphrase code, errors, or file references. "
    "If choosing between a technical detail and the word limit, preserve the detail."
)

_TOOL_SUMMARIZE_SYSTEM = (
    "Compress this tool output to 2-3 dense lines. "
    "PRESERVE VERBATIM: all error types (TypeError, KeyError, etc.), "
    "all error messages exactly as written, all file paths with line numbers, "
    "all function/class/variable names, all test names and failure reasons. "
    "REMOVE ONLY: progress bars, decorative separators, repeated identical lines, "
    "blank lines, and ASCII art. "
    "Output only the compressed result, no preamble."
)

_tool_result_hashes: _OD[str, dict] = _OD()


def _summarize_tool_result(content: str, tool_name: str = "") -> str:
    """Compress a tool_result to 1-2 lines using MEMORY_MODEL.

    Called by _apply_tool_result_summarization when a result exceeds
    _TOOL_RESULT_SUMMARIZE_THRESHOLD. Falls back to a hard truncation if LLM
    is unavailable or times out.
    """
    if not _memory_call_llm:
        return content[:200] + f" […{len(content)-200}c]"

    hint = f"[{tool_name}] " if tool_name else ""
    summary = _memory_call_llm(
        system=_TOOL_SUMMARIZE_SYSTEM,
        user=content[:2000],
        max_tokens=_TOOL_RESULT_SUMMARIZE_MAX_TOKENS,
        timeout=3.0,
        task="tool_result_summarize",
    )
    if summary:
        return f"{hint}{summary.strip()}"
    return content[:200] + f" […{len(content)-200}c]"


def _apply_tool_result_summarization(messages: list, session_id: str = "") -> list:
    """Technique #1 — Summarize large tool_results before they enter history.

    Intercepts tool_result blocks in user messages that exceed
    _TOOL_RESULT_SUMMARIZE_THRESHOLD chars. Large bash outputs and file reads
    are the biggest context offenders — they sit in history forever even when
    Claude no longer needs the raw content.

    Also applies Technique #4 (dedup): if the exact same content appeared in
    a prior turn, replaces it with a back-reference instead of repeating chars.
    """
    if _TOOL_RESULT_SUMMARIZE_THRESHOLD <= 0:
        return messages

    session_hashes: dict = _tool_result_hashes.get(session_id, {}) if session_id else {}

    result = []
    turn_index = 0
    summarized_count = 0
    deduped_count = 0

    for msg in messages:
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
        elif isinstance(msg, dict):
            md = msg
        else:
            result.append(msg)
            continue

        role = md.get("role", "")
        content = md.get("content", "")

        if role != "user" or not isinstance(content, list):
            if role == "assistant":
                turn_index += 1
            result.append(msg)
            continue

        new_blocks = []
        changed = False

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                new_blocks.append(block)
                continue

            tool_name = block.get("name", block.get("tool_use_id", ""))
            inner = block.get("content", "")

            if isinstance(inner, str):
                raw_text = inner
            elif isinstance(inner, list):
                raw_text = " ".join(
                    ib.get("text", "") for ib in inner
                    if isinstance(ib, dict) and ib.get("type") == "text"
                )
            else:
                new_blocks.append(block)
                continue

            if len(raw_text) <= _TOOL_RESULT_SUMMARIZE_THRESHOLD:
                new_blocks.append(block)
                continue

            _file_path = ""
            _path_match = _re.search(
                r'(?:[A-Za-z]:\\|/)[\w./\\-]+\.\w{1,6}', raw_text[:500]
            )
            if _path_match:
                _file_path = _path_match.group(0)

            if _file_path and session_id:
                _proj = _session_last_project.get(session_id, "")
                _curr_turn = _session_turn_count.get(session_id, turn_index)
                _strategy = _get_compression_strategy(_proj, _file_path, _curr_turn)
                if _strategy == "skip":
                    new_blocks.append(block)
                    logger.debug(
                        f"STRATEGY_SKIP: {_file_path} — hot file, skipping compression"
                    )
                    continue
                elif _strategy == "aggressive":
                    _fname = os.path.basename(_file_path)
                    replacement = f"[{tool_name}] {_fname}: {raw_text[:80].strip()}..."
                    new_blocks.append({**block, "content": replacement})
                    changed = True
                    summarized_count += 1
                    logger.debug(
                        f"STRATEGY_AGGRESSIVE: {_file_path} — dead weight, 1-line summary"
                    )
                    continue

            content_hash = hashlib.md5(raw_text.encode()).hexdigest()[:16]
            if session_id and content_hash in session_hashes:
                ref_turn = session_hashes[content_hash]
                replacement = f"[Same tool output as {ref_turn} — {len(raw_text):,} chars omitted]"
                logger.info(
                    f"TOOL_DEDUP: session={session_id} hash={content_hash} "
                    f"ref={ref_turn} saved {len(raw_text):,} chars"
                )
                deduped_count += 1
            else:
                replacement = _summarize_tool_result(raw_text, tool_name=tool_name)
                logger.info(
                    f"TOOL_SUMMARIZE: session={session_id} tool={tool_name!r} "
                    f"{len(raw_text):,} chars → {len(replacement):,} chars summary"
                )
                summarized_count += 1
                if session_id:
                    session_hashes[content_hash] = f"turn {turn_index}"
                    _od_set(_tool_result_hashes, session_id, session_hashes)

            new_blocks.append({**block, "content": replacement})
            changed = True

        if changed:
            try:
                from api.models.anthropic import Message as _Msg
                result.append(_Msg(role="user", content=new_blocks))
            except Exception:
                import types
                result.append(types.SimpleNamespace(role="user", content=new_blocks))
        else:
            result.append(msg)

    if summarized_count or deduped_count:
        logger.info(
            f"CTX_ENGINEERING: summarized={summarized_count} deduped={deduped_count} "
            f"tool results for session={session_id}"
        )
    return result


def _truncate_large_outputs(messages: list) -> list:
    """Truncate oversized assistant text BEFORE compression runs.

    The compressor splits at 50% by message count, which means a 13k-token
    assistant output at position 36/51 lands in the "kept" half every time
    and is never compressed. This function caps ALL assistant text blocks
    to ~400 tokens regardless of position, preserving tool_use blocks intact.

    This is the defense-in-depth behind the CLAUDE.md output cap rule:
    even if the model ignores the rule and generates a 13k summary,
    it gets truncated here before it can pollute future calls.
    """
    from api.models.anthropic import Message as _Msg

    cap = _LARGE_OUTPUT_CHAR_CAP
    if cap <= 0:
        return messages

    result = []
    truncated_count = 0
    truncated_chars = 0

    for msg in messages:
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
        elif isinstance(msg, dict):
            md = msg
        else:
            result.append(msg)
            continue

        role = md.get("role", "")
        content = md.get("content", "")

        if role == "user" and isinstance(content, list):
            new_blocks = []
            changed = False
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    new_blocks.append(block)
                    continue
                inner = block.get("content", "")
                if isinstance(inner, str) and len(inner) > cap:
                    original_len = len(inner)
                    new_blocks.append(
                        {**block, "content": inner[:cap] + f"\n[... {original_len - cap} chars truncated ...]"}
                    )
                    changed = True
                    truncated_count += 1
                    truncated_chars += original_len - cap
                    logger.info(f"TRUNCATE_TOOL_RESULT: capped {original_len} → {cap} chars in tool_result")
                elif isinstance(inner, list):
                    inner_blocks = []
                    inner_changed = False
                    for ib in inner:
                        if isinstance(ib, dict) and ib.get("type") == "text":
                            txt = ib.get("text", "")
                            if len(txt) > cap:
                                original_len = len(txt)
                                inner_blocks.append(
                                    {**ib, "text": txt[:cap] + f"\n[... {original_len - cap} chars truncated ...]"}
                                )
                                inner_changed = True
                                truncated_count += 1
                                truncated_chars += original_len - cap
                                logger.info(f"TRUNCATE_TOOL_RESULT: capped {original_len} → {cap} chars in tool_result text block")
                            else:
                                inner_blocks.append(ib)
                        else:
                            inner_blocks.append(ib)
                    if inner_changed:
                        new_blocks.append({**block, "content": inner_blocks})
                        changed = True
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            if changed:
                try:
                    result.append(_Msg(role="user", content=new_blocks))
                except Exception:
                    import types
                    result.append(types.SimpleNamespace(role="user", content=new_blocks))
                continue

        if role != "assistant":
            result.append(msg)
            continue

        if isinstance(content, str) and len(content) > cap:
            keep_head = cap * 2 // 3
            keep_tail = cap - keep_head
            original_len = len(content)
            new_content = (
                content[:keep_head]
                + f"\n[... {original_len - cap} chars truncated ...]\n"
                + content[-keep_tail:]
            )
            truncated_count += 1
            truncated_chars += original_len - len(new_content)
            try:
                result.append(_Msg(role="assistant", content=new_content))
            except Exception:
                import types
                result.append(types.SimpleNamespace(role="assistant", content=new_content))
            continue

        if isinstance(content, list):
            new_blocks = []
            changed = False
            for block in content:
                if not isinstance(block, dict):
                    new_blocks.append(block)
                    continue

                btype = block.get("type", "")

                if btype == "tool_use":
                    new_blocks.append(block)
                    continue

                if btype == "text":
                    text = block.get("text", "")
                    if len(text) > cap:
                        keep_head = cap * 2 // 3
                        keep_tail = cap - keep_head
                        original_len = len(text)
                        new_text = (
                            text[:keep_head]
                            + f"\n[... {original_len - cap} chars truncated ...]\n"
                            + text[-keep_tail:]
                        )
                        new_blocks.append({**block, "text": new_text})
                        changed = True
                        truncated_count += 1
                        truncated_chars += original_len - len(new_text)
                        continue

                new_blocks.append(block)

            if changed:
                try:
                    result.append(_Msg(role="assistant", content=new_blocks))
                except Exception:
                    import types
                    result.append(types.SimpleNamespace(role="assistant", content=new_blocks))
            else:
                result.append(msg)
            continue

        result.append(msg)

    if truncated_count > 0:
        logger.info(
            f"TRUNCATE_OUTPUTS: capped {truncated_count} oversized assistant block(s), "
            f"saved ~{truncated_chars // 3} tokens ({truncated_chars} chars)"
        )
    return result


def _estimate_msg_tokens(msg) -> int:
    """Rough token estimate for a message — recursively includes tool results. Caches result."""
    if hasattr(msg, "_estimated_tokens"):
        return msg._estimated_tokens
    if isinstance(msg, dict) and "_estimated_tokens" in msg:
        return msg["_estimated_tokens"]

    if hasattr(msg, "model_dump"):
        d = msg.model_dump()
    elif isinstance(msg, dict):
        d = msg
    else:
        return max(1, len(str(msg)) // 3)

    role = d.get("role", "user")
    content = d.get("content", "")

    _CHARS_PER_TOKEN = 3

    def _count_recursive(obj) -> int:
        if isinstance(obj, str):
            return len(obj) // _CHARS_PER_TOKEN
        if isinstance(obj, list):
            return sum(_count_recursive(item) for item in obj)
        if isinstance(obj, dict):
            t = obj.get("type", "")
            if t == "text":
                return len(obj.get("text", "")) // _CHARS_PER_TOKEN
            if t == "tool_use":
                name_tok = len(obj.get("name", "")) // _CHARS_PER_TOKEN
                inp = obj.get("input", {})
                inp_tok = len(json.dumps(inp, default=str)) // _CHARS_PER_TOKEN if isinstance(inp, dict) else len(str(inp)) // _CHARS_PER_TOKEN
                return name_tok + inp_tok
            if t == "tool_result":
                inner = obj.get("content", "")
                return _count_recursive(inner)
            return len(obj.get("text", "")) // _CHARS_PER_TOKEN if "text" in obj else 0
        return 0

    total = max(1, _count_recursive(content))

    if hasattr(msg, "__dict__") and not isinstance(msg, dict):
        try:
            msg._estimated_tokens = total
        except (AttributeError, TypeError):
            pass

    return total


def _compress_history(messages: list, session_id: str = "") -> list:
    """
    If conversation history exceeds threshold, summarize the oldest 50% of turns
    using MEMORY_MODEL and replace them with a single summary message.
    Always keeps the last user message intact.

    BUG 5 FIX: Uses max(local_estimate, cache_growth) as the effective token count.
    local_estimate counts messages[] with len//3. cache_growth is the GROWTH of
    Anthropic's cache_read from the turn-1 baseline (which is system+tools overhead).
    Growth isolates actual message history accumulation from static overhead.
    """
    local_estimate = sum(_estimate_msg_tokens(m) for m in messages)

    cache_growth = 0
    if session_id:
        current_cache = _session_last_cache_read.get(session_id, 0)
        baseline = _session_cache_baseline.get(session_id, 0)
        if current_cache > baseline:
            cache_growth = current_cache - baseline

    total = max(local_estimate, cache_growth)

    logger.info(
        f"COMPRESS_DECISION: session={session_id} msgs={len(messages)} "
        f"local={local_estimate} growth={cache_growth} total={total} "
        f"threshold={_HISTORY_TOKEN_THRESHOLD} "
        f"fire={'YES' if total > _HISTORY_TOKEN_THRESHOLD else 'NO'}"
    )
    if total <= _HISTORY_TOKEN_THRESHOLD:
        return messages

    keep_count = min(_HISTORY_KEEP_RECENT, len(messages) - 1)
    target_cutoff = max(1, len(messages) - keep_count)
    if target_cutoff <= 0:
        return messages

    cutoff = 0
    for i in range(target_cutoff, 0, -1):
        msg = messages[i]
        if hasattr(msg, "model_dump"):
            md = msg.model_dump()
            role = md.get("role", "")
            content = md.get("content", "")
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            continue
        if role == "user":
            is_tool_result = False
            if isinstance(content, list):
                for block in content:
                    b = (
                        block
                        if isinstance(block, dict)
                        else (
                            block.model_dump() if hasattr(block, "model_dump") else {}
                        )
                    )
                    if b.get("type") == "tool_result":
                        is_tool_result = True
                        break
            if not is_tool_result:
                cutoff = i
                break

    if cutoff <= 0:
        for i in range(target_cutoff, 0, -1):
            msg = messages[i]
            if hasattr(msg, "model_dump"):
                role = msg.model_dump().get("role", "")
            elif isinstance(msg, dict):
                role = msg.get("role", "")
            else:
                continue
            if role == "user":
                cutoff = i
                logger.info(
                    f"COMPRESS_CUTOFF: pass2 — using tool_result user msg at idx={i} "
                    f"(no clean user messages in compressible range)"
                )
                break

    if cutoff <= 0 and target_cutoff > 0:
        cutoff = target_cutoff
        logger.warning(
            f"COMPRESS_CUTOFF: pass3 — forced cut at idx={target_cutoff} "
            f"(no user messages found in compressible range)"
        )

    if cutoff <= 0:
        return messages

    to_compress = messages[:cutoff]
    to_keep = messages[cutoff:]

    turns = []
    for m in to_compress:
        if hasattr(m, "model_dump"):
            md = m.model_dump()
            role = md.get("role", "?")
            content = md.get("content", "")
        elif isinstance(m, dict):
            role = m.get("role", "?")
            content = m.get("content", "")
        else:
            role, content = "?", str(m)
        if isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    parts.append(str(b))
                    continue
                if b.get("text"):
                    parts.append(b["text"])
                inner = b.get("content", "")
                if isinstance(inner, str) and inner:
                    parts.append(inner[:300])
                elif isinstance(inner, list):
                    parts.extend(
                        ib["text"][:200]
                        for ib in inner
                        if isinstance(ib, dict) and ib.get("text")
                    )
            content = " ".join(parts)
        if content and len(str(content)) > 10:
            turns.append(f"{role.upper()}: {str(content)[:500]}")

    if not turns:
        return messages

    turns_text = "\n---\n".join(turns)
    summary = None
    if _memory_call_llm:
        summary = _l1_call_llm_safe(
            session_id,
            system=_COMPRESS_SYSTEM,
            user=f"Summarize these conversation turns:\n{turns_text}",
            max_tokens=200,
            timeout=4.0,
            task="history_compress",
        )

    if not summary:
        logger.warning(
            f"LLM_TRACE: COMPRESS_FALLBACK session={session_id} — "
            f"LLM returned empty summary. Using truncation fallback. "
            f"tokens_uncompressed={sum(_estimate_msg_tokens(m) for m in to_compress)}"
        )
        fallback_parts = []
        for t in turns[:10]:
            if len(t) > 350:
                fallback_parts.append(t[:150] + " [...] " + t[-150:])
            else:
                fallback_parts.append(t)
        summary = "[Compressed - LLM unavailable] " + " | ".join(fallback_parts)
        if len(summary) > 1500:
            summary = summary[:1500] + " [...]"

    compressed_tokens = sum(_estimate_msg_tokens(m) for m in to_compress)
    logger.info(
        f"HISTORY_COMPRESS: {len(to_compress)} turns → 1 summary "
        f"({compressed_tokens} tokens → ~{len(summary) // 3} tokens)"
    )

    try:
        from api.models.anthropic import Message as _Msg

        summary_msg = _Msg(
            role="user", content=f"[Earlier conversation summary: {summary}]"
        )
    except Exception as _import_err:
        logger.warning(
            f"HISTORY_COMPRESS: Message import failed: {_import_err}, using SimpleNamespace"
        )
        import types

        summary_msg = types.SimpleNamespace(
            role="user",
            content=f"[Earlier conversation summary: {summary}]",
        )

    _kept_tokens = sum(_estimate_msg_tokens(m) for m in to_keep)
    _HARD_CAP_MULTIPLIER = 3
    if _kept_tokens > _HISTORY_TOKEN_THRESHOLD * _HARD_CAP_MULTIPLIER:
        logger.warning(
            f"HISTORY_COMPRESS: kept messages still {_kept_tokens} tokens "
            f"(> {_HISTORY_TOKEN_THRESHOLD * _HARD_CAP_MULTIPLIER} hard cap) — "
            f"truncating tool results in kept messages"
        )
        _trunc_cap = 2000
        for i, km in enumerate(to_keep):
            _kd = km.model_dump() if hasattr(km, "model_dump") else (km if isinstance(km, dict) else None)
            if not _kd:
                continue
            _kc = _kd.get("content", "")
            if not isinstance(_kc, list):
                continue
            _changed = False
            _new_blocks = []
            for blk in _kc:
                if isinstance(blk, dict) and blk.get("type") == "tool_result":
                    inner = blk.get("content", "")
                    if isinstance(inner, str) and len(inner) > _trunc_cap:
                        _new_blocks.append({**blk, "content": inner[:_trunc_cap] + f"\n[...{len(inner)-_trunc_cap}c truncated by safety cap...]"})
                        _changed = True
                    else:
                        _new_blocks.append(blk)
                else:
                    _new_blocks.append(blk)
            if _changed:
                try:
                    from api.models.anthropic import Message as _MsgCap
                    to_keep[i] = _MsgCap(role=_kd["role"], content=_new_blocks)
                except Exception:
                    to_keep[i] = {"role": _kd["role"], "content": _new_blocks}

    return [summary_msg, *to_keep]


def _compress_inherited_context(messages: list, session_id: str) -> list:
    """Detect and compress inherited session context at the start of a new session.

    When a session ends, its full conversation history stays in Anthropic's 1h cache.
    A new session within that hour inherits 20k+ tokens of dead context that gets
    re-read on every single turn.  This function runs ONCE at session start:
    if messages[] exceeds INHERITED_CONTEXT_THRESHOLD tokens, it compresses all
    but the last 2 messages into a 2-3 line summary via the local LLM.
    """
    if _session_compressed.get(session_id):
        return messages

    if not messages:
        _od_set(_session_compressed, session_id, True)
        return messages

    total_tokens = sum(_estimate_msg_tokens(m) for m in messages)
    if total_tokens < INHERITED_CONTEXT_THRESHOLD:
        _od_set(_session_compressed, session_id, True)
        return messages

    keep_recent = min(2, len(messages))
    to_compress = messages[:-keep_recent] if keep_recent else messages
    to_keep = messages[-keep_recent:] if keep_recent else []

    if not to_compress:
        _od_set(_session_compressed, session_id, True)
        return messages

    turns = []
    for m in to_compress:
        if hasattr(m, "model_dump"):
            md = m.model_dump()
            role = md.get("role", "?")
            content = md.get("content", "")
        elif isinstance(m, dict):
            role = m.get("role", "?")
            content = m.get("content", "")
        else:
            role, content = "?", str(m)
        if isinstance(content, list):
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    parts.append(str(b))
                    continue
                if b.get("text"):
                    parts.append(b["text"])
                inner = b.get("content", "")
                if isinstance(inner, str) and inner:
                    parts.append(inner[:300])
                elif isinstance(inner, list):
                    parts.extend(
                        ib["text"][:200]
                        for ib in inner
                        if isinstance(ib, dict) and ib.get("text")
                    )
            content = " ".join(parts)
        if content and len(str(content)) > 10:
            turns.append(f"{role.upper()}: {str(content)[:500]}")

    if not turns:
        _od_set(_session_compressed, session_id, True)
        return messages

    turns_text = "\n---\n".join(turns)
    summary = None
    if _memory_call_llm:
        summary = _l1_call_llm_safe(
            session_id,
            system=(
                "Summarize this inherited session context in 1-2 lines (max 50 words). "
                "Keep: decisions made, file paths modified, current task state, "
                "errors encountered. No filler, no explanation."
            ),
            user=f"Previous session context:\n{turns_text}",
            max_tokens=80,
            timeout=4.0,
            task="inherited_context_compress",
        )

    _od_set(_session_compressed, session_id, True)

    if not summary:
        logger.warning(
            f"INHERITED_COMPRESS: LLM returned empty summary for session={session_id}, "
            f"falling back to truncation ({total_tokens} inherited tokens dropped)"
        )
        return to_keep

    compressed_tokens = sum(_estimate_msg_tokens(m) for m in to_compress)
    logger.info(
        f"INHERITED_COMPRESS: session={session_id} compressed {len(to_compress)} inherited msgs "
        f"({compressed_tokens} tokens → ~{len(summary) // 3} tokens)"
    )

    try:
        from api.models.anthropic import Message as _Msg
        summary_msg = _Msg(
            role="user",
            content=f"[Previous session summary: {summary}]",
        )
    except Exception:
        import types
        summary_msg = types.SimpleNamespace(
            role="user",
            content=f"[Previous session summary: {summary}]",
        )
    return [summary_msg, *to_keep]


router = APIRouter()


_PROXY_SESSION_ID = uuid.uuid4().hex[:12]

_SKIP_FOLDERS = {
    "app",
    "src",
    "lib",
    "bin",
    "tmp",
    "home",
    "users",
    "desktop",
    "documents",
    "downloads",
    "var",
    "etc",
    "usr",
    "opt",
}

_WIN_PATH_RE = _re.compile(
    r"[A-Za-z]:\\Users\\[^\\]+\\(?:Desktop|Documents|Projects|Dev|Code|Github|Work|Repos)\\([^\\ \n\r\t]+)",
    _re.IGNORECASE,
)
_UNIX_PATH_RE = _re.compile(
    r"/(?:home|Users)/[^/]+/(?:Desktop|Documents|projects|dev|code|github|work|repos)/([^/ \n\r\t]+)",
    _re.IGNORECASE,
)
_KEYWORD_RE = _re.compile(
    r"(?:project|repo|workspace|cwd|working on|working in)[:\s]+([^\s\n,\.]{3,40})",
    _re.IGNORECASE,
)

_INVALID_IDS = {
    "n_-",
    "n_",
    "-",
    "_",
    "ok",
    "user",
    "users",
    "desktop",
    "documents",
    "home",
    "app",
    "src",
    "lib",
    "bin",
    "tmp",
    "dev",
    "code",
    "work",
    "repos",
    "github",
    "projects",
    "c",
    "d",
    "e",
    "python",
    "structure",
    "structure_n_",
    "f",
}


def _sanitize_id(s: str, max_len: int = 40) -> str:
    """Sanitize a string into a safe identifier."""
    s = s.lower()
    s = _re.sub(r"[^a-z0-9_-]", "_", s)
    s = _re.sub(r"_+", "_", s)
    s = s.strip("_")
    s = s[:max_len]
    return s or "default"


def _is_valid_id(s: str) -> bool:
    """Check if a project ID is valid (min length + not in garbage list)."""
    return len(s) >= 2 and s not in _INVALID_IDS


def _detect_project_id(raw_request: Request, request_data) -> str:
    """Auto-detect project ID from headers, request body, system prompt paths, or keywords."""

    system_text = ""
    if hasattr(request_data, "system") and request_data.system:
        if isinstance(request_data.system, str):
            system_text = request_data.system
        elif isinstance(request_data.system, list):
            system_text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in request_data.system
            )

    logger.debug(
        f"[Project] system_prompt_first_500='{system_text[:500].replace(chr(10), ' ')}'"
    )
    logger.debug(f"[Project] headers={dict(raw_request.headers)}")

    header_pid = raw_request.headers.get("X-Project-Id", "")
    if header_pid and header_pid != "default":
        logger.info(f"[Project] step=0 candidate='{header_pid}' source='X-Project-Id'")
        return header_pid

    body_cwd = None
    if isinstance(request_data, dict):
        body_cwd = request_data.get("cwd")
    elif hasattr(request_data, "cwd"):
        body_cwd = request_data.cwd

    if body_cwd:
        name = os.path.basename(str(body_cwd).rstrip("/\\"))
        candidate = _sanitize_id(name)
        valid = candidate not in _INVALID_IDS
        logger.info(
            f"[Project] step=1 candidate='{candidate}' source='body_cwd' valid={valid}"
        )
        if valid:
            return candidate

    if system_text:
        patterns = [
            (r"cwd[:\s]+([A-Za-z]:\\[^\n\r]+)", "win_cwd_label"),
            (r"working directory[:\s]+([A-Za-z]:\\[^\n\r]+)", "win_wd_label"),
            (r"Current directory[:\s]+([A-Za-z]:\\[^\n\r]+)", "win_curr_label"),
            (r"cwd[:\s]+(/[^\n\r]+)", "unix_cwd_label"),
            (r"working directory[:\s]+(/[^\n\r]+)", "unix_wd_label"),
        ]
        for i, (pat, label) in enumerate(patterns):
            m = _re.search(pat, system_text, _re.IGNORECASE)
            if m:
                path_val = m.group(1).strip()
                name = os.path.basename(path_val.rstrip("/\\"))
                candidate = _sanitize_id(name)
                valid = candidate not in _INVALID_IDS
                logger.info(
                    f"[Project] pattern={i} label='{label}' match='{path_val}' → '{candidate}' valid={valid}"
                )
                if valid:
                    return candidate

    cwd_header = raw_request.headers.get("X-Cwd", "") or raw_request.headers.get(
        "X-Working-Dir", ""
    )
    if cwd_header:
        name = os.path.basename(cwd_header.rstrip("/\\"))
        candidate = _sanitize_id(name)
        valid = candidate not in _INVALID_IDS
        logger.info(
            f"[Project] step=3 candidate='{candidate}' source='X-Cwd' valid={valid}"
        )
        if valid:
            return candidate

    if system_text:
        win_path_regex = _re.compile(
            r'([A-Za-z]:\\Users\\[^\\]+\\(?:Desktop|Documents|Projects|Dev|Code|Work|Repos|Github)\\([^\\"\n\r\s]{2,40}))',
            _re.IGNORECASE,
        )
        matches = list(win_path_regex.finditer(system_text))
        if matches:
            last_match = matches[-1]
            folder_name = last_match.group(2).strip()
            candidate = _sanitize_id(folder_name)
            valid = candidate not in _INVALID_IDS
            logger.info(
                f"[Project] step=4 candidate='{candidate}' source='win_path_last' valid={valid}"
            )
            if valid:
                return candidate

        unix_patterns = [
            r"/[Uu]sers/[^/]+/(?:Desktop|projects|dev|code|work)/([^/\n\s]{2,40})",
            r"/home/[^/]+/([^/\n\s]{2,40})",
        ]
        for i, pat in enumerate(unix_patterns):
            m = _re.search(pat, system_text)
            if m:
                folder_name = m.group(1).strip()
                candidate = _sanitize_id(folder_name)
                valid = candidate not in _INVALID_IDS
                logger.info(
                    f"[Project] step=5. {i} candidate='{candidate}' source='unix_path' valid={valid}"
                )
                if valid:
                    return candidate

        stable_id = "proj_" + hashlib.sha256(system_text.encode()).hexdigest()[:8]
        logger.info(f"[Project] FINAL id='{stable_id}' source='hash_stable'")
        return stable_id

    logger.info("[Project] FINAL id='default' source='last_resort'")
    return "default"


def _detect_session_id(raw_request: Request, request_data=None) -> str:
    """Stable session ID — must NOT change when Claude Code updates its system prompt.

    Priority: explicit header > CWD hash (stable) > system prompt hash (fallback).
    CWD-based IDs survive Claude Code updates; system prompt hashes do not.
    """
    header = raw_request.headers.get("X-Session-Id", "")
    if header and header != "default":
        return header

    cwd = (
        raw_request.headers.get("X-Cwd", "")
        or raw_request.headers.get("x-cwd", "")
    )

    if not cwd and request_data is not None:
        system = getattr(request_data, "system", None)
        if system:
            raw = system if isinstance(system, str) else str(system)
            win = _re.search(r"cwd[:\s]+([A-Za-z]:\\[^\n\r]+)", raw)
            if win:
                cwd = win.group(1).strip()
            else:
                unix = _re.search(r"cwd[:\s]+(/[^\n\r]+)", raw)
                if unix:
                    cwd = unix.group(1).strip()

    if cwd:
        normalized = os.path.normpath(os.path.expanduser(cwd)).lower()
        return "cwd_" + hashlib.sha256(normalized.encode()).hexdigest()[:12]

    if request_data is not None:
        system = getattr(request_data, "system", None)
        if system:
            logger.warning(
                "SESSION_ID_FALLBACK: CWD not found in request. "
                "Using system prompt hash — session ID will change on Claude Code updates. "
                "This causes unnecessary cold starts."
            )
            import re as _re2
            raw = system if isinstance(system, str) else str(system)
            base = _re2.sub(r"<memory_context>.*?</memory_context_end>", "", raw, flags=_re2.DOTALL)
            base = _re2.sub(r"\[Memory context[^\]]*\]", "", base).strip()
            return "sys_" + hashlib.sha256(base.encode()).hexdigest()[:16]

    return _PROXY_SESSION_ID


_recent_session_registry: _OD[str, tuple] = _OD()

def _check_session_invalidation(project_id: str, session_id: str) -> None:
    """Detect when session ID changes for an active project and log it.

    Re-warm is NOT triggered here because the prewarm function requires
    injection_block/model/tools which aren't available at session detection time.
    The natural request flow will handle cache warming via the injection pipeline.
    """
    import time as _t
    last = _recent_session_registry.get(project_id)
    if last:
        last_sid, last_ts = last
        time_since = _t.time() - last_ts
        if last_sid != session_id and time_since < 3600:
            logger.warning(
                f"SESSION_INVALIDATED | project={project_id} | "
                f"old_session={last_sid[:8]}... | new_session={session_id[:8]}... | "
                f"age={time_since:.0f}s | "
                f"cause=likely_claude_code_update | "
                f"impact=cache_cold_start_next_5_turns"
            )
    _od_set(_recent_session_registry, project_id, (session_id, _t.time()))


@router.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    provider: BaseProvider = Depends(get_provider_for_request),
    settings: Settings = Depends(get_settings),
):
    """Create a message — with memory middleware injected."""
    global _total_requests, _main_event_loop
    _total_requests += 1
    if _main_event_loop is None:
        _main_event_loop = asyncio.get_running_loop()

    try:
        if not request_data.messages:
            raise InvalidRequestError("messages cannot be empty")

        optimized = try_optimizations(request_data, settings)
        if optimized is not None:
            return optimized

        _max_tok = getattr(request_data, "max_tokens", 9999)
        if _max_tok <= 10 and not getattr(request_data, "tools", None):
            _n_msgs = len(request_data.messages)
            _has_system = bool(getattr(request_data, "system", None))
            _last_content = ""
            for _m in reversed(request_data.messages):
                _r = getattr(_m, "role", "") if hasattr(_m, "role") else _m.get("role", "")
                if _r == "user":
                    _c = getattr(_m, "content", "") if hasattr(_m, "content") else _m.get("content", "")
                    if isinstance(_c, list):
                        _last_content = " ".join(
                            b.get("text", "") for b in _c if isinstance(b, dict) and b.get("type") == "text"
                        )
                    elif isinstance(_c, str):
                        _last_content = _c
                    break
            logger.debug(
                f"PREFLIGHT_LEAK: max_tokens={_max_tok} msgs={_n_msgs} "
                f"has_system={_has_system} content={repr((_last_content or '')[:80])}"
            )

        project_id = _detect_project_id(raw_request, request_data)
        session_id = _detect_session_id(raw_request, request_data)
        _check_session_invalidation(project_id, session_id)

        _token = (
            raw_request.headers.get("x-api-key", "")
            or raw_request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        )
        _persist_token = ""
        _persist_model = ""
        if _token and _token.lower() not in ("fake-key", "dummy", "test", "", "sk-placeholder-key-for-proxy"):
            _od_set(_project_last_token, project_id, _token)
            _persist_token = _token
        _req_model = getattr(request_data, "original_model", None) or getattr(request_data, "model", "")
        if _req_model and "claude" in _req_model.lower():
            _od_set(_project_last_model, project_id, _req_model)
            _persist_model = _req_model
        if _persist_token or _persist_model:
            await asyncio.get_running_loop().run_in_executor(
                None, _persist_project_meta, project_id, _persist_model, _persist_token
            )

        _raw_msgs_for_hash = [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in request_data.messages
        ]
        _raw_fingerprint = hashlib.md5(
            json.dumps({
                "messages": _raw_msgs_for_hash,
                "model": getattr(request_data, "model", ""),
            }, sort_keys=True, default=str).encode()
        ).hexdigest()

        _cached_entry = _dedup_cache.get(session_id)
        if _cached_entry and (_cached_entry.get("raw_hash", _cached_entry.get("hash")) == _raw_fingerprint):
            _age = _time.time() - _cached_entry["ts"]
            if (
                _age < DEDUP_TTL_SECONDS
                and _cached_entry["output_tokens"] >= DEDUP_MIN_OUT_TOKENS
            ):
                logger.info(
                    f"EARLY_DEDUP_HIT: session={session_id} — replaying "
                    f"{len(_cached_entry['chunks'])} cached SSE chunks "
                    f"(age={_age:.1f}s, saved full pipeline)"
                )
                _replay_chunks = _cached_entry["chunks"]

                async def _replay_early():
                    for c in _replay_chunks:
                        yield c

                return StreamingResponse(
                    _replay_early(),
                    media_type="text/event-stream",
                    headers={
                        "X-Accel-Buffering": "no",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Project-Id": project_id,
                        "X-Session-Id": session_id,
                        "X-Dedup": "early-hit",
                    },
                )

        _inflight_key = f"{session_id}:{_raw_fingerprint}"
        _existing_future = _inflight_requests.get(_inflight_key)
        if _existing_future is not None and not _existing_future.done():
            logger.info(
                f"INFLIGHT_COALESCE: session={session_id} — awaiting in-flight request"
            )
            try:
                _coalesced_chunks = await asyncio.wait_for(_existing_future, timeout=120.0)
                if _coalesced_chunks:
                    async def _replay_coalesced():
                        for c in _coalesced_chunks:
                            yield c

                    return StreamingResponse(
                        _replay_coalesced(),
                        media_type="text/event-stream",
                        headers={
                            "X-Accel-Buffering": "no",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Project-Id": project_id,
                            "X-Session-Id": session_id,
                            "X-Dedup": "coalesced",
                        },
                    )
            except (asyncio.TimeoutError, Exception) as _coal_err:
                logger.debug(f"INFLIGHT_COALESCE: failed, proceeding normally: {_coal_err}")
                _inflight_requests.pop(_inflight_key, None)

        _inflight_future: asyncio.Future = asyncio.get_running_loop().create_future()
        _inflight_requests[_inflight_key] = _inflight_future
        if len(_inflight_requests) > _MAX_INFLIGHT_ENTRIES:
            _inflight_requests.popitem(last=False)

        _session_is_first = False
        _injection_done = False
        if MEMORY_ENABLED:
            if _is_real_message(request_data):
                real_count = _count_real_user_messages(request_data.messages)
                if real_count >= 2:
                    _od_set(_session_boot_done, session_id, True)

                boot_done = _session_boot_done.get(session_id, False)
                if not boot_done and real_count == 1:
                    msgs = request_data.messages
                    last_asst = next(
                        (m for m in reversed(msgs) if (getattr(m, "role", "") if hasattr(m, "role") else m.get("role", "")) == "assistant"),
                        None
                    )
                    if last_asst:
                        content = getattr(last_asst, "content", []) if hasattr(last_asst, "content") else last_asst.get("content", [])
                        if isinstance(content, list):
                            tool_names = [
                                b.get("name", "") for b in content
                                if isinstance(b, dict) and b.get("type") == "tool_use"
                            ]
                            if any("memory_remember" in n for n in tool_names):
                                logger.info(f"BOOT_GATE: blocking post-boot tool call for session={session_id}")
                                return _dict_to_sse_response({
                                    "id": f"msg_{uuid.uuid4().hex[:12]}",
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [{"type": "text", "text": ""}],
                                    "model": request_data.model,
                                    "stop_reason": "end_turn",
                                    "stop_sequence": None,
                                    "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0,
                                              "cache_read_input_tokens": 0, "output_tokens": 0}
                                })

                _last_user = _extract_last_user_text(request_data.messages)
                _is_command = bool(_MEMORY_COMMAND_RE.match(_last_user)) if _last_user else False

                _skip_injection = False
                _MIN_INJECTION_VALUE = float(os.getenv("MIN_INJECTION_VALUE", "0.05"))
                _session_val = _compute_session_value_score(session_id)
                _turn_ct = _session_turn_count.get(session_id, 0)
                if _turn_ct >= 3 and _session_val < _MIN_INJECTION_VALUE and _session_val > 0:
                    _cached_fallback = _last_injection_block.get(session_id, "")
                    if not _cached_fallback:
                        logger.debug(
                            f"PROPORTIONAL_SKIP: session={session_id} value={_session_val} "
                            f"< {_MIN_INJECTION_VALUE} — skipping injection"
                        )
                        fast_response = None
                        _skip_injection = True
                        _injection_done = True

                _boot_inj_ts = _session_boot_injected.get(session_id)
                if _boot_inj_ts and (_time.time() - _boot_inj_ts) < DEDUP_WINDOW:
                    cached_block = _last_injection_block.get(session_id, "")
                    if cached_block:
                        _inject_system(request_data, cached_block)
                        _injection_done = True
                        request_data._memory_block_stable = True
                        logger.debug(
                            f"BOOT_DEDUP: session={session_id} reused cached block "
                            f"({len(cached_block)} chars, age={_time.time() - _boot_inj_ts:.1f}s)"
                        )
                        fast_response = None
                        _skip_injection = True

                if not _skip_injection:
                    _mem_hash = _get_memory_hash(project_id)
                    _notifs = _notifs_api
                    _cached_block = _hot_cache.get(project_id, _mem_hash) if (_hot_cache and not _is_command) else None

                    if _cached_block and not _is_command:
                        _inject_system(request_data, _cached_block)
                        _injection_done = True
                        request_data._memory_block_stable = True
                        _od_set(_last_injection_block, session_id, _cached_block)
                        logger.debug(f"HOT_CACHE: hit project={project_id} — skipped DB queries")
                        fast_response = None
                    else:
                        if _notifs:
                            _notifs.push("cache", f"Cache miss for {project_id} — rebuilding injection", project_id)

                    if not (_cached_block and not _is_command):
                        seen_ids = _session_seen_memory_ids.get(session_id, set())

                        user_query = ""
                        try:
                            _last_user_msg = next(
                                (m for m in reversed(request_data.messages or [])
                                 if (getattr(m, 'role', None) or m.get('role')) == 'user'),
                                None
                            )
                            if _last_user_msg:
                                _content = getattr(_last_user_msg, 'content', None) or _last_user_msg.get('content', '')
                                if isinstance(_content, str):
                                    user_query = _content[:200]
                                elif isinstance(_content, list):
                                    user_query = ' '.join(
                                        b.get('text', '') for b in _content
                                        if isinstance(b, dict) and b.get('type') == 'text'
                                    )[:200]
                        except Exception:
                            pass

                        _ev_loop = asyncio.get_running_loop()
                        if user_query:
                            _switched = await _ev_loop.run_in_executor(
                                None, _is_topic_switch, session_id, user_query,
                            )
                            if _switched:
                                seen_ids = set()
                                logger.debug(f"[SmartInject] Topic switch — reset seen IDs session={session_id}")

                        _inject_start = _time.time()
                        request_data, fast_response = await _ev_loop.run_in_executor(
                            None, process_memory,
                            request_data, project_id, session_id, seen_ids,
                        )

                        _newly = getattr(request_data, '_newly_seen_ids', set())
                        _od_set(_session_seen_memory_ids, session_id, seen_ids | _newly)

                        _new_block_check = ""
                        _sys_check = getattr(request_data, "system", None)
                        if _sys_check:
                            _raw_check = _sys_check if isinstance(_sys_check, str) else str(_sys_check)
                            _blk_check = _re.search(
                                r"(<memory_context[^>]*>.*?</memory_context_end>)",
                                _raw_check, _re.DOTALL,
                            )
                            if _blk_check:
                                _new_block_check = _blk_check.group(1)
                        if _new_block_check:
                            _od_set(_last_injection_data, session_id, {
                                "ts": _time.time(),
                                "entries_injected": len(_newly),
                                "tokens_estimated": round(len(_new_block_check.split()) * 1.3),
                                "latency_ms": int((_time.time() - _inject_start) * 1000),
                                "block_preview": _new_block_check[:600],
                                "topic_switched": len(seen_ids) == 0,
                            })
                    if fast_response is not None:
                        return _dict_to_sse_response(fast_response)

                    _new_block = ""
                    _sys = getattr(request_data, "system", None)
                    if _sys:
                        _raw = _sys if isinstance(_sys, str) else str(_sys)
                        _blk = _re.search(
                            r"(<memory_context[^>]*>.*?</memory_context_end>)",
                            _raw, _re.DOTALL,
                        )
                        if _blk:
                            _new_block = _blk.group(1)

                    _new_block_hash = hashlib.sha256(_new_block.encode()).hexdigest()[:16] if _new_block else ""
                    _old_block_hash = _last_injection_hash.get(session_id, "")

                    if _new_block_hash != _old_block_hash or _is_command:
                        request_data._memory_block_stable = False
                        _od_set(_last_injection_hash, session_id, _new_block_hash)
                        _od_set(_last_injection_block, session_id, _new_block)
                        if _hot_cache:
                            _hot_cache.set(project_id, _mem_hash, _new_block)
                            if _notifs_api:
                                _notifs_api.push("cache", f"Injection block cached for {project_id} ({len(_new_block)} chars)", project_id)
                        logger.debug(
                            f"MEMORY_BLOCK_CHANGED: session={session_id} "
                            f"old='{_old_block_hash}' new='{_new_block_hash}'"
                        )
                    else:
                        request_data._memory_block_stable = True
                        logger.debug(
                            f"MEMORY_BLOCK_STABLE: session={session_id} (already injected by process_memory)"
                        )

                    _od_set(_session_boot_injected, session_id, _time.time())

                _final_sys = getattr(request_data, "system", None)
                if not _final_sys and not _injection_done:
                    logger.warning(
                        f"EMPTY_SYSTEM_PROMPT: session={session_id} — system prompt is "
                        f"None/empty after injection. This will cause minimal input tokens "
                        f"and potential cache bust."
                    )
                    _fallback = _last_injection_block.get(session_id, "")
                    if _fallback:
                        _inject_system(request_data, _fallback)
                        _injection_done = True
                elif not _final_sys and _injection_done:
                    logger.debug(f"EMPTY_SYSTEM_PROMPT: skipped — injection already done for session={session_id}")
                

                _session_is_first = False

        if MEMORY_ENABLED and session_id and project_id:
            if not _session_ctx_injected.get(session_id):
                _od_set(_session_ctx_injected, session_id, True)
                _parent_block = _get_parent_context_block(project_id, session_id)
                if _parent_block:
                    _inject_system(request_data, _parent_block)
                    _od_set(_last_injection_block, session_id,
                            _last_injection_block.get(session_id, "") + _parent_block)
                    logger.info(
                        f"SUBAGENT_CTX: injected parent context for session={session_id} "
                        f"project={project_id} ({len(_parent_block)} chars)"
                    )

        if MEMORY_ENABLED and request_data.messages:
            _loop = asyncio.get_running_loop()
            request_data.messages = await _loop.run_in_executor(
                None, _compress_inherited_context,
                request_data.messages, session_id,
            )
            request_data.messages = await _loop.run_in_executor(
                None, _apply_tool_result_summarization,
                request_data.messages, session_id,
            )
            request_data.messages = _truncate_large_outputs(request_data.messages)
            request_data.messages = await _loop.run_in_executor(
                None, _compress_history, request_data.messages, session_id,
            )

            _post_compress_tokens = sum(_estimate_msg_tokens(m) for m in request_data.messages)
            _SAFETY_MULTIPLIER = 5
            if _post_compress_tokens > _HISTORY_TOKEN_THRESHOLD * _SAFETY_MULTIPLIER:
                logger.warning(
                    f"SAFETY_CAP: session={session_id} still {_post_compress_tokens} tokens "
                    f"after compression (limit={_HISTORY_TOKEN_THRESHOLD * _SAFETY_MULTIPLIER}) — "
                    f"re-truncating tool results"
                )
                request_data.messages = _truncate_large_outputs(request_data.messages)

            _current_cache = _session_last_cache_read.get(session_id, 0)
            _baseline = _session_cache_baseline.get(session_id, 0)
            _cache_growth = _current_cache - _baseline if _current_cache > _baseline else 0
            _COMPACT_TRIGGER = _HISTORY_TOKEN_THRESHOLD * 3
            _last_out = _session_last_output_tokens.get(session_id, 0)
            if (
                _cache_growth > _COMPACT_TRIGGER
                and _session_turn_count.get(session_id, 0) >= SMART_COMPACT_MIN_TURNS
                and not _session_compact_count.get(session_id, 0)
                and _last_out >= DEDUP_MIN_OUT_TOKENS
            ):
                logger.info(
                    f"CACHE_COMPACT_TRIGGER: session={session_id} "
                    f"cache_growth={_cache_growth:,} > {_COMPACT_TRIGGER:,} — "
                    f"calling _smart_compact directly to reset Anthropic cache prefix"
                )
                try:
                    _compact_result = await _smart_compact(
                        session_id, project_id,
                        request_data.messages,
                        getattr(request_data, "model", ""),
                    )
                    if _compact_result is not None:
                        request_data.messages = _compact_result
                except Exception:
                    pass

            _last_user_text = ""
            if request_data.messages:
                _last_m = request_data.messages[-1]
                _last_md = _last_m.model_dump() if hasattr(_last_m, "model_dump") else (_last_m if isinstance(_last_m, dict) else {})
                _lc = _last_md.get("content", "")
                if isinstance(_lc, str):
                    _last_user_text = _lc.strip()
                elif isinstance(_lc, list):
                    for _lb in _lc:
                        if isinstance(_lb, dict) and _lb.get("type") == "text":
                            _last_user_text = _lb.get("text", "").strip()
                            break

            if _last_user_text.startswith("/compact"):
                _od_set(_session_compact_count, session_id, 0)
                _od_set(_session_compact_reset_ctx, session_id, 0)
                _od_set(_session_compact_last_turn, session_id, 0)
                if not _content_density_check(request_data.messages):
                    logger.warning(
                        f"COMPACT_INTERCEPT: session={session_id} — density too low, "
                        f"skipping DB save. Claude Code will run its own compact."
                    )
                else:
                    _compact_result = await _smart_compact(
                        session_id, project_id,
                        request_data.messages,
                        getattr(request_data, "model", ""),
                    )
                    if _compact_result is not None:
                        request_data.messages = _compact_result

            _turn = _session_turn_count.get(session_id, 0) + 1
            _od_set(_session_turn_count, session_id, _turn)
            _update_file_access(project_id, request_data.messages, _turn)
            _total_tok = sum(_estimate_msg_tokens(m) for m in request_data.messages)
            _tok_hist = _session_tokens.get(session_id, [])
            _tok_hist.append(_total_tok)
            _od_set(_session_tokens, session_id, _tok_hist[-30:])
            _od_set(_session_model, session_id, getattr(request_data, "model", ""))

            _od_set(_session_last_messages, session_id, list(request_data.messages))
            _od_set(_session_last_project, session_id, project_id)

            _last_out_for_compact = _session_last_output_tokens.get(session_id, 0)
            if _turn >= SMART_COMPACT_MIN_TURNS and _last_out_for_compact >= DEDUP_MIN_OUT_TOKENS:
                _compact_result = await _smart_compact(
                    session_id, project_id,
                    request_data.messages,
                    getattr(request_data, "model", ""),
                )
                if _compact_result is not None:
                    request_data.messages = _compact_result

        if _is_real_message(request_data):
            _touch_keepalive(session_id, provider, request_data)
            _increment_real_msg_count(session_id, project_id)

        _post_msgs_for_hash = [
            m.model_dump() if hasattr(m, "model_dump") else m
            for m in request_data.messages
        ]
        _msg_fingerprint = hashlib.md5(
            json.dumps({
                "messages": _post_msgs_for_hash,
                "model": getattr(request_data, "model", ""),
            }, sort_keys=True, default=str).encode()
        ).hexdigest()

        _cached_entry = _dedup_cache.get(session_id)
        if _cached_entry and _cached_entry["hash"] == _msg_fingerprint:
            _age = _time.time() - _cached_entry["ts"]
            if (
                _age < DEDUP_TTL_SECONDS
                and _cached_entry["output_tokens"] >= DEDUP_MIN_OUT_TOKENS
            ):
                logger.info(
                    f"DEDUP_HIT: session={session_id} — replaying "
                    f"{len(_cached_entry['chunks'])} cached SSE chunks "
                    f"(age={_age:.1f}s, out_tok={_cached_entry['output_tokens']})"
                )

                if not _inflight_future.done():
                    _inflight_future.set_result(_cached_entry["chunks"])
                _inflight_requests.pop(_inflight_key, None)

                _replay_chunks = _cached_entry["chunks"]

                async def _replay_cached():
                    for c in _replay_chunks:
                        yield c

                return StreamingResponse(
                    _replay_cached(),
                    media_type="text/event-stream",
                    headers={
                        "X-Accel-Buffering": "no",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Project-Id": project_id,
                        "X-Session-Id": session_id,
                        "X-Dedup": "hit",
                    },
                )

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        log_request_compact(logger, request_id, request_data)

        input_tokens = get_token_count(
            request_data.messages, request_data.system, request_data.tools
        )

        response_stream = provider.stream_response(
            request_data,
            input_tokens=input_tokens,
            request_id=request_id,
        )

        if MEMORY_ENABLED and _session_is_first:
            pass

        _fp = _msg_fingerprint
        _fp_raw = _raw_fingerprint
        _sid = session_id
        _ifl_key = _inflight_key
        _ifl_future = _inflight_future

        async def _stream_and_cache():
            """Stream response to client AND cache SSE chunks for dedup replay.

            Parses output_tokens from the message_delta SSE event.
            Caches ALL responses (Bug Delta fix). DEDUP_MIN_OUT_TOKENS is checked
            at replay time to prefer real responses over preflight fragments.
            Also tracks token usage for dashboard analytics (BUG 4 fix).
            """
            chunks: list[str] = []
            _output_tokens = 0
            _usage_data: dict = {}
            _model_name = ""
            try:
                async for chunk in response_stream:
                    chunks.append(chunk)
                    if "message_start" in chunk:
                        try:
                            _data_part = chunk.split("data: ", 1)
                            if len(_data_part) > 1:
                                _evt = json.loads(_data_part[1].split("\n", 1)[0])
                                _msg = _evt.get("message", {})
                                _model_name = _msg.get("model", "")
                                _start_usage = _msg.get("usage", {})
                                if _start_usage:
                                    _usage_data.update(_start_usage)
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
                    if "message_delta" in chunk:
                        try:
                            _data_part = chunk.split("data: ", 1)
                            if len(_data_part) > 1:
                                _evt = json.loads(_data_part[1].split("\n", 1)[0])
                                _delta_usage = _evt.get("usage", {})
                                _output_tokens = _delta_usage.get(
                                    "output_tokens", _output_tokens
                                )
                                if _delta_usage:
                                    _usage_data.update(_delta_usage)
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                _stop_keepalive(_sid)
                if not _ifl_future.done():
                    _ifl_future.set_result(None)
                _inflight_requests.pop(_ifl_key, None)
                return

            if _model_name:
                _in_tok = _usage_data.get("input_tokens", 0)
                _cache_r = _usage_data.get("cache_read_input_tokens", 0)
                _cache_w = _usage_data.get("cache_creation_input_tokens", 0)
                logger.info(
                    f"ANTHROPIC_RESPONSE: session={_sid} model={_model_name} "
                    f"input={_in_tok:,} cache_read={_cache_r:,} cache_write={_cache_w:,} "
                    f"output={_output_tokens:,}"
                )

                if _cache_r > 0:
                    _od_set(_session_last_cache_read, _sid, _cache_r)
                    if _sid not in _session_cache_baseline:
                        _od_set(_session_cache_baseline, _sid, _cache_r)
                        logger.debug(
                            f"CACHE_BASELINE: session={_sid} baseline={_cache_r:,} "
                            f"(system+tools+injection overhead)"
                        )

            if _usage_data:
                try:
                    from api.hot_cache import extract_and_track_tokens
                    _track_pid = project_id or os.getenv("MEMORY_PROJECT_ID", "default")
                    if not project_id:
                        logger.info(f"TOKEN_TRACK: project_id empty, using fallback '{_track_pid}'")
                    extract_and_track_tokens(
                        {"usage": _usage_data, "model": _model_name},
                        _track_pid,
                    )
                except Exception:
                    pass

                try:
                    asyncio.get_running_loop().call_soon(
                        lambda: asyncio.ensure_future(broadcast_sse_event({
                            "type": "response",
                            "project_id": project_id,
                            "session_id": _sid,
                            "model": _model_name,
                            "input_tokens": _usage_data.get("input_tokens", 0),
                            "cache_read": _usage_data.get("cache_read_input_tokens", 0),
                            "cache_write": _usage_data.get("cache_creation_input_tokens", 0),
                            "output_tokens": _output_tokens,
                            "ts": _time.time(),
                        }))
                    )
                except Exception:
                    pass

                try:
                    _in = _usage_data.get("input_tokens", 0)
                    _out = _output_tokens
                    _cr = _usage_data.get("cache_read_input_tokens", 0)
                    _cw = _usage_data.get("cache_creation_input_tokens", 0)
                    from api.hot_cache import get_model_pricing as _get_pricing
                    _ri, _rw, _rr, _ro = _get_pricing(_model_name)
                    _est = (_in * _ri + _cr * _rr + _cw * _rw + _out * _ro) / 1_000_000
                    _prev = _session_cost.get(_sid, 0.0)
                    _od_set(_session_cost, _sid, _prev + _est)
                except Exception:
                    pass

            _od_set(_session_last_output_tokens, _sid, _output_tokens)

            if chunks:
                _od_set_dedup(_dedup_cache, _sid, {
                    "hash": _fp,
                    "raw_hash": _fp_raw,
                    "chunks": chunks,
                    "output_tokens": _output_tokens,
                    "ts": _time.time(),
                })
                if not _ifl_future.done():
                    _ifl_future.set_result(chunks)
                _inflight_requests.pop(_ifl_key, None)
                logger.info(
                    f"DEDUP_CACHE: session={_sid} cached {len(chunks)} chunks "
                    f"(out_tok={_output_tokens}, ttl={DEDUP_TTL_SECONDS}s)"
                )
            else:
                if not _ifl_future.done():
                    _ifl_future.set_result(None)
                _inflight_requests.pop(_ifl_key, None)

            if project_id and _output_tokens >= DEDUP_MIN_OUT_TOKENS:
                try:
                    _msgs_copy = list(request_data.messages) if hasattr(request_data, "messages") else []
                    asyncio.get_running_loop().run_in_executor(
                        None, _update_live_context_async, _sid, project_id, _msgs_copy,
                    )
                except Exception:
                    pass

        return StreamingResponse(
            _stream_and_cache(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Project-Id": project_id,
                "X-Session-Id": session_id,
            },
        )

    except ProviderError:
        if "_inflight_key" in dir() and "_inflight_future" in dir():
            if not _inflight_future.done():
                _inflight_future.set_result(None)
            _inflight_requests.pop(_inflight_key, None)
        raise
    except Exception as e:
        if "_inflight_key" in dir() and "_inflight_future" in dir():
            if not _inflight_future.done():
                _inflight_future.set_result(None)
            _inflight_requests.pop(_inflight_key, None)
        logger.error(f"Error: {e!s}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=getattr(e, "status_code", 500),
            detail=get_user_facing_error_message(e),
        ) from e


def _dict_to_sse_response(response_dict: dict):
    """Convert a fast response dict to a proper SSE StreamingResponse."""
    msg_id = response_dict["id"]
    model = response_dict["model"]
    text = response_dict["content"][0]["text"]
    usage_in = response_dict["usage"]["input_tokens"]
    usage_out = response_dict["usage"]["output_tokens"]

    async def sse_gen():
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': usage_in, 'output_tokens': 0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield 'event: ping\ndata: {"type":"ping"}\n\n'
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': usage_out}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest):
    """Count tokens for a request."""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    with logger.contextualize(request_id=request_id):
        try:
            tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            summary = build_request_summary(request_data)
            summary["request_id"] = request_id
            summary["input_tokens"] = tokens
            logger.info("COUNT_TOKENS: {}", json.dumps(summary))
            return TokenCountResponse(input_tokens=tokens)
        except Exception as e:
            logger.error(
                "COUNT_TOKENS_ERROR: request_id={} error={}\n{}",
                request_id,
                get_user_facing_error_message(e),
                traceback.format_exc(),
            )
            raise HTTPException(
                status_code=500, detail=get_user_facing_error_message(e)
            ) from e


@router.post("/memory/save")
async def memory_save(body: dict):
    """Save to memory via REST."""
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not available")
    from memory import _save

    _pid = body.get("project_id", os.getenv("MEMORY_PROJECT_ID", "default"))
    _sid = body.get("session_id", "default")
    row_id = _save(
        _pid,
        _sid,
        body["text"],
        pinned=body.get("pinned", False),
    )

    if row_id and row_id > 0:
        _invalidate_hash_cache(_pid)
        try:
            if _hot_cache:
                _hot_cache.invalidate(_pid)
        except Exception:
            pass
        await asyncio.get_running_loop().run_in_executor(None, _fire_prewarm_after_save, _pid)

    return {"status": "saved", "id": row_id}


@router.post("/memory/search")
async def memory_search(body: dict):
    """Search memory via REST (scoped to project by default)."""
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not available")
    from memory import _search

    hits = _search(
        body.get("query", ""),
        project_id=body.get("project_id", os.getenv("MEMORY_PROJECT_ID", "default")),
        top_k=body.get("top_k", 5),
    )
    return {
        "hits": [
            {
                "text": h.get("text", ""),
                "score": h.get("score", 0),
                "age": h.get("age", ""),
                "tags": h.get("tags", "[]"),
                "pinned": h.get("pinned", False),
                "session": h.get("session_id", ""),
            }
            for h in hits
        ]
    }


@router.delete("/memory/{entry_id}")
async def memory_forget(entry_id: int, body: dict | None = None):
    """Soft-delete a memory entry via REST."""
    if body is None:
        body = {}
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not available")
    from memory import _soft_delete

    pid = body.get("project_id", os.getenv("MEMORY_PROJECT_ID", "default"))
    ok = _soft_delete(entry_id, pid)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Entry {entry_id} not found")
    try:
        from api.hot_cache import hot_cache as _hc
        _hc.invalidate(pid)
    except Exception:
        pass
    _invalidate_hash_cache(pid)
    await asyncio.get_running_loop().run_in_executor(None, _fire_prewarm_after_save, pid)
    return {"status": "deleted", "id": entry_id}


@router.get("/memory/status")
async def memory_status(
    project_id: str = "default",
    session_id: str = "default",
):
    """Memory stats for a given project+session."""
    if not MEMORY_ENABLED:
        return {"enabled": False}
    from memory import _get_stats

    if project_id == "default":
        project_id = os.getenv("MEMORY_PROJECT_ID", "default")
    stats = _get_stats(project_id, session_id)
    return {"enabled": True, **stats}


@router.get("/")
async def root(settings: Settings = Depends(get_settings)):
    from pathlib import Path

    version = (
        Path("VERSION").read_text(encoding="utf-8").strip()
        if Path("VERSION").exists()
        else "dev"
    )
    return {
        "status": "ok",
        "version": version,
        "provider": settings.provider_type,
        "model": settings.model,
        "memory": MEMORY_ENABLED,
        "requests": _total_requests,
        "active_sessions": len(_session_last_msg),
    }


@router.get("/health")
async def health():
    return {"status": "healthy"}


@router.post("/stop")
async def stop_cli(request: Request):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()

    sessions = list(_keepalive_tasks.keys())
    for sid in sessions:
        _stop_keepalive(sid)

    return {
        "status": "stopped",
        "cancelled_count": count,
        "keepalive_stopped": len(sessions),
    }


@router.post("/hook/save")
async def hook_save(body: dict):
    """
    Smart hook save endpoint — hooks POST here to save memories.
    Returns immediately, processes judge + save in background thread
    to avoid blocking Claude Code hooks (which have 2s timeout).
    """
    if not MEMORY_ENABLED:
        return {"saved": False, "reason": "memory_disabled"}

    text = body.get("text", "")
    if not text or not text.strip():
        return {"saved": False, "reason": "empty_text"}

    project_id = body.get("project_id", os.getenv("MEMORY_PROJECT_ID", "default"))
    session_id = body.get("session_id", "default")
    pin = body.get("pin", False)
    trigger = body.get("trigger", "")
    outcome = body.get("outcome", "")
    component = body.get("component", "general")
    is_manual = body.get("is_manual", False)

    _did_save = False

    def _process_hook_save():
        """Background worker for judge + save pipeline. Returns True if saved."""
        nonlocal _did_save
        _pin = pin
        try:
            if not is_manual:
                from memory import _call_judge

                verdict, _source = _call_judge(text)
                if verdict == "SKIP":
                    logger.debug(f"[Hook] Skipped by judge: {text[:60]}")
                    return
                if verdict == "SAVE_PIN":
                    _pin = True

            from memory import _save

            row_id = _save(
                project_id,
                session_id,
                text,
                pinned=_pin,
                source="hook",
                check_dedup=True,
                is_manual=False,
                trigger=trigger,
                outcome=outcome,
                component=component,
            )
            _did_save = isinstance(row_id, int) and row_id > 0
        except Exception as e:
            logger.warning(f"[Hook] Background save failed: {e}")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _process_hook_save)

    if _did_save:
        _invalidate_hash_cache(project_id)
        try:
            if _hot_cache:
                _hot_cache.invalidate(project_id)
        except Exception:
            pass
        await asyncio.get_running_loop().run_in_executor(None, _fire_prewarm_after_save, project_id)

    return {"saved": _did_save, "reason": "completed" if _did_save else "skipped"}


@router.post("/hook/context")
async def hook_context(body: dict):
    """
    Hook context endpoint — returns relevant memories for a query.
    Used by session_start.py hook for CMI injection.
    """
    if not MEMORY_ENABLED:
        return {"context": ""}

    query = body.get("query", "")
    project_id = body.get("project_id", os.getenv("MEMORY_PROJECT_ID", "default"))
    top_k = body.get("top_k", 5)

    if not query:
        return {"context": ""}

    try:
        from memory import _search

        hits = _search(query, project_id=project_id, top_k=top_k)
        if not hits:
            return {
                "context": "<memory_context empty>No relevant context found.</memory_context_end>"
            }

        relevant = [h for h in hits if h.get("score", 0) >= 0.60]
        if not relevant:
            return {
                "context": f"<memory_context empty>No relevant context above threshold "
                f"(best score: {hits[0]['score']:.2f}).</memory_context_end>"
            }

        lines = []
        for h in relevant:
            score = h.get("score", 0)
            age = h.get("age", "")
            pin_tag = " [pinned]" if h.get("pinned") else ""
            lines.append(
                f"[{score:.0%} match{', ' + age if age else ''}] {h['text']}{pin_tag}"
            )

        context_block = "\n".join(lines)
        return {
            "context": f'<memory_context project="{project_id}" entries="{len(relevant)}">\n'
            f"{context_block}\n</memory_context_end>"
        }
    except Exception as e:
        logger.warning(f"[Hook] Context search failed: {e}")
        return {"context": ""}


@router.post("/hooks/PostToolUse")
async def hook_post_tool_use(body: dict):
    """Legacy PostToolUse hook endpoint."""
    if not MEMORY_ENABLED:
        return {"status": "disabled"}
    tool_name = body.get("data", {}).get("tool_name", "")
    tool_output = body.get("data", {}).get("output", "")
    session_id = body.get("data", {}).get("session_id", "default")
    project_id = body.get("data", {}).get(
        "cwd", os.getenv("MEMORY_PROJECT_ID", "default")
    )
    if tool_output and len(tool_output) > 100:
        try:
            from memory import process_tool_output

            process_tool_output(tool_name, tool_output[:500], project_id, session_id)
        except ImportError:
            pass
    return {"status": "ok"}


@router.post("/hooks/SessionStart")
async def hook_session_start(body: dict):
    """Legacy SessionStart hook endpoint."""
    logger.info(f"SESSION_START: {body.get('data', {})}")
    return {"status": "ok"}


@router.post("/hooks/SessionEnd")
async def hook_session_end(body: dict):
    """Legacy SessionEnd hook endpoint."""
    logger.info(f"SESSION_END: {body.get('data', {})}")
    return {"status": "ok"}


@router.get("/api/projects")
async def api_list_projects():
    """List all projects with entry counts."""
    if not MEMORY_ENABLED:
        return {"projects": []}
    from contextlib import closing

    from memory import _get_conn

    with closing(_get_conn()) as conn:
        rows = conn.execute(
            "SELECT project_id, COUNT(*) as count, "
            "SUM(CASE WHEN pinned=1 THEN 1 ELSE 0 END) as pinned, "
            "MAX(created_at) as last_activity "
            "FROM memories WHERE deleted=0 AND superseded=0 "
            "GROUP BY project_id ORDER BY last_activity DESC"
        ).fetchall()
    return {
        "projects": [
            {
                "project_id": r["project_id"],
                "count": r["count"],
                "pinned": r["pinned"],
                "last_activity": r["last_activity"],
            }
            for r in rows
        ]
    }


@router.get("/api/projects/{project_id}/sessions")
async def api_list_sessions(project_id: str):
    """List all sessions for a project."""
    if not MEMORY_ENABLED:
        return {"sessions": []}
    from contextlib import closing

    from memory import _get_conn

    with closing(_get_conn()) as conn:
        rows = conn.execute(
            "SELECT session_id, COUNT(*) as count, "
            "SUM(CASE WHEN pinned=1 THEN 1 ELSE 0 END) as pinned, "
            "MIN(created_at) as started, MAX(created_at) as last_activity "
            "FROM memories WHERE project_id=? AND deleted=0 AND superseded=0 "
            "GROUP BY session_id ORDER BY last_activity DESC",
            (project_id,),
        ).fetchall()
    return {
        "sessions": [
            {
                "session_id": r["session_id"],
                "count": r["count"],
                "pinned": r["pinned"],
                "started": r["started"],
                "last_activity": r["last_activity"],
            }
            for r in rows
        ]
    }


@router.get("/api/projects/{project_id}/entries")
async def api_list_entries(project_id: str, session_id: str = "", limit: int = 0):
    """List entries for a project (optionally filtered by session or limit)."""
    if not MEMORY_ENABLED:
        return {"entries": []}
    from contextlib import closing

    from memory import _age_label, _get_conn

    with closing(_get_conn()) as conn:
        if session_id:
            rows = conn.execute(
                "SELECT id, text, tags, pinned, source, session_id, created_at, token_count, trigger, outcome, component "
                "FROM memories WHERE project_id=? AND session_id=? AND deleted=0 AND superseded=0 "
                "ORDER BY id DESC",
                (project_id, session_id),
            ).fetchall()
        else:
            query = "SELECT id, text, tags, pinned, source, session_id, created_at, token_count, trigger, outcome, component FROM memories WHERE project_id=? AND deleted=0 AND superseded=0 ORDER BY id DESC"
            params = [project_id]
            if limit > 0:
                query += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(query, params).fetchall()
    return {
        "entries": [
            {
                "id": r["id"],
                "text": r["text"],
                "tags": r["tags"],
                "pinned": bool(r["pinned"]),
                "source": r["source"],
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "age": _age_label(r["created_at"]),
                "token_count": r["token_count"],
                "trigger": r["trigger"],
                "outcome": r["outcome"],
                "component": r["component"],
            }
            for r in rows
        ]
    }


@router.post("/api/entries/{entry_id}/pin")
async def api_pin_entry(entry_id: int):
    """Toggle pin on an entry."""
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not available")
    from contextlib import closing

    from memory import _get_conn

    with closing(_get_conn()) as conn:
        row = conn.execute(
            "SELECT pinned FROM memories WHERE id=?", (entry_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Entry not found")
        new_val = 0 if row["pinned"] else 1
        conn.execute("UPDATE memories SET pinned=? WHERE id=?", (new_val, entry_id))
        conn.commit()
    return {"id": entry_id, "pinned": bool(new_val)}


@router.delete("/api/entries/{entry_id}")
async def api_delete_entry(entry_id: int):
    """Soft-delete an entry by ID."""
    if not MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory not available")
    from contextlib import closing

    from memory import _get_conn, _soft_delete

    with closing(_get_conn()) as conn:
        row = conn.execute(
            "SELECT project_id FROM memories WHERE id=?", (entry_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Entry not found")
    ok = _soft_delete(entry_id, row["project_id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Delete failed")
    _del_pid = row["project_id"]
    try:
        from api.hot_cache import hot_cache as _hc
        _hc.invalidate(_del_pid)
    except Exception:
        pass
    _invalidate_hash_cache(_del_pid)
    await asyncio.get_running_loop().run_in_executor(None, _fire_prewarm_after_save, _del_pid)
    return {"status": "deleted", "id": entry_id}


@router.get("/api/stats/{project_id}")
async def api_project_stats(project_id: str):
    """Full stats for the dashboard overview."""
    if not MEMORY_ENABLED:
        return {}
    from contextlib import closing

    from memory import _get_conn

    with closing(_get_conn()) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE project_id=? AND deleted=0 AND superseded=0",
            (project_id,),
        ).fetchone()[0]
        pinned = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE project_id=? AND deleted=0 AND pinned=1",
            (project_id,),
        ).fetchone()[0]
        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM memories WHERE project_id=? AND deleted=0 AND superseded=0",
            (project_id,),
        ).fetchone()[0]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM memories WHERE project_id=? AND deleted=0",
            (project_id,),
        ).fetchone()[0]
        sources = conn.execute(
            "SELECT source, COUNT(*) as c FROM memories WHERE project_id=? AND deleted=0 AND superseded=0 GROUP BY source",
            (project_id,),
        ).fetchall()
        recent = conn.execute(
            "SELECT created_at FROM memories WHERE project_id=? AND deleted=0 AND superseded=0 ORDER BY id DESC LIMIT 1",
            (project_id,),
        ).fetchone()

    return {
        "total": total,
        "pinned": pinned,
        "total_tokens": total_tokens,
        "sessions": sessions,
        "sources": {r["source"]: r["c"] for r in sources},
        "last_save": recent["created_at"] if recent else None,
        "last_injection": _get_last_injection(project_id),
        "memory_model": os.getenv("MEMORY_MODEL", "") or os.getenv("MODEL", "N/A"),
    }


def _get_last_injection(project_id: str) -> dict | None:
    """Get last injection stats for this project from in-memory tracker."""
    try:
        from memory import _injection_history

        for inj in _injection_history:
            if inj.get("project_id") == project_id:
                return inj
    except ImportError:
        pass
    return None


@router.get("/api/injections/{project_id}")
async def api_injection_history(project_id: str):
    """Injection history for the dashboard — per-prompt token tracking."""
    try:
        from memory import _injection_history

        filtered = [i for i in _injection_history if i.get("project_id") == project_id]
        return {"injections": filtered[:20]}
    except ImportError:
        return {"injections": []}


@router.get("/api/events")
async def api_get_events():
    """Get recent LLM activity and injection events."""
    try:
        from memory import _llm_events

        return {"events": _llm_events[:20]}
    except ImportError:
        return {"events": []}


@router.get("/api/llm-trace")
async def api_llm_trace(limit: int = 100):
    """Full LLM workflow trace — all calls, retries, fallbacks and stats.

    Used by the dashboard LLM Trace tab to visualize the Cerebras workflow.
    """
    try:
        from memory import get_llm_trace
        return get_llm_trace(limit=min(limit, 200))
    except ImportError:
        return {"events": [], "stats": {}, "by_task": {}}


_sse_clients: list[asyncio.Queue] = []
_sse_clients_lock = asyncio.Lock()


async def broadcast_sse_event(event: dict) -> None:
    """Push an event to all connected SSE clients."""
    async with _sse_clients_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


@router.get("/api/events/stream")
async def api_events_stream():
    """Server-Sent Events stream for real-time dashboard updates."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)

    async with _sse_clients_lock:
        _sse_clients.append(queue)

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield ": connected\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            async with _sse_clients_lock:
                if queue in _sse_clients:
                    _sse_clients.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.get("/api/cache-stats/{project_id}")
async def api_cache_stats(project_id: str):
    """Cache hit rate + cost savings for the dashboard analytics tab."""
    try:
        from api.hot_cache import hit_tracker
        return hit_tracker.get_stats(project_id)
    except ImportError:
        return {"cache_hits": 0, "cache_misses": 0, "cache_hit_rate": 0.0, "cost_usd": 0.0, "saved_usd": 0.0}


@router.get("/api/notifications")
async def api_notifications(project: str = "", limit: int = 20):
    """Real-time notifications for the dashboard."""
    try:
        from api.hot_cache import notifications
        return {"events": notifications.get_recent(project_id=project, limit=limit)}
    except ImportError:
        return {"events": []}


@router.get("/api/graph/{project_id}")
async def api_graph_info(project_id: str):
    """Graph search stats — edges, components, partition sizes."""
    if not MEMORY_ENABLED:
        return {}
    from contextlib import closing

    from memory import _get_conn
    try:
        with closing(_get_conn()) as conn:
            components = conn.execute(
                "SELECT COALESCE(component,'general') as c, COUNT(*) as n "
                "FROM memories WHERE project_id=? AND deleted=0 GROUP BY c ORDER BY n DESC",
                (project_id,),
            ).fetchall()
            try:
                edges = conn.execute(
                    "SELECT COUNT(*) FROM memory_edges WHERE project_id=?",
                    (project_id,),
                ).fetchone()[0]
            except Exception:
                edges = 0
        return {
            "components": [{"name": r[0], "count": r[1]} for r in components],
            "total_edges": edges,
        }
    except Exception as e:
        return {"error": str(e)}


_last_llm_health: dict = {"ok": True, "ms": 0, "ts": 0.0}

@router.get("/api/health")
async def health_check():
    """Health status for dashboard dot indicator."""
    import time as _t

    db_ok = True
    db_count = 0
    try:
        from memory import _get_conn
        from contextlib import closing
        with closing(_get_conn()) as conn:
            db_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE deleted=0 AND superseded=0"
            ).fetchone()[0]
    except Exception as e:
        db_ok = False
        logger.warning(f"[Health] DB check failed: {e}")

    chroma_ok = True
    try:
        from memory import _chroma_ok
        chroma_ok = _chroma_ok()
    except Exception:
        chroma_ok = False

    llm_provider = os.getenv("MEMORY_MODEL", "").split("/")[0] if "/" in os.getenv("MEMORY_MODEL", "") else "unknown"
    now = _t.time()
    if now - _last_llm_health["ts"] > 60.0:
        try:
            t0 = _t.time()
            _loop = asyncio.get_running_loop()
            await _loop.run_in_executor(
                None, _memory_call_llm,
                "ping", "1", 1, 3.0, "health_ping",
            )
            _last_llm_health["ok"] = True
            _last_llm_health["ms"] = int((_t.time() - t0) * 1000)
        except Exception:
            _last_llm_health["ok"] = False
            _last_llm_health["ms"] = 0
        _last_llm_health["ts"] = now

    llm_ok = _last_llm_health["ok"]
    llm_ms = _last_llm_health["ms"]

    status = "ok" if (db_ok and llm_ok) else ("degraded" if db_ok else "error")
    return {
        "status": status,
        "db_ok": db_ok,
        "db_entries": db_count,
        "chroma_ok": chroma_ok,
        "llm_ok": llm_ok,
        "llm_provider": llm_provider,
        "llm_ms": llm_ms,
        "groq_ok": llm_ok,
        "groq_ms": llm_ms,
        "total_requests": _total_requests,
        "active_sessions": len(_session_last_msg),
    }


@router.get("/api/live")
async def live_stats():
    """Per-session live stats. Poll every 2 seconds from dashboard."""
    try:
        from api.hot_cache import hit_tracker
        sessions = []
        all_stats = hit_tracker.get_all_stats()
        for project_id, state in all_stats.items():
            sessions.append({
                "project_id": project_id,
                "turns": state.get("api_calls", 0),
                "cost_usd": round(state.get("cost_usd", 0.0), 6),
                "cache_read_tokens": state.get("cache_read_tokens", 0),
                "cache_write_tokens": state.get("cache_write_tokens", 0),
            })
        return {"sessions": sessions, "active": len(sessions)}
    except Exception as e:
        return {"sessions": [], "active": 0, "error": str(e)}


@router.get("/api/cost/summary")
async def cost_summary():
    """Today + week cost breakdown by model."""
    try:
        from api.hot_cache import hit_tracker
        all_stats = hit_tracker.get_all_stats()
        total_cost = sum(s.get("cost_usd", 0) for s in all_stats.values())
        total_saved = sum(s.get("saved_usd", 0) for s in all_stats.values())
        return {
            "total_cost_usd": round(total_cost, 6),
            "total_saved_usd": round(total_saved, 6),
            "projects": {pid: {"cost": round(s.get("cost_usd", 0), 6), "saved": round(s.get("saved_usd", 0), 6)} for pid, s in all_stats.items()},
        }
    except Exception as e:
        return {"error": str(e), "total_cost_usd": 0, "total_saved_usd": 0}


@router.get("/api/last-injection/{session_id}")
async def last_injection(session_id: str):
    """Last memory injection event for a session — for dashboard injection panel."""
    return _last_injection_data.get(session_id, {})
