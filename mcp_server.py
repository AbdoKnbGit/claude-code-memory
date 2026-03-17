import json
import os
import re
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from mcp.server.fastmcp import FastMCP

_DOCKER_APP_PREFIX = "//app"

sys.path.insert(0, str(Path(__file__).parent.absolute()))

from memory import (
    LLM_RERANK_CANDIDATES,
    LLM_RERANK_ENABLED,
    MEMORY_TOKEN_BUDGET,
    _encode,
    _get_client,
    _get_conn,
    _get_stats,
    _llm_select_and_build,
    _save,
    _search,
    _soft_delete,
)

_PIN_KEYWORDS = {"important", "always", "never", "rule", "convention", "must", "critical"}

def _should_pin_local(text: str) -> bool:
    """Heuristic: pin if text contains strong directive keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _PIN_KEYWORDS)

def _summarize_before_save(text: str) -> str | None:
    """Pass-through: return text as-is (no LLM summarization in MCP path)."""
    return text if len(text) < 2000 else text[:2000]

mcp = FastMCP("cc-memory", dependencies=["mcp", "loguru", "chromadb", "sentence-transformers", "langdetect", "anthropic"])

_SESSION_START_DT = datetime.now()
_SESSION_ID = _SESSION_START_DT.strftime("%Y%m%d_%H%M%S")

def _human_session(session_id: str) -> str:
    """Convert timestamp session ID to human readable relative time."""
    try:
        dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        now = datetime.now()
        diff = now - dt
        if diff.days == 0:
            return f"today {dt.strftime('%H:%M')}"
        elif diff.days == 1:
            return f"yesterday {dt.strftime('%H:%M')}"
        else:
            return f"{diff.days} days ago {dt.strftime('%H:%M')}"
    except:
        return session_id

def _get_project_id() -> str:
    """
    Detect project ID from environment.
    Priority:
    1. CC_MEMORY_PROJECT env var (set via .mcp.json) — ONLY reliable source
    2. CC_PROJECT_PATH env var -> extract folder name (if valid)
    3. "uninitialized" — instructs user to call memory_init()
    """
    _INVALID_IDS = {
        "app", "src", "default", "work", "home", "tmp", "temp",
        "root", "user", "users", "opt", "var", "etc", "lib", "bin",
        "workspace", "code", "dev", "local", "",
    }

    val = os.getenv("CC_MEMORY_PROJECT", "").strip()
    if val and val.lower() not in _INVALID_IDS:
        return val.lower().replace(" ", "_")

    path = os.getenv("CC_PROJECT_PATH", "").strip()
    if path:
        name = os.path.basename(path.rstrip("/\\")).lower().replace(" ", "_")
        if name and name not in _INVALID_IDS:
            return name

    logger.warning(
        "[Project] CC_MEMORY_PROJECT not set — project='uninitialized'. "
        "Run memory_init(project_path='<cwd>') to fix."
    )
    return "uninitialized"

_PROJECT_ID = _get_project_id()

def get_defaults(project_id: str = "", session_id: str = ""):
    pid = project_id or _get_project_id()
    sid = session_id or _SESSION_ID
    return pid, sid

def _respond(text: str) -> str:
    """Guarantee non-empty string response from any tool."""
    if not text or not str(text).strip():
        return "Error: Tool returned empty result."
    return str(text).strip()


def _invalidate_cache(project_id: str) -> None:
    """Invalidate hot cache after any DB mutation so stale blocks aren't served."""
    try:
        from api.hot_cache import hot_cache as _hc
        _hc.invalidate(project_id)
    except Exception:
        pass


def _chroma_delete_ids(project_id: str, ids: list[int]) -> int:
    """Delete entries from ChromaDB by id list. Returns count deleted."""
    if not ids or not _chroma_ok():
        return 0
    deleted = 0
    try:
        from memory import _get_collection
        coll = _get_collection(project_id)
        str_ids = [str(i) for i in ids]
        coll.delete(ids=str_ids)
        deleted = len(str_ids)
    except Exception as exc:
        logger.warning(f"[MCP] ChromaDB bulk delete failed for project={project_id}: {exc}")
    return deleted


def _chroma_ok() -> bool:
    """Check if ChromaDB is available."""
    try:
        _get_client()
        return True
    except Exception:
        return False


def _relative_age(created_at: str) -> str:
    """Convert timestamp to human-readable age: '2h ago', 'yesterday', etc."""
    try:
        if isinstance(created_at, str):
            dt = datetime.fromisoformat(created_at)
        else:
            return ""
        now = datetime.now(UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        diff = (now - dt).total_seconds()
        if diff < 3600:
            return f"{int(diff/60)}min ago"
        if diff < 86400:
            return f"{int(diff/3600)}h ago"
        if diff < 172800:
            return "yesterday"
        return f"{int(diff/86400)}d ago"
    except Exception:
        return ""


def _search_with_scores(query: str, project_id: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search returning entries with cosine similarity scores.
    Reuses existing _search() from memory module.
    """
    hits = _search(query, project_id=project_id, session_id=_SESSION_ID, top_k=top_k)
    for h in hits:
        if "age" not in h or not h["age"]:
            h["age"] = ""
    return hits


def _get_pinned(project_id: str) -> list[dict]:
    """Fetch pinned entries for a project."""
    with closing(_get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, text, tags FROM memories "
            "WHERE project_id=? AND deleted=0 AND pinned=1 AND superseded=0",
            (project_id,),
        ).fetchall()
    return [{"id": r["id"], "text": r["text"], "tags": r["tags"]} for r in rows]


@mcp.tool()
def memory_save(text: str = "", project_id: str = "", session_id: str = "", pin: bool = False, pinned: bool = False) -> str:
    """Save information to long-term memory. EXPLICITLY pass project_id."""
    try:
        pid, sid = get_defaults(project_id, session_id)
        if pid == "uninitialized":
            return _respond("Error: Project not identified.")

        if len(pid) > 40 or pid.count("_") > 4:
            return _respond(
                f"project_id '{pid}' looks invalid. "
                f"Use the folder name only (e.g. 'test_mcp_init')."
            )

        if not text:
            return _respond("Error: 'text' field is required for memory_save.")

        pin_override = bool(pin) or bool(pinned) or _should_pin_local(text)
        logger.info(f"[Pin] local_detect={_should_pin_local(text)} manual_pin={bool(pin) or bool(pinned)} final={pin_override}")

        text_to_save = text
        try:
            summarized = _summarize_before_save(text)
            if summarized and len(summarized) > 10:
                text_to_save = summarized
                logger.info(f"[MCP] summarized: '{text[:40]}' → '{text_to_save[:40]}'")
        except Exception as e:
            logger.warning(f"[MCP] summarize failed, using original: {e}")

        result = _save(pid, sid, text_to_save, pinned=pin_override, source="user", check_dedup=True, is_manual=True)

        if isinstance(result, dict):
            if result.get("error") == "duplicate":
                return _respond(f"Already in memory (id={result['id']}, similarity={result['similarity']:.2f})")

            tags_str = ", ".join(result.get("tags", []))
            pin_msg = " [pinned]" if result.get("pinned") else ""
            msg = f"Saved{pin_msg} as id {result['id']}"
            if tags_str:
                msg += f", tagged: {tags_str}"
            if result.get("contradiction"):
                c = result["contradiction"]
                msg += f". This replaces id {c['superseded_id']} (\"{c['old_text'][:60]}\")"
            return _respond(msg)

        return _respond("Error during save.")
    except Exception as exc:
        logger.error(f"[memory_save] {exc}")
        return _respond(f"Error: {exc}")


def _fmt_memories(entries: list, query: str = "") -> str:
    """Format memory entries as clean human-readable text."""
    if not entries:
        return f"No memories found{f' for: {query}' if query else ''}."

    lines = [f"{len(entries)} memories{f' matching \"{query}\"' if query else ''}:\n"]
    for i, e in enumerate(entries, 1):
        score = float(e.get("score") or 0)
        pin = " [pinned]" if e.get("pinned") else ""
        eid = e.get("id", "?")
        pct = f"{score:.0%}" if score else "—"
        lines.append(f"  {i}. (id {eid}, {pct} match{pin})")
        lines.append(f"     {(e.get('text') or '')[:200]}")
        tags = e.get("tags") or []
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        if tags:
            lines.append(f"     tags: {', '.join(tags[:5])}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def memory_search(query: str = "recent project context", project_id: str = "", session_id: str = "", top_k: int = 5) -> str:
    """Search memory using semantic search. EXPLICITLY pass project_id."""
    try:
        pid, sid = get_defaults(project_id, session_id)
        if pid == "uninitialized":
            return _respond("Error: Project not identified.")
        if not query:
            query = "recent project context"
        hits = _search(query, project_id=pid, session_id=sid, top_k=top_k)

        if not hits:
            return _respond("No results found.")

        return _respond(_fmt_memories(hits, query))
    except Exception as exc:
        logger.error(f"[memory_search] {exc}")
        return _respond(f"Search error: {exc}")


@mcp.tool()
def memory_context(query: str, project_id: str = "", session_id: str = "") -> str:
    """Smart context injection — returns ONLY memories relevant to this query. EXPLICITLY pass project_id."""
    try:
        pid, sid = get_defaults(project_id=project_id, session_id=session_id)
        if pid == "uninitialized":
            return _respond("Error: Project not identified.")

        if not query or not query.strip():
            return _respond("No query provided — nothing to inject.")

        if LLM_RERANK_ENABLED:
            num_candidates = LLM_RERANK_CANDIDATES
            hits = _search_with_scores(query, pid, top_k=num_candidates)
            if not hits:
                return _respond("<memory_context empty>No relevant context found.</memory_context_end>")

            block = _llm_select_and_build(query, hits, project_id=pid, session_id=sid)
            if block:
                return _respond(block)

        results = _search_with_scores(query, pid, top_k=5)

        if not results:
            return _respond("<memory_context empty>No relevant context found.</memory_context_end>")

        relevant = [r for r in results if r["score"] >= 0.60]

        if not relevant:
            return _respond(
                f"<memory_context empty>No relevant context above threshold "
                f"(best score: {results[0]['score']:.2f}).</memory_context_end>"
            )

        pinned = _get_pinned(pid)
        relevant_ids = {r["id"] for r in relevant}
        extra_pinned = [p for p in pinned if p["id"] not in relevant_ids]

        all_entries = extra_pinned + relevant
        tokens_used = sum(len(e["text"].split()) * 1.3 for e in all_entries)

        lines = []
        for e in extra_pinned:
            lines.append(f"[PINNED] {e['text']}")
        for e in relevant:
            age = e.get("age", "")
            score = e["score"]
            lines.append(f"[{score:.0%} match{', ' + age if age else ''}] {e['text']}")

        context_block = "\n".join(lines)

        return _respond(
            f"<memory_context project=\"{pid}\" "
            f"entries=\"{len(all_entries)}\" "
            f"tokens=\"~{int(tokens_used)}\" "
            f"query=\"{query[:50]}\">\n"
            f"{context_block}\n"
            f"</memory_context_end>"
        )
    except Exception as exc:
        logger.error(f"[memory_context] {exc}")
        return _respond(f"Context error: {exc}")


@mcp.tool()
def memory_remember(query: str = "recent decisions and context", project_id: str = "", session_id: str = "") -> str:
    """Inject relevant context into the current session. EXPLICITLY pass project_id."""
    try:
        pid, sid = get_defaults(project_id, session_id)
        if pid == "uninitialized":
            return _respond("Error: Project not identified.")
        if not query:
            query = "recent decisions and context"
        if LLM_RERANK_ENABLED:
            num_candidates = LLM_RERANK_CANDIDATES
            hits = _search(query, project_id=pid, session_id=sid, top_k=num_candidates)
            if not hits:
                return _respond("No relevant context found.")

            block_text = _llm_select_and_build(query, hits, project_id=pid, session_id=sid)
            if block_text:
                return _respond(f"Found relevant context:\n\n{block_text}")

        hits = _search(query, project_id=pid, session_id=sid, top_k=10)
        if not hits:
            return _respond("No relevant context found.")
        lines = []
        for h in hits:
            tags = h.get("tags", "[]")
            tags_list = json.loads(tags) if isinstance(tags, str) else tags
            tags_str = ("#" + " #".join(tags_list)) if tags_list else ""
            pin = " [pinned]" if h.get("pinned") else ""
            age = h.get("age", "")
            lines.append(f"[{h['score']:.2f}] {age} {tags_str} (id={h['id']}): {h['text']}{pin}")
        block = "\n".join(lines)
        return _respond(f"Found relevant context:\n\n{block}")
    except Exception as exc:
        logger.error(f"[memory_remember] {exc}")
        return _respond(f"Remember error: {exc}")

def _resolve_id(**kwargs) -> int:
    """
    General-purpose ID resolver. Accepts any parameter name containing 'id'
    (id, entry_id, memory_id, etc.) and returns the integer value.
    """
    if kwargs.get("id"):
        return int(kwargs["id"])
    for key, val in kwargs.items():
        if "id" in key.lower() and val:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return 0


@mcp.tool()
def memory_manage(action: str, id: int = 0, entry_id: int = 0, project_id: str = "") -> str:
    """Forget or unpin a memory entry. Provide 'forget' or 'unpin' action and entry id."""
    try:
        resolved = _resolve_id(id=id, entry_id=entry_id)
        if not resolved:
            return _respond("Error: 'id' is required.")
        pid, _ = get_defaults(project_id)

        if action == "forget":
            ok = _soft_delete(resolved, pid)
            if ok:
                _invalidate_cache(pid)
                return _respond(f"Forgotten entry {resolved}.")
            return _respond(f"Entry {resolved} not found.")
        elif action == "unpin":
            with closing(_get_conn()) as conn:
                cur = conn.execute("UPDATE memories SET pinned=0 WHERE id=? AND project_id=?", (resolved, pid))
                affected = cur.rowcount
                conn.commit()
            if affected > 0:
                _invalidate_cache(pid)
                return _respond(f"Unpinned entry {resolved}.")
            return _respond(f"Entry {resolved} not found.")
        elif action == "pin":
            with closing(_get_conn()) as conn:
                cur = conn.execute("UPDATE memories SET pinned=1 WHERE id=? AND project_id=?", (resolved, pid))
                affected = cur.rowcount
                conn.commit()
            if affected > 0:
                _invalidate_cache(pid)
                return _respond(f"Pinned entry {resolved}.")
            return _respond(f"Entry {resolved} not found.")

        return _respond("Invalid action. Use 'forget', 'unpin', or 'pin'.")
    except Exception as exc:
        logger.error(f"[memory_manage] {exc}")
        return _respond(f"Manage error: {exc}")


@mcp.tool()
def memory_status(project_id: str = "", session_id: str = "") -> str:
    """Get statistics about memory usage. EXPLICITLY pass project_id."""
    try:
        pid, sid = get_defaults(project_id, session_id)
        if pid == "uninitialized":
            return _respond("Error: Project not identified.")
        s = _get_stats(pid, sid)

        if not s:
            return _respond(f"Memory for {pid}: no data yet.")

        history_lines = []
        sessions = s.get("session_history", [])
        real_msg_counts = s.get("real_msg_counts", {})
        if sessions:
            for sid_item, count, _ in sessions:
                label = _human_session(sid_item)
                is_curr = " (current)" if sid_item == sid else ""
                real_msgs = real_msg_counts.get(sid_item, 0)
                history_lines.append(f"    {label}{is_curr}, {real_msgs} messages, {count} saves")

        history_str = "\n".join(history_lines) if history_lines else "    No past sessions."
        active = s.get('active', 0)
        pinned = s.get('pinned', 0)
        tokens = s.get('tokens_injected', 0)
        budget = s.get('token_budget', MEMORY_TOKEN_BUDGET)
        chroma = s.get('chroma_size', '0')

        return _respond(
            f"Memory status for project \"{pid}\"\n\n"
            f"Current session: {_human_session(sid)}\n"
            f"Sessions:\n{history_str}\n\n"
            f"{active} active entries ({pinned} pinned), "
            f"~{tokens}/{budget} tokens used, "
            f"{chroma} vectors indexed."
        )
    except Exception as exc:
        logger.error(f"[memory_status] {exc}")
        return _respond(f"Status error: {exc}")


@mcp.tool()
def memory_export(project_id: str = "") -> str:
    """Export all memory entries for a project. Returns the full JSON content so you can write it to a file in the project directory."""
    try:
        pid, _ = get_defaults(project_id)
        with closing(_get_conn()) as conn:
            rows = conn.execute(
                "SELECT id, project_id, session_id, text, tags, pinned, superseded, source, created_at, token_count FROM memories "
                "WHERE project_id=? AND deleted=0 ORDER BY created_at ASC",
                (pid,),
            ).fetchall()

        if not rows:
            return _respond("No entries to export.")

        entries = [dict(r) for r in rows]
        json_content = json.dumps(entries, indent=2, ensure_ascii=False, default=str)

        try:
            export_dir = Path(_DOCKER_APP_PREFIX.lstrip("/")) / "data" / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            (export_dir / filename).write_text(json_content, encoding="utf-8")
        except Exception:
            pass

        suggested_name = f"memory_export_{pid}.json"
        return _respond(
            f"Exported {len(entries)} entries for project \"{pid}\".\n"
            f"Write this to a file named \"{suggested_name}\" in the project directory:\n\n"
            f"{json_content}"
        )
    except Exception as exc:
        logger.error(f"[memory_export] {exc}")
        return _respond(f"Export error: {exc}")


@mcp.tool()
def memory_clear(scope: str = "session", target: str = "", project_id: str = "", session_id: str = "", confirm: bool = False) -> str:
    """Clear memory for a session or project. EXPLICITLY pass project_id."""
    try:
        pid, sid = get_defaults(project_id, session_id)
        effective_scope = (target or scope).strip().lower()

        if effective_scope == "session":
            with closing(_get_conn()) as conn:
                id_rows = conn.execute(
                    "SELECT id FROM memories WHERE project_id=? AND session_id=? AND deleted=0",
                    (pid, sid),
                ).fetchall()
                ids_to_delete = [r["id"] for r in id_rows]
                cur = conn.execute("UPDATE memories SET deleted=1 WHERE project_id=? AND session_id=? AND deleted=0", (pid, sid))
                count = cur.rowcount
                conn.commit()
            if ids_to_delete:
                _chroma_delete_ids(pid, ids_to_delete)
                _invalidate_cache(pid)
            return _respond(f"Session cleared ({count} entries removed from DB and search index).")

        elif effective_scope == "project":
            if not confirm:
                return _respond("Are you sure? Use 'confirm=True' to clear the entire project memory.")

            with closing(_get_conn()) as conn:
                id_rows = conn.execute(
                    "SELECT id FROM memories WHERE project_id=? AND deleted=0",
                    (pid,),
                ).fetchall()
                ids_to_delete = [r["id"] for r in id_rows]
                cur = conn.execute("UPDATE memories SET deleted=1 WHERE project_id=? AND deleted=0", (pid,))
                count = cur.rowcount
                conn.commit()
            if ids_to_delete:
                _chroma_delete_ids(pid, ids_to_delete)
                _invalidate_cache(pid)

            return _respond(f"Project cleared ({count} entries removed from DB and search index).")

        return _respond("Invalid scope. Use 'session' or 'project'.")
    except Exception as exc:
        logger.error(f"[memory_clear] {exc}")
        return _respond(f"Clear error: {exc}")


@mcp.tool()
def memory_reindex(project_id: str = "") -> str:
    """Rebuild ChromaDB embeddings from SQLite. EXPLICITLY pass project_id."""
    try:
        pid, _ = get_defaults(project_id)

        with closing(_get_conn()) as conn:
            rows = conn.execute(
                "SELECT id, text, session_id, pinned, created_at, token_count, tags, source "
                "FROM memories WHERE project_id=? AND deleted=0 AND superseded=0",
                (pid,),
            ).fetchall()

        if not rows:
            return _respond("No entries to reindex.")

        safe_pid = re.sub(r"[^a-zA-Z0-9]", "_", pid).strip("_") or "default"
        coll_name = f"memories_{safe_pid}"[:63]
        try:
            _get_client().delete_collection(coll_name)
        except Exception:
            pass
        coll = _get_client().create_collection(coll_name, metadata={"hnsw:space": "cosine"})

        indexed = 0
        for row in rows:
            try:
                emb = _encode(row["text"])
                now_str = row["created_at"] or datetime.now(UTC).isoformat()
                coll.add(
                    ids=[str(row["id"])],
                    embeddings=[emb],
                    metadatas=[{
                        "id": row["id"],
                        "project_id": pid,
                        "session_id": row["session_id"],
                        "pinned": row["pinned"],
                        "created_at": now_str,
                        "token_count": row["token_count"],
                        "tags": row["tags"] or "[]",
                        "source": row["source"] or "user",
                    }],
                    documents=[row["text"]],
                )
                indexed += 1
            except Exception as exc:
                logger.warning(f"[Reindex] Failed id={row['id']}: {exc}")

        return _respond(
            f"Reindexed {indexed}/{len(rows)} entries for project '{pid}'.\n"
            f"ChromaDB collection '{coll_name}' rebuilt with fresh embeddings."
        )
    except Exception as exc:
        logger.error(f"[memory_reindex] {exc}")
        return _respond(f"Reindex error: {exc}")


@mcp.tool()
def memory_init(project_path: str = "", path: str = "") -> str:
    """Initialize memory for a new project. Provide project_path."""
    try:
        import json as _json
        import os

        project_path = (project_path or path).strip()
        if not project_path:
            return _respond("Error: 'project_path' is required.")

        path_normalized = project_path.replace("\\", "/")
        folder = path_normalized.rstrip("/").split("/")[-1]
        project_name = re.sub(r"[^a-z0-9_-]", "_", folder.lower()).strip("_") or "default"
        if not project_name:
            return _respond("Error: could not extract project name from path.")

        mcp_config = {
            "mcpServers": {
                "cc-memory": {
                    "type": "stdio",
                    "command": "docker",
                    "args": [
                        "exec", "-i",
                        "-e", f"CC_MEMORY_PROJECT={project_name}",
                        "cc-nim-memory", "python", f"{_DOCKER_APP_PREFIX}/mcp_server.py"
                    ]
                }
            }
        }
        mcp_json_path = os.path.join(project_path.replace("/", os.sep), ".mcp.json")

        hooks_config = {
            "hooks": {
                "PostToolUse": [
                    {
                        "matcher": "Write|Edit|MultiEdit|Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"docker exec -i -e CC_MEMORY_PROJECT={project_name} cc-nim-memory python {_DOCKER_APP_PREFIX}/hooks/post_tool_use.py"
                            }
                        ]
                    }
                ],
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"docker exec -i -e CC_MEMORY_PROJECT={project_name} cc-nim-memory python {_DOCKER_APP_PREFIX}/hooks/user_prompt.py"
                            }
                        ]
                    }
                ]
            }
        }
        settings_json_path = os.path.join(project_path.replace("/", os.sep), ".claude", "settings.json")

        return _respond(
            f"Write these two files to initialize project \"{project_name}\".\n\n"
            f"File 1: {mcp_json_path}\n"
            f"{_json.dumps(mcp_config, indent=2)}\n\n"
            f"File 2: {settings_json_path}\n"
            f"{_json.dumps(hooks_config, indent=2)}\n\n"
            f"The MCP server runs inside Docker and cannot write to the host filesystem, "
            f"so you need to create these files yourself. "
            f"Once written, tell the user to restart Claude Code."
        )
    except Exception as exc:
        logger.error(f"[memory_init] {exc}")
        return _respond(f"Init error: {exc}")
@mcp.tool()
def memory_reduce(project_id: str = "") -> str:
    """Reduce memory by removing oldest non-pinned entries. Call when over 350 tokens or 10 entries."""
    pid, _ = get_defaults(project_id)
    if pid == "uninitialized" or not pid:
        return _respond("Project not identified.")

    try:
        deleted_ids = []
        remaining_entries = 0
        remaining_tokens = 0

        with closing(_get_conn()) as conn:
            rows = conn.execute(
                "SELECT id, text, pinned, LENGTH(text) as length FROM memories "
                "WHERE project_id=? AND deleted=0 AND superseded=0 ORDER BY created_at ASC",
                (pid,)
            ).fetchall()

            total_tokens = sum(r["length"] for r in rows) // 3
            total_entries = len(rows)

            if total_tokens <= 350 and total_entries <= 10:
                return _respond(f"No reduction needed, {total_entries} entries at ~{total_tokens} tokens.")

            for row in rows:
                if total_tokens <= 300 and total_entries <= 8:
                    break
                entry_id = row["id"]
                if row["pinned"]:
                    continue
                conn.execute("UPDATE memories SET deleted=1 WHERE id=?", (entry_id,))
                deleted_ids.append(entry_id)
                total_tokens -= row["length"] // 3
                total_entries -= 1

            conn.commit()
            remaining_entries = total_entries
            remaining_tokens = total_tokens

        if not deleted_ids:
            return _respond("All entries are pinned. Unpin some to allow reduction.")

        _chroma_delete_ids(pid, deleted_ids)

        _invalidate_cache(pid)

        logger.info(
            f"[memory_reduce] project={pid} removed={len(deleted_ids)} "
            f"ids={deleted_ids} remaining={remaining_entries} tokens=~{remaining_tokens}"
        )

        return _respond(
            f"Reduced: removed {len(deleted_ids)} entries (ids: {deleted_ids}). "
            f"Now {remaining_entries} entries, ~{remaining_tokens} tokens."
        )
    except Exception as exc:
        logger.error(f"[memory_reduce] {exc}")
        return _respond(f"Reduce error: {exc}")

@mcp.tool()
def memory_suggest(context: str = "", project_id: str = "") -> str:
    """Suggest that the current context is worth saving. EXPLICITLY pass project_id."""
    pid, _ = get_defaults(project_id)
    if pid == "uninitialized" or not pid:
        return _respond("Project not identified.")

    return _respond(
        f"Suggested for memory: \"{context}\"\n\n"
        f"Reply with \"store\" to confirm, or ignore to skip."
    )

if __name__ == "__main__":
    import uvicorn
    if _PROJECT_ID == "uninitialized":
        logger.warning("[Startup] CC_MEMORY_PROJECT not set — new project detected")
    else:
        logger.info(f"[Startup] project={_PROJECT_ID} session={_SESSION_ID}")

    if "--http" in sys.argv:
        port = 8083
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        logger.info(f"[MCP] Starting HTTP/SSE server on port {port}")
        uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
    else:
        mcp.run()
