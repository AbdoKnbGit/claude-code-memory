"""
Shared utilities for cc-memory hooks.
Reads CC_MEMORY_PROJECT from env — same as mcp_server.py.

OPTIMIZED: save() is fire-and-forget — spawns background thread
so hooks return instantly to Claude Code.
"""

import json
import os
import re
import sys
import threading
import urllib.request

MEMORY_API = os.getenv("MEMORY_HOOK_API", "http://localhost:8082")
PROJECT_ID_ENV = os.getenv("CC_MEMORY_PROJECT", "")
SESSION_ID_ENV = os.getenv("CC_MEMORY_SESSION", "")


def get_project_context(data: dict) -> tuple[str, str]:
    """Extract project_id and session_id from hook data or env."""
    _INVALID_IDS = {
        "app", "src", "default", "work", "home", "tmp", "temp",
        "root", "user", "users", "opt", "var", "etc", "lib", "bin",
        "workspace", "code", "dev", "local", "",
    }

    pid = PROJECT_ID_ENV
    sid = SESSION_ID_ENV

    if pid and pid.lower() in _INVALID_IDS:
        pid = ""

    cwd = data.get("cwd")
    if not pid and cwd:
        path_normalized = cwd.replace("\\", "/")
        folder = path_normalized.rstrip("/").split("/")[-1]
        candidate = re.sub(r"[^a-z0-9_-]", "_", folder.lower()).strip("_") or ""
        if candidate and candidate not in _INVALID_IDS:
            pid = candidate

    return pid, sid


def _do_save(payload_bytes: bytes) -> None:
    """Background worker — sends save request to API. Silent on failure."""
    try:
        req = urllib.request.Request(
            f"{MEMORY_API}/hook/save",
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read().decode())

        try:
            log_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "hooks.log")
            with open(log_path, "a", encoding="utf-8") as f:
                saved = result.get("saved", False)
                status = "SAVED" if saved else f"SKIPPED ({result.get('reason', 'unknown')})"
                _payload = json.loads(payload_bytes)
                text_preview = _payload.get("text", "")[:100]
                pid = _payload.get("project_id", "?")
                f.write(f"[{pid}] {status}: {text_preview}...\n")
        except Exception:
            pass
    except Exception as e:
        try:
            log_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "hooks.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[bg] ERROR: {e!s}\n")
        except Exception:
            pass


def save(text: str, project_id: str, session_id: str = "", pin: bool = False, trigger: str = "", outcome: str = "", component: str = "general") -> bool:
    """
    Send save request to memory hook API.
    FIRE-AND-FORGET: spawns background thread so hook returns instantly.
    Returns True immediately (actual save happens async).
    """
    if not text or not text.strip() or not project_id:
        return False

    payload = json.dumps({
        "text": text,
        "project_id": project_id,
        "session_id": session_id,
        "pin": pin,
        "trigger": trigger,
        "outcome": outcome,
        "component": component,
        "is_manual": False,
    }).encode("utf-8")

    t = threading.Thread(target=_do_save, args=(payload,), daemon=True)
    t.start()
    return True


def read_stdin() -> dict:
    """Read JSON from stdin (Claude Code hook format)."""
    try:
        return json.loads(sys.stdin.read())
    except Exception:
        return {}
