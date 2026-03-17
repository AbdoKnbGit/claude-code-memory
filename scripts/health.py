#!/usr/bin/env python3
"""
Health check script — run after install to verify everything is working.
Can also be used standalone: python scripts/health.py
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

CONTAINER  = "cc-nim-memory"
MCP_NAME   = "cc-memory"
PROXY_PORT = 8082
CLAUDE_DIR = Path.home() / ".claude"


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def ok(msg):  print(f"  [OK] {msg}")
def fail(msg): print(f"  [!!] {msg}"); return False
def warn(msg): print(f"  [??] {msg}")


print("\n  cc-memory health check\n")
issues = []

r = run(f"docker inspect {CONTAINER} --format={{{{.State.Running}}}}")
if r.returncode == 0 and "true" in r.stdout:
    ok(f"Container '{CONTAINER}' running")
else:
    fail(f"Container '{CONTAINER}' not running")
    issues.append("container")

r = run(
    f'docker exec {CONTAINER} python -c '
    f'"import urllib.request,json; r=urllib.request.urlopen(\'http://localhost:{PROXY_PORT}/health\'); '
    f'd=json.loads(r.read()); print(d)"'
)
if r.returncode == 0 and "healthy" in r.stdout:
    ok("API server healthy")
else:
    fail("API server not responding")
    issues.append("api")

r = run(f'docker exec {CONTAINER} python -c "from memory import MEMORY_ENABLED; print(MEMORY_ENABLED)"')
if r.returncode == 0 and "True" in r.stdout:
    ok("Memory module loaded")
else:
    fail("Memory module not loaded")
    issues.append("memory")

settings_path = CLAUDE_DIR / "settings.json"
if settings_path.exists():
    try:
        s = json.loads(settings_path.read_text(encoding="utf-8"))
        h = s.get("hooks", {})
        expected = ["PreToolUse", "PostToolUse", "UserPromptSubmit"]
        registered = [k for k in expected if k in h]
        if len(registered) == len(expected):
            ok(f"All {len(expected)} hooks registered")
        else:
            warn(f"Only {len(registered)}/{len(expected)} hooks: {registered}")
            issues.append("hooks")
    except Exception as e:
        fail(f"Cannot read settings.json: {e}")
        issues.append("hooks")
else:
    warn("~/.claude/settings.json not found")

claude_bin = shutil.which("claude") or shutil.which("claude.cmd")
if claude_bin:
    r = run(f'"{claude_bin}" mcp list')
    if r.returncode == 0 and MCP_NAME in r.stdout:
        ok(f"MCP server '{MCP_NAME}' registered")
    else:
        warn(f"MCP server '{MCP_NAME}' not in mcp list")
        issues.append("mcp")
else:
    warn("Claude Code CLI not found — cannot verify MCP")

env_path = Path(".env")
if env_path.exists():
    env_text = env_path.read_text(encoding="utf-8")
    if "ANTHROPIC_BASE_URL" in env_text:
        ok(".env has ANTHROPIC_BASE_URL")
    else:
        warn(".env missing ANTHROPIC_BASE_URL — proxy may not be active")
else:
    warn(".env not found")

print()
if not issues:
    print("  Everything looks good! Start Claude Code with: claude")
else:
    print(f"  {len(issues)} issue(s): {', '.join(issues)}")
    print("  Run 'python setup.py' to repair.")
    sys.exit(1)
