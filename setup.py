#!/usr/bin/env python3
"""
CC-Memory - Complete Setup
============================
Run from the cc-memory directory (where docker-compose.yml is).

    python setup.py

Prerequisites:
  - Docker installed and running
  - Claude Code CLI installed (npm install -g @anthropic-ai/claude-code)
  - Run setup.py — it will log you in and create .env automatically

What it does:
  1. Ensures Claude Code login (.credentials.json) — launches 'claude login' if needed
  2. Auto-creates .env with defaults if it doesn't exist
  3. Builds and starts Docker
  4. Detects host IP (WSL2-safe) and writes ANTHROPIC_BASE_URL to .env
  5. Registers MCP server in Claude Code (--scope user)
  6. Registers ALL hooks in Claude Code user settings
  7. Installs CLAUDE.md (global memory rules)
  8. Verifies everything

Works on Windows (cmd/PowerShell/Git Bash), Linux, Mac.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

def _bootstrap():
    """
    If we're not inside the project venv, install uv (if needed),
    create a Python 3.12 venv, and re-exec this script inside it.
    This is idempotent — safe to call on every run.
    """
    import os, sys, subprocess, shutil, platform
    from pathlib import Path

    venv_dir   = (Path(__file__).parent / ".venv").resolve()
    is_win     = platform.system() == "Windows"
    venv_python = venv_dir / ("Scripts/python.exe" if is_win else "bin/python")

    guard_env = "__CC_MEMORY_SETUP_RECURSION_GUARD__"
    if os.environ.get(guard_env) == "1":
        return

    exe_path = Path(sys.executable).resolve()
    if is_win:
        is_in_venv = str(exe_path).lower().startswith(str(venv_dir).lower())
    else:
        is_in_venv = str(exe_path).startswith(str(venv_dir))

    if is_in_venv:
        return

    print("  [setup] Checking Python environment...")

    uv = shutil.which("uv")
    if not uv:
        print("  [setup] uv not found — installing uv (Python env manager)...")
        install_cmd = (
            'powershell -ExecutionPolicy Bypass -Command '
            '"irm https://astral.sh/uv/install.ps1 | iex"'
            if is_win else
            'curl -LsSf https://astral.sh/uv/install.sh | sh'
        )
        r = subprocess.run(install_cmd, shell=True)
        if r.returncode != 0:
            print("  [!!] uv install failed. Install manually: https://docs.astral.sh/uv/")
            sys.exit(1)

        for candidate in [
            Path.home() / ".local/bin/uv",
            Path.home() / ".cargo/bin/uv",
            Path.home() / ".uv/bin/uv",
        ]:
            if candidate.exists():
                uv = str(candidate)
                break
        if not uv:
            uv = shutil.which("uv")
        if not uv:
            print("  [!!] uv installed but not found in PATH. Open a new terminal and re-run.")
            sys.exit(1)
        print(f"  [OK] uv installed: {uv}")

    if not venv_python.exists():
        print("  [setup] Creating Python 3.12 virtual environment via uv...")
        r = subprocess.run(
            [uv, "sync", "--python", "3.12"],
            cwd=Path(__file__).parent,
        )
        if r.returncode != 0:
            print("  [!!] uv sync failed — check pyproject.toml")
            sys.exit(1)
        print("  [OK] Python 3.12 environment ready")
    else:
        subprocess.run(
            [uv, "sync", "--python", "3.12"],
            cwd=Path(__file__).parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(f"  [setup] Re-launching inside Python 3.12 environment...\n")
    os.environ[guard_env] = "1"
    if is_win:
        args = subprocess.list2cmdline(sys.argv)
        exit_code = os.system(f'"{venv_python}" {args}')
        sys.exit(exit_code)
    else:
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

_bootstrap()


try:
    from typing import Any, cast
    if hasattr(sys.stdout, "reconfigure"):
        cast(Any, sys.stdout).reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        cast(Any, sys.stderr).reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


CONTAINER  = "cc-nim-memory"
MCP_NAME   = "cc-memory"
PROXY_PORT = 8082
CLAUDE_DIR = Path.home() / ".claude"


def is_win():
    return platform.system() == "Windows"

def is_wsl():
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False

def run(cmd, **kwargs):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace', **kwargs)

def _check_port(port):
    import socket
    s = socket.socket()
    try:
        s.bind(("", port))
        s.close()
        return True
    except OSError:
        return False

def ask(prompt, default=""):
    if "--auto" in sys.argv:
        return default
    try:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        return default

def _safe(msg):
    try:
        msg.encode(sys.stdout.encoding or "utf-8")
        return msg
    except (UnicodeEncodeError, LookupError):
        return msg.encode("ascii", "replace").decode("ascii")

def green(msg): print(_safe(f"  [OK] {msg}"))
def red(msg):   print(_safe(f"  [!!] {msg}"))
def warn(msg):  print(_safe(f"  [??] {msg}"))
def info(msg):  print(_safe(f"  >>  {msg}"))
def hdr(title): print(_safe(f"\n+------ {title} ------+"))


def _docker_compose(subcmd):
    """Try 'docker compose <subcmd>' first, fall back to 'docker-compose <subcmd>'."""
    r = run(f"docker compose {subcmd}")
    if r.returncode == 0 or "unknown command" not in (r.stderr or "").lower():
        return r
    return run(f"docker-compose {subcmd}")

def _check_network():
    """Quick check for Docker registry connectivity (no-op, always passes)."""
    return True


def _detect_host_ip():
    """Return the IP Claude Code should use to reach the proxy container."""
    if is_wsl():
        r = run("ip route show default")
        m = re.search(r"via (\d+\.\d+\.\d+\.\d+)", r.stdout)
        if m:
            return m.group(1)

    if is_win():
        r = run("ping -n 1 host.docker.internal")
        m = re.search(r"\[(\d+\.\d+\.\d+\.\d+)\]", r.stdout)
        if m:
            return m.group(1)

    if platform.system() == "Darwin":
        r = run("ping -c 1 host.docker.internal")
        m = re.search(r"\((\d+\.\d+\.\d+\.\d+)\)", r.stdout)
        if m:
            return m.group(1)

    fmt = '{{range .IPAM.Config}}{{.Gateway}}{{end}}'
    r = run(f'docker network inspect bridge --format "{fmt}"')
    ip = r.stdout.strip().strip("'\"")
    if ip and re.match(r"\d+\.\d+\.\d+\.\d+", ip):
        return ip

    return "127.0.0.1"


def _read_env():
    env = {}
    p = Path(".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env

def _write_env_key(key, value):
    """Add or replace a key in .env (preserves all other lines)."""
    p = Path(".env")
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}=") and not line.strip().startswith("#"):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f"{key}={value}")
    p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


_DEFAULT_ENV_TEMPLATE = """\
# CC-Memory configuration
# Copy this file to .env and fill in your values.
#
# MEMORY_MODEL: the cheap/fast LLM used internally for summarisation,
# compression, judge tasks. Cerebras is free-tier and fastest.
#
# Set ANTHROPIC_BASE_URL to http://<host-ip>:8082 after setup.
#

# Memory summarisation model (Cerebras free tier — fastest inference)
MEMORY_MODEL=cerebras/llama-3.3-70b
CEREBRAS_API_KEY=           # https://cloud.cerebras.ai — free tier available

# Alternative: Groq (also free tier)
# MEMORY_MODEL=groq/llama-3.1-8b-instant
# GROQ_API_KEY=your-key

# Alternative: Gemini Flash (free tier)
# MEMORY_MODEL=gemini/gemini-2.0-flash
# GEMINI_API_KEY=your-key

# Claude Code uses OAuth login — no ANTHROPIC_API_KEY needed
# (If you prefer raw API keys instead of OAuth, uncomment below)
# ANTHROPIC_API_KEY=sk-ant-your-key

# Proxy URL — set after setup detects your host IP
# ANTHROPIC_BASE_URL=http://<host-ip>:8082

# Memory tuning
MEMORY_ENABLED=true
MEMORY_TOKEN_BUDGET=400
MEMORY_DEDUP_THRESHOLD=0.98

# Output / history limits
LARGE_OUTPUT_CAP=3000
HISTORY_COMPRESS_THRESHOLD=4600
HISTORY_KEEP_RECENT=8
TOOL_RESULT_SUMMARIZE_THRESHOLD=1500

# Smart compaction
SMART_COMPACT_THRESHOLD_PCT=0.30
SMART_COMPACT_GROWTH_TRIGGER=2000
SMART_COMPACT_FLOOR_TOKENS=15000
SMART_COMPACT_MIN_TURNS=8
SMART_COMPACT_COOLDOWN_TURNS=4
SMART_COMPACT_COST_CEILING=60000

# Memory injection
INJECTION_BUDGET=3000
INJECTION_TTL_DAYS=14

# Deduplication
DEDUP_WINDOW=2.0
DEDUP_MIN_OUT_TOKENS=15
DEDUP_TTL_SECONDS=10.0
MIN_INJECTION_VALUE=0.05

# Subagent context window
SUBAGENT_CTX_WINDOW_SEC=1800
"""

def _auto_create_env(creds):
    """Print instructions for creating .env — setup.py never writes .env."""
    Path(".env.example").write_text(_DEFAULT_ENV_TEMPLATE, encoding="utf-8")
    warn(".env not found — copy .env.example to .env and fill in your values:")
    info("  cp .env.example .env")
    info("Then set at minimum: MEMORY_MODEL and the matching API key.")


def setup_memory_llm_key():
    """Ensure the API key for the configured MEMORY_MODEL provider is set."""
    env = _read_env()
    model = env.get("MEMORY_MODEL", "cerebras/llama-3.3-70b")
    provider = model.split("/")[0].lower() if "/" in model else ""

    key_map = {
        "cerebras": ("CEREBRAS_API_KEY", "cloud.cerebras.ai"),
        "groq":     ("GROQ_API_KEY", "console.groq.com"),
        "gemini":   ("GEMINI_API_KEY", "aistudio.google.com"),
        "deepseek": ("DEEPSEEK_API_KEY", "platform.deepseek.com"),
    }

    if provider not in key_map:
        info(f"Memory model provider '{provider}' — no API key setup needed")
        return

    key_name, signup_url = key_map[provider]
    if env.get(key_name):
        green(f"{key_name} already set")
        return

    info(f"{key_name} needed for memory compression (free at {signup_url})")
    key = ask(f"Paste your {key_name} (or press Enter to skip)", "")
    if key:
        _write_env_key(key_name, key)
        green(f"{key_name} saved to .env")
    else:
        warn(f"Skipped — memory compression will fail without {key_name}")
        info(f"Add later: edit .env and set {key_name}=your-key")


def _wait_for_login():
    """Block until ~/.claude/.credentials.json appears (user runs 'claude login')."""
    claude_bin = shutil.which("claude") or shutil.which("claude.cmd")
    creds_path = CLAUDE_DIR / ".credentials.json"

    if not claude_bin:
        red("Claude Code CLI not found — install first:")
        info("npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    print()
    info("No credentials found. Launching 'claude login' for you...")
    info("Complete the browser login, then come back here.")
    print()

    try:
        subprocess.run(f'"{claude_bin}" login', shell=True)
    except KeyboardInterrupt:
        pass

    for _ in range(60):
        if creds_path.exists():
            try:
                c = json.loads(creds_path.read_text(encoding="utf-8"))
                if c.get("claudeAiOauth"):
                    return c
            except Exception:
                pass
        time.sleep(1)
        print(".", end="", flush=True)

    print()
    red("Login timed out or credentials not written.")
    red("Run 'claude login' manually, then re-run setup.py.")
    sys.exit(1)


def check_env():
    hdr("Step 1/6 - Credentials + .env")

    creds_path = CLAUDE_DIR / ".credentials.json"
    creds = {}


    if creds_path.exists():
        try:
            creds = json.loads(creds_path.read_text(encoding="utf-8"))
            if creds.get("claudeAiOauth"):
                green("Claude Code OAuth credentials found")
            else:
                warn(".credentials.json exists but OAuth token missing — re-logging in")
                creds = _wait_for_login()
        except Exception:
            warn(".credentials.json unreadable — re-logging in")
            creds = _wait_for_login()
    else:
        creds = _wait_for_login()
        print()
        green("Login successful")


    if not Path(".env").exists():
        _auto_create_env(creds)
        sys.exit(1)
    else:
        green(".env found")

    env = _read_env()
    model = env.get("MEMORY_MODEL", env.get("MODEL", "(not set)"))
    info(f"Memory model: {model}")

    provider = model.split("/")[0].lower() if "/" in model else ""
    key_hints = {
        "cerebras": ("CEREBRAS_API_KEY", "cloud.cerebras.ai"),
        "groq": ("GROQ_API_KEY", "console.groq.com"),
        "gemini": ("GEMINI_API_KEY", "aistudio.google.com"),
    }
    if provider in key_hints:
        key_name, url = key_hints[provider]
        if not env.get(key_name):
            info(f"Tip: add {key_name} to .env (free at {url})")

    return env


def setup_docker():
    hdr("Step 2/6 - Docker")
    global PROXY_PORT

    r = run("docker --version")
    if r.returncode != 0:
        red("Docker not found — install Docker Desktop first")
        sys.exit(1)
    green(r.stdout.strip())

    for _attempt in range(10):
        if _check_port(PROXY_PORT):
            break
        chk = run(f"docker inspect {CONTAINER} --format={{{{.State.Running}}}}")
        if "true" in chk.stdout:
            green("Port taken by our own container — already running")
            return True
        warn(f"Port {PROXY_PORT} is in use.")
        new_port = ask("Enter a different port", "8083")
        try:
            PROXY_PORT = int(new_port)
        except ValueError:
            PROXY_PORT = 8083
    else:
        red(f"Could not find a free port after 10 attempts")
        return False

    chk = run(f"docker inspect {CONTAINER} --format={{{{.State.Running}}}}")
    if chk.returncode == 0 and "true" in chk.stdout:
        green(f"Container '{CONTAINER}' already running")
        if "--auto" in sys.argv:
            return True
        resp = ask("Rebuild? (y/n)", "n")
        if resp.lower() != "y":
            return True

    info(f"[{time.strftime('%H:%M:%S')}] Building Docker image — output streamed live...")
    info("Do NOT press Ctrl+C — Docker is working even if it looks slow.")
    
    _check_network()

    build_r = subprocess.run("docker compose build", shell=True, cwd=Path(__file__).parent)
    if build_r.returncode != 0:
        red("Docker build failed — scroll up to see the error")
        return False
    green("Image built")

    info(f"[{time.strftime('%H:%M:%S')}] Starting container...")
    _docker_compose("down")
    up_r = _docker_compose("up -d")
    if up_r.returncode != 0:
        red("Docker start failed")
        stderr = up_r.stderr or ""
        err = stderr[:500].encode("ascii", "replace").decode("ascii")
        print(f"     {err}")
        return False

    info("Waiting for container to be healthy (first run may take 2-3min for model download)...")
    for _ in range(120):
        h = run(
            f"docker exec {CONTAINER} python -c "
            f"\"import urllib.request; urllib.request.urlopen('http://localhost:{PROXY_PORT}/health')\""
        )
        if h.returncode == 0:
            print()
            green("Container healthy")
            return True
        time.sleep(1)
        print(".", end="", flush=True)

    print()
    red("Container not healthy after 120s")
    info(f"Check logs: docker logs {CONTAINER}")
    return False


def setup_base_url(env):
    hdr("Step 3/6 - Host IP + ANTHROPIC_BASE_URL")

    existing = env.get("ANTHROPIC_BASE_URL", "")
    if existing:
        info(f"ANTHROPIC_BASE_URL already set: {existing}")
        resp = ask("Re-detect and update? (y/n)", "n")
        if resp.lower() != "y":
            green(f"Keeping: {existing}")
            return existing

    ip = _detect_host_ip()
    url = f"http://{ip}:{PROXY_PORT}"
    info(f"Detected host IP: {ip}")

    resp = ask(f"Use ANTHROPIC_BASE_URL={url}  (enter to confirm or type new IP)", ip)
    if resp != ip:
        ip = resp
        url = f"http://{ip}:{PROXY_PORT}"

    info(f"Add this to your .env:  ANTHROPIC_BASE_URL={url}")
    green(f"ANTHROPIC_BASE_URL={url}")
    return url


def setup_mcp():
    hdr("Step 4/6 - MCP Server")

    claude_bin = shutil.which("claude") or shutil.which("claude.cmd")
    if not claude_bin:
        warn("Claude Code CLI not found")
        info("Install: npm install -g @anthropic-ai/claude-code")
        return
    green(f"Claude Code found: {claude_bin}")

    r = run(f'"{claude_bin}" mcp list')
    if r.returncode == 0 and MCP_NAME in r.stdout:
        green(f"MCP server '{MCP_NAME}' already registered")
        return

    mcp_path = "//app/mcp_server.py" if is_win() else "/app/mcp_server.py"
    cmd = (
        f'"{claude_bin}" mcp add {MCP_NAME} --transport stdio --scope user '
        f'-- docker exec -i {CONTAINER} python {mcp_path}'
    )
    info("Registering MCP server...")
    r = run(cmd)
    if r.returncode == 0:
        green(f"MCP server '{MCP_NAME}' registered (scope: user)")
    else:
        warn("Auto-register failed. Run manually:")
        print(f"     {cmd}")


def setup_hooks():
    hdr("Step 5/6 - Claude Code Hooks")

    settings_path = CLAUDE_DIR / "settings.json"
    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)

    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            settings = {}

    project_name = Path.cwd().name.lower().replace(" ", "_")
    project_name = re.sub(r"[^a-z0-9_-]", "_", project_name).strip("_") or "default"
    env_flag = f"-e CC_MEMORY_PROJECT={project_name} "


    def hook(script):
        path = "//app/hooks/" if is_win() else "/app/hooks/"
        return {
            "type": "command",
            "command": f"docker exec -i {env_flag}{CONTAINER} python {path}{script}"
        }

    existing_hooks = settings.get("hooks", {})

    our_hooks = {
        "PostToolUse":      [{"matcher": "", "hooks": [hook("post_tool_use.py")]}],
        "UserPromptSubmit": [{"matcher": "", "hooks": [hook("user_prompt.py")]}],
    }

    for event, entries in our_hooks.items():
        existing = existing_hooks.get(event, [])
        filtered = [e for e in existing
                    if not any(CONTAINER in h.get("command", "")
                               for h in e.get("hooks", []))]
        existing_hooks[event] = filtered + entries

    settings["hooks"] = existing_hooks

    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    green(f"PostToolUse      -> post_tool_use.py  (CC_MEMORY_PROJECT={project_name})")
    green(f"UserPromptSubmit -> user_prompt.py    (CC_MEMORY_PROJECT={project_name})")
    info("PreToolUse and Notification intentionally NOT registered (safe 2-hook standard)")
    info(f"Saved to {settings_path}")

def setup_shell_profile(base_url):
    """Guard for ANTHROPIC_BASE_URL conflict."""
    existing = os.environ.get("ANTHROPIC_BASE_URL", "")

    if existing and existing != base_url:
        warn(f"ANTHROPIC_BASE_URL already set: {existing}")
        warn("This points to another proxy/MCP.")
        resp = ask("Overwrite? This will disable your existing proxy. (y/n)", "n")
        if resp.lower() != "y":
            info("Skipping. To use cc-memory, set manually before launching Claude Code:")
            info(f'export ANTHROPIC_BASE_URL="{base_url}"')
            info("Or create an alias:")
            info(f'alias claude-mem="ANTHROPIC_BASE_URL={base_url} claude"')
            return False
    return True


def setup_claude_md(base_url):
    hdr("Step 6/6 - CLAUDE.md + Verify")

    target = CLAUDE_DIR / "CLAUDE.md"
    MARKER = "# cc-memory"

    NEW_CONTENT = f"""{MARKER}
# CLAUDE.md — Memory Rules

---
## ⚠️ MANDATORY FIRST ACTION — BEFORE ANYTHING ELSE

**Before reading the user's message. Before thinking. Before responding.**
**Run this check immediately when the session starts:**

```bash
ls .mcp.json 2>/dev/null && echo "EXISTS" || echo "MISSING"
```

### If result is MISSING:
1. `Bash("pwd")` → get the current path
2. `memory_init(project_path="<FULL_PATH>")`
3. Write the two generated files exactly as returned
4. Output ONLY this — nothing else:
   `✅ Memory ready for **<folder>**. Please restart Claude Code.`
5. **FULL STOP.** Do not read the user message. Do not continue. Wait.

### If result is EXISTS:
- Boot is done. Now read the user's message and proceed normally.
- Do NOT call any memory tool yet. Just answer.

> This check runs ONCE per session, on the very first exchange, no exceptions.
> Even if the user says "hello" or asks something simple — check first.

---
## WHEN TO CALL MEMORY TOOLS

```
User sends a message (boot check already done above)
│
├─ First message of session AND it's a project question?
│   └─ memory_remember(project_id="<folder>") — ONCE, silently, then answer
│
├─ Any question about this project's code, config, stack, or decisions?
│   (what port, what stack, how does X work, what did we use, where is Y)
│   └─ memory_context(query="<user message>", project_id="<folder>")
│       Score < 0.60 → nothing found → answer normally
│
├─ User says "search" / "find" / "do we have X"?
│   └─ memory_search(query="<what>", project_id="<folder>")
│
└─ Everything else → answer directly, no memory call
```

`memory_remember` — first project question only, once per session, never again.
`memory_context` — every project question after the first. Silent.
`memory_search` — only when user explicitly asks to search.

---
## SAVING TO MEMORY

**When to suggest:** a meaningful technical decision was just made:
- Bug fixed with non-obvious solution
- Architecture decision confirmed
- Feature completed
- Tech choice locked in

**How:**
1. Call `memory_suggest(context="<one sentence>", project_id="<folder>")`
2. Append exactly this line — no variations:
```
💾 Store in memory? `store` · `pin` · `skip`
```
Use `store`/`pin` — never `save` (intercepted by editor).

**User replies:**
- `store` → `memory_save(text="<one sentence>", project_id="<folder>", pin=False)`
- `pin`   → `memory_save(text="<one sentence>", project_id="<folder>", pin=True)`
- `skip` or silence → nothing

**Save format:** plain text only. No emojis, no markdown. Max 30 tokens. WHAT + WHY + constraint.
Never call `memory_save()` without explicit user confirmation.

**After every save:** if entries > 10 OR tokens > 350 → call `memory_reduce(project_id="<folder>")` immediately.

---
## OUTPUT CAP — COST CRITICAL

Every output token is uncached input on every future call.

| Situation | Max output |
|---|---|
| After writing 3+ files | 2 sentences |
| After bulk bash/install | 1 sentence |
| Tool confirmation | Silent, move to next step |
| Mid-task between tool calls | 0 filler |
| User asks "what did you do" | Then and only then summarize |

Never generate: file lists, directory trees, feature recaps, setup instructions.

---
## OTHER RULES

**Conflict:** memory says X, user says Y → "Memory says X — you're saying Y. Which is correct?" Then update.

**project_id:** always explicit on every memory tool call. Use the CWD folder name. Never rely on env defaults.

**memory_manage / memory_clear / memory_reindex / memory_export:** only on explicit user request. Never auto-call.

---
## TOOL REFERENCE

| Tool | When |
|---|---|
| `memory_remember(project_id)` | First project question of session — once only |
| `memory_context(query, project_id)` | Every project question after the first |
| `memory_search(query, project_id)` | User explicitly asks to search |
| `memory_suggest(context, project_id)` | Decision detected — before asking user |
| `memory_save(text, project_id, pin)` | After user confirms with "store" or "pin" |
| `memory_reduce(project_id)` | Auto after save if over budget |
| `memory_manage(action, id, project_id)` | User request only |
| `memory_status(project_id)` | User asks |
| `memory_init(project_path)` | Only if .mcp.json missing — handled in boot |

---
"""

    CLAUDE_DIR.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = target.read_text(encoding="utf-8")
        if MARKER in existing:
            # Already injected — skip to avoid duplicates
            green("CLAUDE.md already configured — skipping")
        else:
            # Concat: our rules first, then the existing content
            combined = NEW_CONTENT + "\n" + existing
            target.write_text(combined, encoding="utf-8")
            green(f"CLAUDE.md updated — memory rules prepended to existing content")
    else:
        # Fresh install — just write ours
        target.write_text(NEW_CONTENT, encoding="utf-8")
        green(f"CLAUDE.md installed to {target}")

    print("\n  -- Verification --")
    errors = []

    r = run(f"docker exec {CONTAINER} echo ok")
    if r.returncode == 0:
        green("Docker container running")
    else:
        red("Docker container not running")
        errors.append("docker")

    api_ok = False
    for _ in range(10):
        r = run(
            f"docker exec {CONTAINER} python -c "
            f"\"import urllib.request,json; r=urllib.request.urlopen('http://localhost:{PROXY_PORT}/health'); "
            f"d=json.loads(r.read()); print(d)\""
        )
        if r.returncode == 0 and "healthy" in r.stdout:
            green("API server healthy")
            api_ok = True
            break
        time.sleep(2)
    if not api_ok:
        red("API not responding")
        errors.append("api")

    r = run(f'docker exec {CONTAINER} python -c "from memory import MEMORY_ENABLED; print(MEMORY_ENABLED)"')
    if r.returncode == 0 and "True" in r.stdout:
        green("Memory module loaded")
    else:
        red("Memory module failed")
        errors.append("memory")

    env_now = _read_env()
    burl = env_now.get("ANTHROPIC_BASE_URL", "")
    if burl:
        green(f"ANTHROPIC_BASE_URL={burl}")
    else:
        red("ANTHROPIC_BASE_URL not set in .env — proxy won't be used!")
        errors.append("base_url")

    settings_path = CLAUDE_DIR / "settings.json"
    if settings_path.exists():
        try:
            s = json.loads(settings_path.read_text(encoding="utf-8"))
            h = s.get("hooks", {})
            if isinstance(h, dict):
                registered = [k for k in ["PostToolUse", "UserPromptSubmit"] if k in h]
                if len(registered) == 2:
                    green("Both hooks registered (PostToolUse + UserPromptSubmit)")
                else:
                    warn(f"Only {len(registered)}/2 hooks: {registered}")
        except Exception:
            red("Cannot read Claude settings")
            errors.append("settings")

    claude_bin = shutil.which("claude") or shutil.which("claude.cmd")
    if claude_bin:
        r = run(f'"{claude_bin}" mcp list')
        if r.returncode == 0 and MCP_NAME in r.stdout:
            green(f"MCP server '{MCP_NAME}' registered")
        else:
            warn("MCP server not detected in claude mcp list")
    else:
        warn("Claude CLI not found — cannot verify MCP")

    if (CLAUDE_DIR / "CLAUDE.md").exists():
        green("CLAUDE.md installed")

    print("\n" + "=" * 52)
    if not errors:
        print("  Setup complete!")
        print("=" * 52)
        print(f"""
  Launch Claude Code (proxy active automatically):

    claude

  Watch memory activity:
    docker logs {CONTAINER} -f
""")
    else:
        print(f"  Issues found: {', '.join(errors)}")
        print("=" * 52)
        print("  Fix them and run setup.py again.")


def main():
    print()
    print("  +-------------------------------------------+")
    print("  |     CC-Memory - Complete Setup            |")
    print("  |     One command. Any machine. Done.       |")
    print("  +-------------------------------------------+")

    if not Path("docker-compose.yml").exists():
        red("Run this from the cc-memory directory")
        info("cd /path/to/cc-memory && python setup.py")
        sys.exit(1)

    env = check_env()
    setup_memory_llm_key()

    ok = setup_docker()
    if not ok:
        warn("Docker had issues - continuing anyway")

    base_url = setup_base_url(env)

    existing = os.environ.get("ANTHROPIC_BASE_URL", "")
    if not existing:
        setup_shell_profile(base_url)
    elif existing == base_url:
        green("ANTHROPIC_BASE_URL already correct in environment")
    else:
        warn(f"Another proxy detected: {existing}")
        resp = ask("Use CC-Memory instead? (y/n)", "n")
        if resp.lower() == "y":
            setup_shell_profile(base_url)

    setup_mcp()
    setup_hooks()
    setup_claude_md(base_url)


if __name__ == "__main__":
    main()