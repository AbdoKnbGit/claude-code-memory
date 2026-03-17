#!/usr/bin/env python3
"""
CC-Memory Hook Setup Script
============================
Run this ONCE to configure Claude Code hooks properly.
Works on Windows, Linux, Mac — hooks run inside Docker.

Usage:
    python setup_hooks.py

Then restart Claude Code.

CORRECT HOOKS (2 only):
  PostToolUse      → saves Write/Bash outcomes to memory
  UserPromptSubmit → captures user decisions

NEVER register:
  PreToolUse  + session_start.py  → fires before EVERY tool = token bomb
  Notification + any script       → wrong event, crashes silently
  SessionStart + session_start.py → handled internally by proxy
"""

import json
import os
import sys
from pathlib import Path

DANGEROUS_HOOKS = {
    "PreToolUse": (
        "PreToolUse fires before EVERY tool call — not just session start.\n"
        "  Registering session_start.py here injects memory context 8-16x per session\n"
        "  instead of once, flooding conversation history with duplicate tokens."
    ),
    "SessionStart": (
        "SessionStart + session_start.py is redundant.\n"
        "  The proxy (routes.py) already handles memory injection at boot correctly.\n"
        "  Running it here adds a duplicate injection on top."
    ),
    "Notification": (
        "Notification is for desktop alerts — not for running scripts.\n"
        "  setup_hooks.py registered here crashes silently on every notification."
    ),
}


def find_claude_settings() -> Path:
    """Find or create Claude Code user settings file."""
    home = Path.home()
    settings_dir = home / ".claude"
    settings_dir.mkdir(exist_ok=True)
    settings_file = settings_dir / "settings.json"
    if not settings_file.exists():
        settings_file.write_text("{}", encoding="utf-8")
    return settings_file


def _docker_path(container: str) -> str:
    """Return correct path prefix for Docker exec commands.
    Double-slash //app ensures Windows Git Bash doesn't mangle paths,
    and is perfectly valid in Linux/Mac Docker environments too.
    """
    return "//app"


def audit_existing_hooks(hooks: dict) -> list[str]:
    """Check for dangerous hook registrations. Returns list of warnings."""
    warnings = []
    for event, entries in hooks.items():
        if event in DANGEROUS_HOOKS:
            warnings.append(f"\n  Dangerous hook found: [{event}]\n  {DANGEROUS_HOOKS[event]}")
    return warnings


def main():
    print("=" * 54)
    print("  CC-Memory Hook Setup")
    print("=" * 54)

    settings_path = find_claude_settings()
    print(f"\nSettings file: {settings_path}")

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        settings = {}

    current_hooks = settings.get("hooks", {})
    if current_hooks:
        print(f"\nExisting hooks: {list(current_hooks.keys())}")
        warnings = audit_existing_hooks(current_hooks)
        if warnings:
            print("\nDANGEROUS CONFIGURATIONS DETECTED:")
            for w in warnings:
                print(w)
            print()
        resp = input("Replace all hooks with correct configuration? (y/n): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return
    else:
        print("\nNo existing hooks found — fresh setup.")

    default_container = "cc-nim-memory"
    container = input(f"\nDocker container name [{default_container}]: ").strip() or default_container

    project_name = input("\nProject name (folder name of your project, or leave blank for cwd detection): ").strip()
    env_flag = f"-e CC_MEMORY_PROJECT={project_name} " if project_name else ""

    print(f"\nTesting Docker container '{container}'...")
    null_dev = "nul" if sys.platform == "win32" else "/dev/null"
    exit_code = os.system(f'docker exec {container} echo "ok" > {null_dev} 2>&1')
    if exit_code != 0:
        print(f"Cannot reach container '{container}'. Is it running?")
        print("   Try: docker ps")
        return
    print(f"Container '{container}' is reachable")

    app_path = _docker_path(container)

    hooks = {
        "PostToolUse": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"docker exec -i {env_flag}{container} python {app_path}/hooks/post_tool_use.py"
                    }
                ]
            }
        ],
        "UserPromptSubmit": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"docker exec -i {env_flag}{container} python {app_path}/hooks/user_prompt.py"
                    }
                ]
            }
        ],
    }

    settings["hooks"] = hooks

    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nHooks configured in {settings_path}")
    print("\nRegistered hooks (2 total):")
    print(f"  PostToolUse      → {app_path}/hooks/post_tool_use.py")
    print(f"  UserPromptSubmit → {app_path}/hooks/user_prompt.py")
    print("\nNOT registered (intentional):")
    print("  PreToolUse  — would fire before every tool, flooding history with tokens")
    print("  SessionStart — memory injection handled by proxy, not needed here")
    print("  Notification — wrong event type for scripts")
    print("\nRestart Claude Code for changes to take effect.")
    print("\nTo verify hooks are firing:")
    print(f"  docker logs {container} -f")
    print("  Then run a command in Claude Code — you should see [Hook] entries in logs.")


if __name__ == "__main__":
    main()
