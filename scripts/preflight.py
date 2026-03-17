#!/usr/bin/env python3
"""
Preflight check — run before setup.py to verify all prerequisites.
Called automatically by the plugin marketplace before install.
Exit 0 = all good. Exit 1 = missing prerequisites.
"""

import shutil
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


checks = []


def ok(msg):
    print(f"  [OK] {msg}")


def fail(msg):
    print(f"  [!!] {msg}")
    checks.append(msg)


def warn(msg):
    print(f"  [??] {msg}")


print("\n  cc-memory preflight check\n")

if sys.version_info >= (3, 10):
    ok(f"Python {sys.version.split()[0]}")
else:
    fail(f"Python 3.10+ required (got {sys.version.split()[0]})")

r = run("docker --version")
if r.returncode == 0:
    ok(r.stdout.strip())
else:
    fail("Docker not found — install Docker Desktop: https://docker.com")

r = run("docker info")
if r.returncode == 0:
    ok("Docker daemon running")
else:
    fail("Docker daemon not running — start Docker Desktop")

claude_bin = shutil.which("claude") or shutil.which("claude.cmd")
if claude_bin:
    r = run(f'"{claude_bin}" --version')
    ver = r.stdout.strip() or "found"
    ok(f"Claude Code CLI: {ver}")
else:
    fail("Claude Code CLI not found — install: npm install -g @anthropic-ai/claude-code")

r = run("docker compose version")
if r.returncode == 0:
    ok(r.stdout.strip())
else:
    r2 = run("docker-compose --version")
    if r2.returncode == 0:
        warn(f"Using legacy docker-compose: {r2.stdout.strip()}")
    else:
        fail("Neither 'docker compose' nor 'docker-compose' found")

print()
if checks:
    print(f"  {len(checks)} issue(s) found — fix them before installing:\n")
    for c in checks:
        print(f"    • {c}")
    print()
    sys.exit(1)
else:
    print("  All checks passed. Ready to install.")
    sys.exit(0)
