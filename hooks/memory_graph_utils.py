"""
hooks/memory_graph_utils.py

Thin shim exposing graph.detect_component to hooks without importing the full memory module.
Hooks run as standalone subprocesses — they can't import from /app/memory directly.
This file lives in /app/hooks and is lightweight.
"""

import re

COMPONENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"auth|login|jwt|token|session|password|oauth|permission|role", re.I), "auth"),
    (re.compile(r"route|router|endpoint|controller|handler|api|view|request|response", re.I), "api"),
    (re.compile(r"model|schema|migration|database|db|sql|prisma|alembic|orm|entity", re.I), "db"),
    (re.compile(r"component|page|template|style|css|frontend|react|vue|svelte|html|ui", re.I), "ui"),
    (re.compile(r"docker|compose|nginx|deploy|k8s|kubernetes|infra|server|config|env|yaml", re.I), "infra"),
    (re.compile(r"test|spec|fixture|mock|assert|pytest|jest|coverage", re.I), "test"),
    (re.compile(r"hook|event|trigger|listener|middleware|interceptor", re.I), "hooks"),
    (re.compile(r"util|helper|common|shared|lib|tool|parser|format", re.I), "utils"),
    (re.compile(r"package|requirements|pyproject|cargo|gemfile|pom|gradle|dep", re.I), "deps"),
]


def detect_component_from_text(text: str, trigger: str = "") -> str:
    combined = f"{text} {trigger}".lower()
    for pattern, component in COMPONENT_PATTERNS:
        if pattern.search(combined):
            return component
    return "general"
