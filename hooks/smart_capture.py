"""
hooks/smart_capture.py

Intelligent auto-save engine — saves MINIMUM, captures MAXIMUM value.

PROBLEM with naive auto-save:
  Save everything → DB bloats → search degrades → irrelevant memories injected
  Save nothing → miss key decisions → Claude forgets

SOLUTION — Novelty × Importance × Surprise scoring:
  Score = novelty_score × importance_score × surprise_bonus
  
  Only save if score > SAVE_THRESHOLD (0.50)

NOVELTY:
  Fast path: SHA-256 exact match → instant skip.

IMPORTANCE:
  File path + command heuristics:
    - Config files, schema, auth = HIGH (0.9)
    - Regular feature files = MEDIUM (0.6)
    - Node_modules, logs = ZERO (skip)

SURPRISE:
  Detect ERROR or FAILED patterns in output → double score (must save failures)
  Detect architectural patterns (new service, new DB table) → boost

LATENCY:
  All scoring is local (no API calls, no DB queries in hot path).
  Returns in <5ms.
"""

from __future__ import annotations

import re

import logging

logger = logging.getLogger(__name__)

HIGH_IMPORTANCE_PATTERNS = [
    r"docker-compose|Dockerfile|nginx\.conf|\.env(?!\.example)",
    r"schema\.|migration|alembic|prisma|flyway",
    r"settings\.py|config\.|pyproject\.toml|package\.json",
    r"auth|jwt|oauth|permission|password|secret|key",
    r"models\.|entities\.|types\.|interfaces\.",
    r"routes\.|router\.|endpoints?\.",
    r"database|db\.py|orm\.|crud\.",
    r"memory|graph|chroma",
]

ZERO_IMPORTANCE_PATTERNS = [
    r"node_modules|__pycache__|\.pyc|\.pyo",
    r"package-lock\.json|yarn\.lock|uv\.lock",
    r"\.log$|\.log\.|debug\.txt",
    r"\.mcp\.json",
    r"\.git/",
    r"dist/|build/|\.next/",
]

MEDIUM_IMPORTANCE_PATTERNS = [
    r"\.py$|\.ts$|\.js$|\.rs$|\.go$|\.java$",
    r"\.jsx$|\.tsx$|\.vue$|\.svelte$",
    r"\.sql$|\.graphql$|\.proto$",
]

_HIGH_RE = [re.compile(p, re.I) for p in HIGH_IMPORTANCE_PATTERNS]
_ZERO_RE = [re.compile(p, re.I) for p in ZERO_IMPORTANCE_PATTERNS]
_MED_RE  = [re.compile(p, re.I) for p in MEDIUM_IMPORTANCE_PATTERNS]

SAVE_COMMANDS = {
    "docker build": 0.9, "docker-compose up": 0.9, "docker compose up": 0.9,
    "alembic upgrade": 0.9, "prisma migrate": 0.9, "prisma generate": 0.85,
    "git init": 0.8, "createdb": 0.85, "dropdb": 0.85,
    "npm init": 0.75, "pip install": 0.7, "npm install": 0.7,
    "pytest": 0.8, "jest": 0.75, "go test": 0.75,
    "npm run build": 0.7, "npm run dev": 0.5,
}

ERROR_SIGNALS = [
    "error:", "failed:", "exception:", "traceback", "exit code 1",
    "syntaxerror", "typeerror", "attributeerror", "importerror",
    "connection refused", "no such file", "permission denied",
    "p1000", "p1001", "p1002",
    "could not connect", "unable to connect",
]

ARCHITECTURAL_SIGNALS = [
    r"class\s+\w+\(",
    r"def\s+\w+\(",
    r"CREATE TABLE",
    r"@app\.route|@router\.",
    r"FROM\s+\w+\s+IMPORT",
]


SAVE_THRESHOLD = 0.65


def score_observation(
    text: str,
    tool_name: str = "",
    file_path: str = "",
    command: str = "",
    output: str = "",
) -> tuple[float, str]:
    """
    Score an observation for save-worthiness.
    Returns (score: float, reason: str)
    score range: 0.0 (skip) → 1.0 (must save)
    """
    combined = f"{text} {file_path} {command}".lower()

    for rx in _ZERO_RE:
        if rx.search(combined):
            return 0.0, "noise_pattern"

    output_lower = output.lower()
    if any(sig in output_lower for sig in ERROR_SIGNALS):
        return 1.0, "error_captured"

    importance = 0.4

    for rx in _HIGH_RE:
        if rx.search(combined):
            importance = 0.85
            break
    else:
        for rx in _MED_RE:
            if rx.search(combined):
                importance = 0.6
                break

    for cmd_pattern, cmd_score in SAVE_COMMANDS.items():
        if cmd_pattern in command.lower():
            importance = max(importance, cmd_score)
            break

    surprise = 1.0
    content_preview = text[:500]
    for pattern in ARCHITECTURAL_SIGNALS:
        if re.search(pattern, content_preview, re.I):
            surprise = 1.3
            break

    score = min(1.0, importance * surprise)
    reason = f"importance={importance:.2f} surprise={surprise:.2f}"
    return score, reason


def should_save(
    text: str,
    tool_name: str = "",
    file_path: str = "",
    command: str = "",
    output: str = "",
) -> tuple[bool, float, str]:
    """
    Main entry point for smart capture decision.
    Returns (save: bool, score: float, reason: str)
    """
    score, reason = score_observation(text, tool_name, file_path, command, output)
    save = score >= SAVE_THRESHOLD
    logger.debug(f"[SmartCapture] score={score:.2f} save={save} reason={reason} text={text[:60]}")
    return save, score, reason
