from __future__ import annotations

"""
memory/__init__.py

MCP Memory layer — persistent memory for Claude Code proxy.

Features:
  - SQLite + ChromaDB dual storage with WAL mode
  - Multilingual support (FR/EN + fallback) via langdetect
  - Intelligent autosave with 8-step filtering pipeline
  - Deduplication (cosine similarity > threshold)
  - Contradiction detection & superseding
  - Temporal + pinned + session proximity scoring
  - 14 slash commands with fast-path SSE responses
  - Graceful degradation on storage failures

Commands:
  /save <text>           Save decision/info
  /save! <text>          Save + pin (never compressed)
  /search <query>        Semantic search
  /forget <id>           Soft-delete entry
  /remember [query]      Inject top memories into system prompt
  /rollback              Show last saved entry
  /pin <id>              Pin entry
  /unpin <id>            Unpin entry
  /reduce                Batch-compress unpinned entries via AI
  /status                Stats (zero API calls)
  /export                Dump JSON
  /clear session         Clear current session
  /clear project         Clear entire project (with confirmation)
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import uuid
from collections import OrderedDict
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager, closing
import threading

import chromadb
from loguru import logger
import httpx

MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")
MEMORY_TOKEN_BUDGET: int = int(os.getenv("MEMORY_TOKEN_BUDGET", "400"))
MEMORY_TOP_K: int = int(os.getenv("MEMORY_TOP_K", "8"))
LLM_RERANK_ENABLED: bool = os.getenv("LLM_RERANK_ENABLED", "true").lower() == "true"
LLM_RERANK_CANDIDATES: int = int(os.getenv("LLM_RERANK_CANDIDATES", "20"))
MEMORY_DEDUP_THRESHOLD: float = float(os.getenv("MEMORY_DEDUP_THRESHOLD", "0.98"))
TIER_FULL_THRESHOLD: float = float(os.getenv("TIER_FULL_THRESHOLD", "0.82"))
TIER_HEAD_THRESHOLD: float = float(os.getenv("TIER_HEAD_THRESHOLD", "0.65"))
MEMORY_AUTO_REDUCE_THRESHOLD: int = int(os.getenv("MEMORY_AUTO_REDUCE_THRESHOLD", "200"))
MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
MEMORY_CHROMA_PATH: str = os.getenv("MEMORY_CHROMA_PATH", "./data/chroma")
MEMORY_EMBEDDING_MODEL: str = os.getenv(
    "MEMORY_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)


if sys.platform == "win32":
    MEMORY_CHROMA_PATH = os.path.abspath(MEMORY_CHROMA_PATH).replace("\\", "/")
    if hasattr(sys.stdout, 'reconfigure') and sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

Path(MEMORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(MEMORY_CHROMA_PATH).mkdir(parents=True, exist_ok=True)


_chroma_checked: set[str] = set()
_llm_events: list[dict] = []
_LLM_EVENTS_MAX = 200
_llm_events_lock = threading.Lock()

_llm_fallback_stats: dict[str, dict] = {
    "totals": {"calls": 0, "successes": 0, "failures": 0, "fallbacks": 0},
    "by_task": {},
}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning models (kimi-k2, deepseek-r1, qwq, etc.)."""
    if not text:
        return text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()


_SMART_CLASSIFIER_SYSTEM = """You are a memory classifier for a coding assistant.
Analyze the text and respond with ONLY valid JSON, no markdown, no explanation.

JSON schema:
{
  "action": "SAVE_PIN" | "SAVE" | "SKIP",
  "confidence": 0.0-1.0,
  "summary": "<1 dense sentence, max 120 chars, same language as input>",
  "tags": ["#tag1", "#tag2"],
  "component": "auth"|"api"|"db"|"ui"|"infra"|"test"|"hooks"|"utils"|"deps"|"general",
  "is_resume_signal": true|false,
  "reasoning": "<10 words max>"
}

Rules:
- SAVE_PIN: architecture decisions, constraints, "never/always use X", user preferences, critical bugs fixed
- SAVE: tech choices with rationale, implemented features, bug fixes, project structure facts
- SKIP: greetings, planning language ("I will", "let me", "I'll create"), pure code blocks, confirmations ("ok", "sure", "done"), filler
- summary: extract the core fact only, dense, no filler words
- tags: pick from #auth #api #database #architecture #security #testing #devops #bug #performance #frontend #backend #config
- component: the code partition this memory belongs to
- is_resume_signal: true if text implies continuing from a previous session ("where were we", "last time", "continue", "reprendre")
"""

_FALLBACK_SKIP_PREFIXES = [
    "i'll ", "i will ", "let me ", "i'm going to ", "here's ", "sure,",
    "okay,", "ok,", "je vais ", "voici ", "d'accord",
]
_FALLBACK_SAVE_PIN_KEYWORDS = [
    "we decided", "never use", "always use", "rule:", "constraint:",
    "architecture:", "on a décidé", "ne jamais", "toujours utiliser",
]
_FALLBACK_SAVE_KEYWORDS = [
    "i chose", "instead of", "because", "i implemented", "we use",
    "j'ai choisi", "plutôt que", "j'ai implémenté",
]
_FALLBACK_RESUME_SIGNALS = [
    "continue", "reprendre", "where were we", "last time", "back to",
    "on était où", "suite de", "what did we",
]

NEGATORS_EN = [
    "no longer", "instead", "switched", "replaced", "dropped",
    "abandoned", "changed", "deprecated", "removed", "now", "not",
]
NEGATORS_FR = [
    "ne plus", "plus", "finalement", "abandonnée", "remplacé",
    "abandonné", "remplacée", "changé", "changée", "désormais", "non",
]

PROVIDER_CONFIG = {
    "nvidia_nim": {
        "url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "key_env": "NVIDIA_NIM_API_KEY",
        "format": "openai",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
        "format": "anthropic",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "format": "openai",
    },
    "open_router": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "format": "openai",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "format": "openai",
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "format": "openai",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "format": "openai",
    },
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "key_env": "CEREBRAS_API_KEY",
        "format": "openai",
    },
    "lmstudio": {
        "url": None,
        "key_env": None,
        "format": "openai",
    },
}

def _call_llm(system: str, user: str, max_tokens: int = 100, timeout: float = 12.0, task: str = "ai_task") -> str | None:
    """
    Unified LLM call. Reads MEMORY_MODEL (or MODEL fallback) to pick provider and model.
    Returns raw text or None on failure.
    Strips <think> tags automatically.
    """
    _t0 = time.time()

    _llm_fallback_stats["totals"]["calls"] += 1
    task_stats = _llm_fallback_stats["by_task"].setdefault(task, {"calls": 0, "successes": 0, "failures": 0, "fallbacks": 0})
    task_stats["calls"] += 1

    model_env = os.getenv("MEMORY_MODEL", "") or os.getenv("MODEL", "")
    if not model_env or "/" not in model_env:
        logger.warning("LLM_TRACE: SKIP task=%s reason=MEMORY_MODEL_NOT_SET", task)
        _llm_fallback_stats["totals"]["failures"] += 1
        task_stats["failures"] += 1
        return None

    provider, model_name = model_env.split("/", 1)
    provider = provider.lower()

    event = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "timestamp": _t0,
        "model": f"{provider}/{model_name}",
        "provider": provider,
        "task": task,
        "status": "working...",
        "input_chars": len(user),
        "input_tokens_est": len(user) // 3,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "attempts": [],
        "fallback_triggered": False,
    }
    with _llm_events_lock:
        _llm_events.insert(0, event)
        while len(_llm_events) > _LLM_EVENTS_MAX:
            _llm_events.pop()

    cfg = PROVIDER_CONFIG.get(provider)
    if not cfg:
        event["status"] = "ERROR: unknown_provider"
        event["latency_ms"] = int((time.time() - _t0) * 1000)
        logger.warning(f"LLM_TRACE: FAIL task={task} reason=unknown_provider provider={provider}")
        _llm_fallback_stats["totals"]["failures"] += 1
        task_stats["failures"] += 1
        return None

    if provider == "lmstudio":
        base = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        url = f"{base}/chat/completions"
    else:
        url = cfg["url"]

    api_key = None
    if cfg["key_env"]:
        api_key = os.getenv(cfg["key_env"], "").strip()
        if not api_key:
            event["status"] = f"ERROR: {cfg['key_env']}_not_set"
            event["latency_ms"] = int((time.time() - _t0) * 1000)
            logger.warning(f"LLM_TRACE: FAIL task={task} reason=api_key_missing key={cfg['key_env']}")
            _llm_fallback_stats["totals"]["failures"] += 1
            task_stats["failures"] += 1
            return None

    for attempt in range(3):
        _attempt_t0 = time.time()
        attempt_info = {"attempt": attempt + 1, "status": "pending"}
        event["attempts"].append(attempt_info)
        try:
            if cfg["format"] == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                body = {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
                response = httpx.post(url, headers=headers, json=body, timeout=timeout)
                response.raise_for_status()
                _resp_json = response.json()
                raw = _resp_json["content"][0]["text"].strip()
            else:
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                body = {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                response = httpx.post(url, headers=headers, json=body, timeout=timeout)
                response.raise_for_status()
                _resp_json = response.json()
                raw = _resp_json["choices"][0]["message"]["content"].strip()

            raw = _strip_thinking(raw)

            _resp_model = _resp_json.get("model", model_name)

            _latency = int((time.time() - _t0) * 1000)
            _attempt_latency = int((time.time() - _attempt_t0) * 1000)

            attempt_info["status"] = "OK"
            attempt_info["latency_ms"] = _attempt_latency
            attempt_info["http_status"] = response.status_code

            event["status"] = "OK"
            event["responded_model"] = _resp_model
            event["output_chars"] = len(raw)
            event["output_tokens_est"] = len(raw) // 3
            event["latency_ms"] = _latency
            event["http_status"] = response.status_code
            event["final_attempt"] = attempt + 1

            _llm_fallback_stats["totals"]["successes"] += 1
            task_stats["successes"] += 1

            logger.info(
                f"LLM_TRACE: OK task={task} provider={provider} "
                f"model={_resp_model} latency={_latency}ms "
                f"in≈{event['input_tokens_est']}tok out≈{event['output_tokens_est']}tok "
                f"attempt={attempt+1}/3"
            )
            return raw

        except Exception as e:
            err_str = str(e)
            _attempt_latency = int((time.time() - _attempt_t0) * 1000)
            is_rate_limit = "429" in err_str
            is_retryable = is_rate_limit or any(x in err_str for x in ("500", "503", "rate"))

            _http_status = 0
            if hasattr(e, 'response') and e.response is not None:
                _http_status = getattr(e.response, 'status_code', 0)

            attempt_info["status"] = "RETRY" if attempt < 2 else "FAIL"
            attempt_info["error"] = err_str[:80]
            attempt_info["latency_ms"] = _attempt_latency
            attempt_info["http_status"] = _http_status
            attempt_info["is_rate_limit"] = is_rate_limit

            if is_rate_limit:
                retry_after = 60
                if hasattr(e, 'response') and e.response is not None:
                    retry_after = int(e.response.headers.get("retry-after", 60))
                wait = min(retry_after, 60)
                logger.warning(
                    f"LLM_TRACE: RATE_LIMITED task={task} http={_http_status} "
                    f"wait={wait}s attempt={attempt+1}/3"
                )
            else:
                wait = (2 ** attempt) if is_retryable else 0.5
                logger.warning(
                    f"LLM_TRACE: RETRY task={task} http={_http_status} "
                    f"error={err_str[:60]} wait={wait}s attempt={attempt+1}/3"
                )

            if attempt < 2:
                time.sleep(wait)
                continue

            _latency = int((time.time() - _t0) * 1000)
            event["status"] = f"FAILED"
            event["error"] = err_str[:80]
            event["latency_ms"] = _latency
            event["http_status"] = _http_status
            event["final_attempt"] = 3
            event["fallback_triggered"] = True

            _llm_fallback_stats["totals"]["failures"] += 1
            _llm_fallback_stats["totals"]["fallbacks"] += 1
            task_stats["failures"] += 1
            task_stats["fallbacks"] += 1

            if is_rate_limit:
                logger.error(
                    f"LLM_TRACE: EXHAUSTED task={task} latency={_latency}ms "
                    f"reason=rate_limit_3x — FALLBACK WILL ACTIVATE. "
                    f"Raise TOOL_RESULT_SUMMARIZE_THRESHOLD to reduce call frequency."
                )
            else:
                logger.error(
                    f"LLM_TRACE: EXHAUSTED task={task} latency={_latency}ms "
                    f"error={err_str[:60]} — FALLBACK WILL ACTIVATE"
                )
            return None


def get_llm_trace(limit: int = 100) -> dict:
    """Return structured LLM trace data for the dashboard.

    Includes recent events and aggregate fallback statistics.
    """
    with _llm_events_lock:
        events = _llm_events[:limit]
    stats = _llm_fallback_stats
    totals = stats["totals"]
    success_rate = (
        round(totals["successes"] / totals["calls"] * 100, 1)
        if totals["calls"] > 0 else 100.0
    )
    fallback_rate = (
        round(totals["fallbacks"] / totals["calls"] * 100, 1)
        if totals["calls"] > 0 else 0.0
    )
    return {
        "events": events,
        "stats": {
            "total_calls": totals["calls"],
            "successes": totals["successes"],
            "failures": totals["failures"],
            "fallbacks": totals["fallbacks"],
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
        },
        "by_task": stats["by_task"],
    }


TAG_PATTERNS: dict[str, list[str]] = {
    "#auth":         ["auth", "jwt", "token", "password", "login", "oauth", "session", "bearer"],
    "#database":     ["database", "sql", "postgres", "sqlite", "mysql", "mongodb", "orm", "migration", "schema"],
    "#api":          ["api", "route", "endpoint", "request", "response", "rest", "graphql", "fastapi"],
    "#architecture": ["structure", "folder", "module", "service", "repository", "pattern", "layer"],
    "#security":     ["security", "permission", "role", "rbac", "encrypt", "hash", "cors", "xss"],
    "#testing":      ["test", "pytest", "mock", "coverage", "fixture", "assert", "unit", "integration"],
    "#devops":       ["docker", "deploy", "ci", "cd", "prod", "kubernetes", "nginx", "env", "build"],
    "#bug":          ["bug", "error", "fix", "issue", "crash", "exception", "traceback", "fail"],
    "#performance":  ["cache", "redis", "async", "queue", "worker", "optimize", "index", "latency"],
}

def _should_pin_local(text: str) -> bool:
    """Fallback-only pin heuristic. Primary path is _smart_classify()."""
    text_lower = text.lower()
    return any(p in text_lower for p in _FALLBACK_SAVE_PIN_KEYWORDS)

CODE_INDICATORS = re.compile(r"[{}\[\]=;]|^(def |class |import |from |return |async )", re.MULTILINE)


_chroma_client: chromadb.PersistentClient | None = None
_embed_model = None
_chroma_available: bool = True
_clear_project_pending: dict[str, float] = {}
_collection_cache: dict[str, "chromadb.Collection"] = {}


_injection_history: list[dict] = []
_INJECTION_HISTORY_MAX = 50

_judge_cache: OrderedDict = OrderedDict()
_judge_lock = threading.Lock()


def _judge_cache_get(text: str) -> tuple[str, str] | None:
    """Check judge cache. Returns (verdict, source) or None."""
    key = hashlib.sha256(text[:200].encode()).hexdigest()[:16]
    with _judge_lock:
        if key in _judge_cache:
            verdict, source, ts = _judge_cache[key]
            if time.time() - ts < 3600:
                return (verdict, source)
            else:
                del _judge_cache[key]
    return None


def _judge_cache_set(text: str, verdict: str, source: str) -> None:
    """Store verdict in judge cache (max 500 entries, FIFO eviction)."""
    key = hashlib.sha256(text[:200].encode()).hexdigest()[:16]
    with _judge_lock:
        if key in _judge_cache:
            _judge_cache.move_to_end(key)
        _judge_cache[key] = (verdict, source, time.time())
        if len(_judge_cache) > 500:
            _judge_cache.popitem(last=False)


def _smart_classify(text: str) -> dict:
    """
    Unified smart classifier. Single LLM call replacing ALL hardcoded keyword checks:
    - save/skip/pin decision
    - tag extraction
    - component detection
    - session resume signal detection
    - summary generation

    Returns dict with keys: action, confidence, summary, tags, component,
    is_resume_signal, reasoning.
    Falls back to keyword heuristics only when LLM is unavailable.
    """
    cached = _judge_cache_get(text)
    if cached:
        if isinstance(cached, tuple):
            return {"action": cached[0], "confidence": 0.9, "summary": text[:120],
                    "tags": [], "component": "general", "is_resume_signal": False,
                    "reasoning": "cached", "_source": cached[1]}
        return cached

    result = None
    if _call_llm:
        raw = _call_llm(
            system=_SMART_CLASSIFIER_SYSTEM,
            user=text[:1200],
            max_tokens=120,
            timeout=4.0,
            task="smart_classify",
        )
        if raw:
            raw = _strip_thinking(raw)
            try:
                clean = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(clean)
                if result.get("action") in ("SAVE_PIN", "SAVE", "SKIP"):
                    result["_source"] = "llm"
                    _judge_cache_set(text, result["action"], "llm")
                    logger.debug(
                        f"[Classifier] {result['action']} conf={result.get('confidence',0):.2f} "
                        f"comp={result.get('component')} tags={result.get('tags')} "
                        f"reason={result.get('reasoning','')}"
                    )
                    return result
            except Exception as _pe:
                logger.warning(f"[Classifier] JSON parse failed: {_pe} | raw={raw[:100]}")

    logger.warning("[Classifier] LLM unavailable — keyword fallback active")
    text_lower = text.lower().strip()

    is_resume = any(s in text_lower for s in _FALLBACK_RESUME_SIGNALS)

    if any(text_lower.startswith(p) for p in _FALLBACK_SKIP_PREFIXES):
        action = "SKIP"
    elif any(p in text_lower for p in _FALLBACK_SAVE_PIN_KEYWORDS):
        action = "SAVE_PIN"
    elif any(p in text_lower for p in _FALLBACK_SAVE_KEYWORDS):
        action = "SAVE"
    else:
        action = "SKIP"

    tags = _extract_tags(text)
    fallback = {
        "action": action,
        "confidence": 0.5,
        "summary": text[:120],
        "tags": tags,
        "component": "general",
        "is_resume_signal": is_resume,
        "reasoning": "keyword_fallback",
        "_source": "auto_keyword",
    }
    _judge_cache_set(text, action, "auto_keyword")
    return fallback


def _call_judge(text: str) -> tuple[str, str]:
    """Compatibility shim — calls _smart_classify and returns (verdict, source) tuple."""
    result = _smart_classify(text)
    return result.get("action", "SKIP"), result.get("_source", "auto")


def _summarize_before_save(text: str) -> str:
    """
    Call the configured LLM to extract the core decision/rule from text.
    Returns a clean, concise summary (max 150 chars).
    Falls back to original text if API unavailable.
    """
    if not text or len(text) < 200:
        return text

    SUMMARIZE_SYSTEM = (
        "Technical memory extractor. Extract ONE concise decision/fact (max 150 chars). "
        "PRESERVE VERBATIM: error types (TypeError, KeyError), file paths with line numbers, "
        "function/class/variable names, test names, exact error messages. "
        "STRICT RULES: "
        "(1) Copy facts exactly as stated — never infer, combine, or relate separate facts. "
        "(2) If multiple distinct tools/technologies mentioned, pick the MOST specific one only. "
        "(3) Never write 'both X and Y' — that is always wrong. "
        "(4) Never paraphrase code references, errors, or file paths. "
        "(5) Exact terms, no fluff, original language. Output ONLY the sentence."
    )
    USER_PROMPT = f"INPUT: \"{text[:1000]}\"\nOUTPUT (one fact, no combining):"

    summary = _call_llm(
        system=SUMMARIZE_SYSTEM,
        user=USER_PROMPT,
        max_tokens=40,
        timeout=3.0,
        task="memory_summarization"
    )

    if summary:
        summary = re.sub(r"^OUTPUT:\s*", "", summary, flags=re.IGNORECASE)
        summary = summary.strip(' "')

        if 5 < len(summary) <= 200:
            logger.info(f"[Summarize] '{text[:40]}...' -> '{summary[:40]}'")
            return summary
        else:
            logger.warning(f"[Summarize] Rejected summary (len={len(summary)}): too verbose or broken.")

    return text


_INJECTED_IDS: dict[str, set[int]] = {}
_INJECTED_TS: dict[str, float] = {}
_injected_lock = threading.Lock()


def _track_injected(session_id: str, ids: list[int]) -> None:
    """Record entry IDs already injected into a session."""
    with _injected_lock:
        _INJECTED_IDS.setdefault(session_id, set()).update(ids)
        _INJECTED_TS[session_id] = time.time()
        cutoff = time.time() - 7200
        stale = [s for s, ts in list(_INJECTED_TS.items()) if ts < cutoff]
        for s in stale:
            _INJECTED_IDS.pop(s, None)
            _INJECTED_TS.pop(s, None)


def _already_injected(session_id: str, entry_id: int) -> bool:
    """Check if an entry was already injected in this session."""
    with _injected_lock:
        return entry_id in _INJECTED_IDS.get(session_id, set())


def _detect_lang(text: str) -> str:
    """Detect language of text. Returns 'fr', 'en', etc. Defaults to 'en'."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


def _extract_tags(text: str) -> list[str]:
    """Extract tags from text via keyword matching."""
    text_lower = text.lower()
    tags = []
    for tag, keywords in TAG_PATTERNS.items():
        for kw in keywords:
            if kw in text_lower:
                tags.append(tag)
                break
    return tags


def _get_embed_model():
    """Lazy-load the sentence-transformers embedding model with cache."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        cache_dir = os.path.expanduser("~/.cache/cc-memory/models")
        os.makedirs(cache_dir, exist_ok=True)
        _embed_model = SentenceTransformer(MEMORY_EMBEDDING_MODEL, cache_folder=cache_dir)
        logger.info(f" Embedding model ready: {MEMORY_EMBEDDING_MODEL}")
    return _embed_model


_encode_cache: dict[str, list[float]] = {}
_ENCODE_CACHE_MAX = 256
_encode_lock = threading.Lock()

def _encode(text: str) -> list[float]:
    """Encode text to embedding vector (with LRU cache)."""
    key = text[:500]
    with _encode_lock:
        cached = _encode_cache.get(key)
        if cached is not None:
            return cached
    emb = _get_embed_model().encode(text).tolist()
    with _encode_lock:
        if len(_encode_cache) >= _ENCODE_CACHE_MAX:
            try:
                _encode_cache.pop(next(iter(_encode_cache)))
            except StopIteration:
                pass
        _encode_cache[key] = emb
    return emb


_Q_PREFIX = re.compile(
    r"^(what(?:'s| is| are| was| were| do| does| did)?|"
    r"where(?:'s| is| are| was| were| do| does| did)?|"
    r"when(?:'s| is| are| was| were| do| does| did)?|"
    r"who(?:'s| is| are| was| were)?|"
    r"which|how(?:'s| is| are| do| does| did| much| many)?|"
    r"do|does|did|is|are|was|were|can|could|will|would|should|shall|"
    r"have|has|had)\s+",
    re.IGNORECASE,
)

def _to_declarative(query: str) -> str | None:
    """Strip question syntax to create a declarative echo of a query.
    Returns None if the query isn't a recognizable question.
    'what's my name?' → 'my name'
    'what port do we use?' → 'port we use'
    'how does auth work?' → 'auth work'
    """
    q = query.strip().rstrip("?").strip()
    m = _Q_PREFIX.match(q)
    if not m:
        return None
    decl = q[m.end():].strip()
    return decl if len(decl) >= 3 else None


def _encode_query(query: str) -> list[float]:
    """Encode a search query, blending with its declarative form for questions.
    This bridges the question↔fact embedding gap so 'what's my name?'
    lands closer to 'User's name is Ali' in vector space.
    """
    emb_q = _encode(query)
    decl = _to_declarative(query)
    if decl is None:
        return emb_q
    emb_d = _encode(decl)
    # Average the two embeddings (70% original, 30% declarative)
    blended = [(a * 0.7 + b * 0.3) for a, b in zip(emb_q, emb_d)]
    return blended


_KW_STOPWORDS = frozenset({
    "what", "where", "when", "which", "who", "how", "does", "did",
    "the", "this", "that", "with", "from", "have", "has", "was",
    "were", "been", "are", "for", "not", "but", "and", "our",
    "your", "their", "about", "into", "over", "can", "will",
    "just", "also", "than", "then", "some", "other",
})


def _get_client() -> chromadb.PersistentClient:
    """Return ChromaDB PersistentClient singleton."""
    global _chroma_client
    if _chroma_client is None:
        Path(MEMORY_CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=MEMORY_CHROMA_PATH)
        logger.info("ChromaDB initialized")
    return _chroma_client


def _get_collection(project_id: str) -> chromadb.Collection:
    """Get or create ChromaDB collection for a project (PERF 2: cached)."""
    if project_id in _collection_cache:
        return _collection_cache[project_id]
    safe_pid = re.sub(r"[^a-zA-Z0-9]", "_", project_id).strip("_")
    if len(safe_pid) < 1:
        safe_pid = "default"
    coll_name = f"memories_{safe_pid}"[:63]
    client = _get_client()
    try:
        coll = client.get_collection(coll_name)
    except Exception as e:
        err_str = str(e).lower()
        if "does not exist" in err_str or "not found" in err_str:
            logger.debug(f"[ChromaDB] Creating new collection: {coll_name}")
            coll = client.create_collection(coll_name, metadata={"hnsw:space": "cosine"})
        else:
            logger.error(f"[ChromaDB] Failed to get collection {coll_name}: {e}")
            raise
    _collection_cache[project_id] = coll
    return coll


def _chroma_ok() -> bool:
    """Check if ChromaDB is available."""
    global _chroma_available
    if not _chroma_available:
        return False
    try:
        _get_client()
        return True
    except Exception as exc:
        _chroma_available = False
        logger.warning(f"ChromaDB unavailable — SQLite fallback active: {exc}")
        return False


_thread_local = threading.local()

class _PooledConnection:
    """Wrapper that prevents closing() from destroying the actual connection."""
    __slots__ = ('_conn',)
    def __init__(self, conn):
        self._conn = conn
    def execute(self, *args, **kw):
        return self._conn.execute(*args, **kw)
    def commit(self):
        return self._conn.commit()
    def close(self):
        pass
    @property
    def row_factory(self):
        return self._conn.row_factory
    @row_factory.setter
    def row_factory(self, val):
        self._conn.row_factory = val

def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection with WAL mode (thread-local pooled)."""
    conn = getattr(_thread_local, 'sqlite_conn', None)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return _PooledConnection(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.row_factory = sqlite3.Row
    _thread_local.sqlite_conn = conn
    return _PooledConnection(conn)


def _init_db() -> None:
    """Create the memories table if it doesn't exist."""
    with closing(_get_conn()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id  TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                text        TEXT NOT NULL,
                tags        TEXT DEFAULT '[]',
                pinned      INTEGER DEFAULT 0,
                superseded  INTEGER DEFAULT 0,
                deleted     INTEGER DEFAULT 0,
                source      TEXT DEFAULT 'user',
                created_at  TEXT,
                token_count INTEGER DEFAULT 0,
                trigger     TEXT DEFAULT '',
                outcome     TEXT DEFAULT '',
                component   TEXT DEFAULT 'general'
            )
        """)
        for col in ("trigger", "outcome", "component"):
            try:
                default = "'general'" if col == "component" else "''"
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.OperationalError:
                pass
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_active
              ON memories(project_id, deleted, superseded, id DESC)
              WHERE deleted=0 AND superseded=0
        """)
        conn.commit()


try:
    _init_db()
    logger.info("SQLite memory DB initialized (WAL mode)")
except Exception as exc:
    logger.error(f"SQLite init failed: {exc}")
    MEMORY_ENABLED = False


try:
    if MEMORY_ENABLED:
        _chroma_ok()
except Exception as exc:
    logger.warning(f"ChromaDB startup check failed: {exc}")
    _chroma_available = False


def _age_label(created_at: str) -> str:
    """Cache-stable age label using integer day buckets.

    Uses calendar-day boundaries so the label stays constant within a
    session (changes only at midnight UTC).  This prevents the injected
    memory block from changing every minute.
    """
    try:
        created = datetime.fromisoformat(created_at).replace(tzinfo=UTC)
        today = datetime.now(UTC).date()
        days = (today - created.date()).days
        if days == 0:
            return "[today]"
        if days == 1:
            return "[1d]"
        if days < 7:
            return f"[{days}d]"
        weeks = days // 7
        if weeks < 5:
            return f"[{weeks}w]"
        months = days // 30
        return f"[{months}mo]"
    except Exception:
        return "[?]"


def _temporal_weight(created_at: str) -> float:
    """Temporal weight using date boundaries for cache stability.

    Uses calendar-day boundaries instead of rolling hours so the weight
    stays constant within a session (changes only at midnight UTC).
    """
    try:
        created = datetime.fromisoformat(created_at).replace(tzinfo=UTC)
        today = datetime.now(UTC).date()
        delta_days = (today - created.date()).days
        if delta_days == 0:
            return 2.0
        if delta_days < 7:
            return 1.5
        return 1.0
    except Exception:
        return 1.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_entities(text: str) -> set[str]:
    """Extract key technical terms, nouns and identifiers as entities for comparison."""
    raw_words = re.findall(r'[a-z0-9]+', text.lower())

    STOP_WORDS = {
        "the", "and", "for", "with", "this", "that", "from", "over", "instead",
        "was", "were", "been", "have", "has", "had", "are", "not", "use",
        "chosen", "using", "used", "after", "before", "each", "both", "some"
    }
    return {w for w in raw_words if w not in STOP_WORDS}


def _check_dedup(text: str, project_id: str, emb: list[float]) -> tuple[int, float] | None:
    """
    Check for duplicates using ChromaDB vector search (O(1) instead of O(n)).
    Returns (existing_id, similarity) if duplicate found, else None.
    """
    if _chroma_ok():
        try:
            coll = _get_collection(project_id)
            count = coll.count()
            if count == 0:
                return None
            res = coll.query(
                query_embeddings=[emb],
                n_results=min(5, count),
                include=["documents", "distances", "metadatas"],
            )
            for doc, dist, meta in zip(
                res.get("documents", [[]])[0],
                res.get("distances", [[]])[0],
                res.get("metadatas", [[]])[0],
            ):
                sim = max(0.0, 1.0 - dist)
                if sim > MEMORY_DEDUP_THRESHOLD:
                    entry_id = meta.get("id", -1)
                    with closing(_get_conn()) as _conn:
                        _row = _conn.execute(
                            "SELECT deleted FROM memories WHERE id=? AND project_id=?",
                            (entry_id, project_id),
                        ).fetchone()
                    if _row and _row["deleted"]:
                        logger.debug(f"[Memory] ChromaDB dedup: skipping deleted id={entry_id}")
                        continue
                    logger.info(f"[Memory] ChromaDB dedup: sim={sim:.2f} (id={entry_id})")
                    return (entry_id, sim)
            return None
        except Exception as exc:
            logger.warning(f"[Memory] ChromaDB dedup failed, trying SQLite fallback: {exc}")

    rows = []
    with closing(_get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, text FROM memories "
            "WHERE project_id=? AND deleted=0 AND superseded=0 "
            "ORDER BY id DESC LIMIT 20",
            (project_id,),
        ).fetchall()

    if not rows:
        return None

    new_entities = _extract_entities(text)
    for row in rows:
        existing_entities = _extract_entities(row["text"])
        if new_entities and existing_entities:
            overlap = len(new_entities & existing_entities) / max(len(new_entities), len(existing_entities))
            if overlap >= 0.85:
                logger.info(f"[Memory] Entity-based dedup: overlap={overlap:.2f} (id={row['id']})")
                return (row["id"], overlap)

    return None


def _is_duplicate(text: str, project_id: str) -> bool:
    """Quick boolean check: is this text semantically duplicate of recent entries?
    Uses ChromaDB query directly for speed."""
    try:
        emb = _encode(text)
        return _check_dedup(text, project_id, emb) is not None
    except Exception as exc:
        logger.warning(f"[Memory] _is_duplicate check failed: {exc}")
        return False


def _check_contradiction(text: str, project_id: str, emb: list[float], lang: str):
    """
    Detect contradictions using ChromaDB vector search + LLM judge.
    Uses ChromaDB query (O(1)) instead of re-encoding entries (O(n)).
    """
    candidates = []
    if _chroma_ok():
        try:
            coll = _get_collection(project_id)
            count = coll.count()
            if count == 0:
                return None
            res = coll.query(
                query_embeddings=[emb],
                n_results=min(10, count),
                include=["documents", "distances", "metadatas"],
            )
            for doc, dist, meta in zip(
                res.get("documents", [[]])[0],
                res.get("distances", [[]])[0],
                res.get("metadatas", [[]])[0],
            ):
                sim = max(0.0, 1.0 - dist)
                if sim >= 0.40:
                    entry_id = meta.get("id", -1)
                    with closing(_get_conn()) as _conn:
                        _row = _conn.execute(
                            "SELECT deleted FROM memories WHERE id=? AND project_id=?",
                            (entry_id, project_id),
                        ).fetchone()
                    if _row and _row["deleted"]:
                        logger.debug(f"[Contradiction] skipping deleted id={entry_id}")
                        continue
                    candidates.append((entry_id, doc, sim))
        except Exception as exc:
            logger.warning(f"[Contradiction] ChromaDB query failed: {exc}")

    if not candidates:
        with closing(_get_conn()) as conn:
            rows = conn.execute(
                "SELECT id, text FROM memories "
                "WHERE project_id=? AND deleted=0 AND superseded=0 "
                "ORDER BY id DESC LIMIT 20",
                (project_id,),
            ).fetchall()
        if not rows:
            return None
        new_entities = _extract_entities(text)
        for row in rows:
            if new_entities:
                existing_entities = _extract_entities(row["text"])
                if existing_entities:
                    overlap = len(new_entities & existing_entities) / max(len(new_entities), len(existing_entities))
                    if overlap >= 0.40:
                        candidates.append((row["id"], row["text"], overlap))

    if not candidates:
        return None

    candidates_text = "\n".join(f"[id={c[0]}] {c[1]}" for c in candidates)

    SYSTEM = (
        "You are a contradiction detector for a project memory system. "
        "Decide if the new entry SUPERSEDES any existing entry. "
        "Supersede = new entry makes an existing entry wrong or outdated. "
        "Examples that supersede: "
        "'we use FastAPI' supersedes 'we use Spring Boot'. "
        "'migrated to MongoDB' supersedes 'PostgreSQL as database'. "
        "'switched to Redis' supersedes 'using in-memory cache'. "
        "Examples that do NOT supersede: "
        "'JWT in cookies' does not supersede 'we use React'. "
        "'added Tailwind' does not supersede 'FastAPI backend'. "
        "Reply with valid JSON only. No text outside JSON. No markdown. "
        "Only mark as superseded if the new entry DIRECTLY CONTRADICTS a specific "
        "factual claim in the old entry (e.g. different technology, different port, "
        "different decision). Semantic similarity alone is NOT sufficient. "
        "When in doubt, return KEEP."
    )

    USER = (
        f"New entry: \"{text}\"\n\n"
        f"Existing entries:\n{candidates_text}\n\n"
        "Reply with exactly one of:\n"
        "{\"supersedes\": true, \"superseded_id\": <int>, \"reason\": \"<one line>\"}\n"
        "or\n"
        "{\"supersedes\": false, \"superseded_id\": null, \"reason\": \"<one line>\"}"
    )

    try:
        raw = _call_llm(system=SYSTEM, user=USER, max_tokens=80, timeout=12.0)
        if raw:
            result = json.loads(raw)

            if result.get("supersedes") and result.get("superseded_id"):
                sid = int(result["superseded_id"])
                with closing(_get_conn()) as conn:
                    old = conn.execute(
                        "SELECT text FROM memories WHERE id=?", (sid,)
                    ).fetchone()
                    conn.execute(
                        "UPDATE memories SET superseded=1 WHERE id=?", (sid,)
                    )
                    conn.commit()
                logger.info(
                    f"[Contradiction] superseded id={sid} — {result.get('reason')}"
                )
                return (sid, old["text"] if old else "")

    except Exception as exc:
        logger.warning(f"[Contradiction] LLM judge failed: {exc}")

    text_lower = text.lower()
    CONTRADICTION_KEYWORDS = ["instead of", "rather than", "changed to", "replaced by", "now runs on"]
    if any(k in text_lower for k in CONTRADICTION_KEYWORDS):
        for row in candidates:
            if text_lower.strip() == row[1].lower().strip():
                continue
            if row[2] >= 0.60:
                sid = row[0]
                with closing(_get_conn()) as conn:
                    old = conn.execute("SELECT text FROM memories WHERE id=?", (sid,)).fetchone()
                    conn.execute("UPDATE memories SET superseded=1 WHERE id=?", (sid,))
                    conn.commit()
                logger.info(f"[Contradiction] Fallback detected: superseded id={sid}")
                return (sid, old["text"] if old else "")

    return None


def _save(
    project_id: str,
    session_id: str,
    text: str,
    pinned: bool = False,
    source: str = "user",
    check_dedup: bool = True,
    is_manual: bool = True,
    trigger: str = "",
    outcome: str = "",
    component: str = "general",
) -> int | dict:
    """
    Persist to SQLite + ChromaDB.
    Returns row_id on success, or dict with error info on dedup/contradiction.
    """
    tags = _extract_tags(text)
    tags_json = json.dumps(tags)
    token_count = len(text) // 3
    lang = _detect_lang(text)

    if not pinned:
        pinned = _should_pin_local(text)

    emb = None
    try:
        emb = _encode(text)
    except Exception as exc:
        logger.warning(f"[Memory] Embedding failed, skipping dedup: {exc}")

    contradiction_info = None
    if emb is not None:
        contradiction_info = _check_contradiction(text, project_id, emb, lang)

    if check_dedup and emb is not None and not contradiction_info:
        dup = _check_dedup(text, project_id, emb)
        if dup:
            dup_id, sim = dup
            if is_manual:
                return {
                    "error": "duplicate",
                    "id": dup_id,
                    "similarity": round(sim, 3),
                }
            else:
                logger.debug(
                    f"[Memory] Autosave dedup: skipped (id={dup_id}, sim={sim:.3f})"
                )
                return -1

    row_id = -1
    now_utc = datetime.now(UTC).isoformat()
    logger.info(f"[Memory] _save: text='{text[:60]}...' project={project_id} pinned={pinned}")
    for attempt in range(3):
        try:
            with closing(_get_conn()) as conn:
                cur = conn.execute(
                    "INSERT INTO memories "
                    "(project_id, session_id, text, tags, pinned, superseded, deleted, "
                    " source, created_at, token_count, trigger, outcome, component) "
                    "VALUES (?,?,?,?,?,0,0,?,?,?,?,?,?)",
                    (project_id, session_id, text, tags_json, int(pinned),
                     source, now_utc, token_count, trigger, outcome, component),
                )
                row_id = cur.lastrowid
                conn.commit()
            logger.info(f"[Memory] _save: SUCCESS id={row_id}")
            break
        except sqlite3.OperationalError as exc:
            wait = [0.1, 0.3, 0.9][attempt]
            logger.warning(
                f"[Memory] SQLite write retry {attempt+1}/3 "
                f"(wait={wait}s): {exc}"
            )
            time.sleep(wait)
            if attempt == 2:
                logger.error(f"[Memory] SQLite write failed after 3 retries: {exc}")
                return -1

    if _chroma_ok() and emb is not None and row_id > 0:
        try:
            _get_collection(project_id).add(
                ids=[str(row_id)],
                embeddings=[emb],
                metadatas=[{
                    "id": row_id,
                    "project_id": project_id,
                    "session_id": session_id,
                    "pinned": int(pinned),
                    "created_at": now_utc,
                    "token_count": token_count,
                    "tags": tags_json,
                }],
                documents=[text],
            )
            logger.info(f"[Memory] _save: ChromaDB indexed id={row_id}")
        except Exception as exc:
            logger.warning(f"[Memory] ChromaDB index failed for id={row_id}: {exc}")

    if row_id > 0:
        try:
            from memory.graph import (
                build_edges_for_entry,
                detect_component,
                ensure_schema,
                get_recent_entry_meta,
            )
            ensure_schema(MEMORY_DB_PATH)
            _comp = component
            if _comp == "general" or not _comp:
                _comp = detect_component(text, trigger)
            
            _recent = get_recent_entry_meta(project_id, MEMORY_DB_PATH, limit=15)
            build_edges_for_entry(row_id, project_id, session_id, _comp, MEMORY_DB_PATH, _recent)
            try:
                if component == "general" or not component:
                    with closing(_get_conn()) as _cc:
                        _cc.execute("UPDATE memories SET component=? WHERE id=?", (_comp, row_id))
                        _cc.commit()
            except Exception:
                pass
        except Exception as _ge:
            logger.debug(f"[Graph] Edge build failed (non-fatal): {_ge}")
        try:
            from api.hot_cache import hot_cache as _hc
            from api.hot_cache import notifications as _notif
            _hc.invalidate(project_id)
            _notif.push("save", f"Saved [{project_id}]: {text[:60]}...", project_id)
        except Exception:
            pass

    if is_manual and contradiction_info:
        return {
            "id": row_id,
            "tags": tags,
            "pinned": pinned,
            "contradiction": {
                "superseded_id": contradiction_info[0],
                "old_text": contradiction_info[1][:100],
            },
        }

    if is_manual:
        return {"id": row_id, "tags": tags, "pinned": pinned}

    return row_id


def _search(
    query: str,
    project_id: str,
    session_id: str = "default",
    top_k: int = MEMORY_TOP_K,
    _precomputed_emb: list[float] | None = None,
) -> list[dict]:
    """
    Semantic search with scoring formula.
    Returns list of dicts: {id, text, score, age, session_id, tags, pinned}.
    Accepts _precomputed_emb to avoid re-encoding the same query.
    """
    if not _chroma_ok():
        return _search_fallback(query, project_id, top_k)

    try:
        emb = _precomputed_emb if _precomputed_emb is not None else _encode_query(query)

        coll = _get_collection(project_id)
        count = coll.count()
        if count == 0:
            return []
        res = coll.query(
            query_embeddings=[emb],
            n_results=min(top_k * 2, count),
            include=["documents", "distances", "metadatas"],
        )
        all_results = list(zip(
            res.get("documents", [[]])[0],
            res.get("distances", [[]])[0],
            res.get("metadatas", [[]])[0],
        ))

        confidence_map = {
            "auto":         1.0,
            "auto_nim":     0.90,
            "auto_keyword": 0.40,
            "user":         1.0,
            "reduce":       0.85,
        }

        query_words = {w.lower() for w in re.findall(r"\w{3,}", query)} - _KW_STOPWORDS

        scored = []
        for doc, dist, meta in all_results:
            semantic_sim = max(0.0, 1.0 - dist)
            t_weight = _temporal_weight(meta.get("created_at", ""))
            p_boost = 1.5 if meta.get("pinned", 0) else 1.0
            confidence = confidence_map.get(meta.get("source", "auto"), 0.7)

            # Keyword overlap boost: bridges question↔fact semantic gap
            if query_words:
                doc_words = {w.lower() for w in re.findall(r"\w{3,}", doc)} - _KW_STOPWORDS
                overlap = len(query_words & doc_words) / len(query_words)
                kw_boost = overlap * 0.15
            else:
                kw_boost = 0.0

            final_score = semantic_sim * t_weight * p_boost * confidence + kw_boost

            entry_id = meta.get("id", -1)

            scored.append({
                "id": entry_id,
                "text": doc,
                "score": round(final_score, 3),
                "semantic_sim": round(semantic_sim, 3),
                "age": _age_label(meta.get("created_at", "")),
                "session_id": meta.get("session_id", "?"),
                "tags": meta.get("tags", "[]"),
                "pinned": bool(meta.get("pinned", 0)),
                "source": meta.get("source", "auto"),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    except Exception as exc:
        logger.warning(f"[Memory] Search failed: {exc}")
        return _search_fallback(query, project_id, top_k)


def _search_fallback(query: str, project_id: str, top_k: int) -> list[dict]:
    """SQLite LIKE fallback when ChromaDB is unavailable."""
    try:
        keywords = query.split()[:5]
        conditions = " OR ".join(["text LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]

        rows = []
        with closing(_get_conn()) as conn:
            rows = conn.execute(
                f"SELECT id, text, tags, pinned, session_id, created_at FROM memories "
                f"WHERE project_id=? AND deleted=0 AND superseded=0 AND ({conditions}) "
                f"ORDER BY id DESC LIMIT ?",
                [project_id] + params + [top_k],
            ).fetchall()

        return [
            {
                "text": r["text"],
                "score": 0.5,
                "age": _age_label(r["created_at"]),
                "session_id": r["session_id"],
                "tags": r["tags"],
                "pinned": bool(r["pinned"]),
                "degraded": True,
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"[Memory] SQLite fallback search failed: {exc}")
        return []


SESSION_RESUME_SIGNALS = [
    "continue", "reprendre", "where were we", "on était où",
    "back to", "last time", "la dernière fois", "suite de",
    "picking up", "recap", "résumé", "what did we", "on avait",
]


def _search_with_tiers(
    query: str,
    project_id: str,
    session_id: str,
    top_k: int = MEMORY_TOP_K,
) -> list[dict]:
    """
    3-tier search with component-aware routing + graph BFS re-ranking.

    Pipeline:
      0. Detect query component (auth/api/db/ui/…) — free, regex-only
      1. Tier 1: current session ChromaDB search
      2. Graph BFS re-rank: seed tier1 → expand through edges → component bonus
      3. Tier 2: all sessions fallback (when tier1 < 3 results)

    Component routing boosts entries whose partition matches the query.
    E.g. “fix login bug” → auth partition entries get +15% score.
    All operations use local embedding model + in-RAM graph index.
    Zero external API calls.
    """
    force_tier2 = any(s in query.lower() for s in _FALLBACK_RESUME_SIGNALS)
    TIER1_MIN = 3

    try:
        from memory.graph import detect_component, get_priority_components, graph_search
        query_component = detect_component(query)
        priority_components = get_priority_components(query)
        _graph_available = True
    except Exception:
        query_component = "general"
        priority_components = ["general"]
        _graph_available = False

    shared_emb = _encode_query(query)

    tier1_hits = _search(query, project_id, session_id, top_k, _precomputed_emb=shared_emb)
    for h in tier1_hits:
        h["score"] = round(min(1.0, h["score"] * 1.5), 3)
        h["tier"] = 1
        entry_comp = h.get("component", "general")
        if entry_comp in priority_components and entry_comp != "general":
            h["score"] = round(min(1.0, h["score"] * 1.15), 3)
            h["_component_boosted"] = True

    if _graph_available and tier1_hits:
        try:
            graph_hits = graph_search(
                seed_results=tier1_hits,
                project_id=project_id,
                query_component=query_component,
                db_path=MEMORY_DB_PATH,
                max_hops=2,
                max_results=top_k * 2,
            )
            if graph_hits:
                graph_ids = {h.get("id"): h for h in graph_hits if h.get("id", -1) >= 0}
                for h in tier1_hits:
                    eid = h.get("id", -1)
                    if eid >= 0 and eid in graph_ids:
                        gs = graph_ids[eid].get("_final_score", 0)
                        if gs > h["score"]:
                            h["score"] = round(gs, 3)
                tier1_ids = {h.get("id") for h in tier1_hits}
                for h in graph_hits:
                    if h.get("_via_graph") and h.get("id") not in tier1_ids:
                        h["tier"] = 1
                        tier1_hits.append(h)
                tier1_hits.sort(key=lambda x: x["score"], reverse=True)
                tier1_hits = tier1_hits[:top_k]
                logger.debug(
                    f"[Search] Graph BFS: component={query_component} "
                    f"priority={priority_components} results={len(tier1_hits)}"
                )
        except Exception as _ge:
            logger.debug(f"[Search] Graph BFS skipped: {_ge}")

    if len(tier1_hits) >= TIER1_MIN and not force_tier2:
        return tier1_hits[:top_k]

    tier2_hits = []
    try:
        if _chroma_ok():
            coll = _get_collection(project_id)
            count = coll.count()
            if count > 0:
                res = coll.query(
                    query_embeddings=[shared_emb],
                    n_results=min(top_k * 3, count),
                    include=["documents", "distances", "metadatas"],
                )
                documents = res.get("documents", [[]])[0]
                distances = res.get("distances", [[]])[0]
                metadatas = res.get("metadatas", [[]])[0]

                _t2_query_words = {w.lower() for w in re.findall(r"\w{3,}", query)} - _KW_STOPWORDS
                for i in range(len(documents)):
                    doc = documents[i]
                    dist = distances[i]
                    meta = metadatas[i]
                    if not meta:
                        continue
                    semantic_sim = max(0.0, 1.0 - dist)
                    # Keyword overlap boost (same as tier 1)
                    if _t2_query_words:
                        _t2_doc_words = {w.lower() for w in re.findall(r"\w{3,}", doc)} - _KW_STOPWORDS
                        _t2_overlap = len(_t2_query_words & _t2_doc_words) / len(_t2_query_words)
                        _t2_kw_boost = _t2_overlap * 0.15
                    else:
                        _t2_kw_boost = 0.0
                    if semantic_sim + _t2_kw_boost < 0.60:
                        continue
                    if meta.get("session_id") == session_id:
                        continue
                    created_at = str(meta.get("created_at", ""))
                    t_weight  = _temporal_weight(created_at)
                    p_boost   = 1.5 if meta.get("pinned", 0) else 1.0
                    score = round(min(1.0, (semantic_sim + _t2_kw_boost) * t_weight * p_boost * 1.2), 3)
                    tier2_hits.append({
                        "id":         meta.get("id", -1),
                        "text":       doc,
                        "score":      score,
                        "semantic_sim": round(semantic_sim, 3),
                        "age":        _age_label(meta.get("created_at", "")),
                        "session_id": meta.get("session_id", "?"),
                        "tags":       meta.get("tags", "[]"),
                        "pinned":     bool(meta.get("pinned", 0)),
                        "tier":       2,
                        "past_session": True,
                    })
    except Exception as exc:
        logger.warning(f"[Memory] Tier2 search failed: {exc}")

    seen_ids: set[int] = {h["id"] for h in tier1_hits if h.get("id") is not None}
    merged = list(tier1_hits)
    for h in tier2_hits:
        hid = h.get("id")
        if hid not in seen_ids:
            merged.append(h)
            if hid is not None:
                seen_ids.add(hid)

    merged.sort(key=lambda x: x["score"], reverse=True)

    if force_tier2:
        logger.info(f"[Memory] Session resume detected — tier2 forced, {len(tier2_hits)} past entries surfaced")

    return merged[:top_k]


def _soft_delete(entry_id: int, project_id: str) -> bool:
    """Soft-delete an entry (SQLite + ChromaDB)."""
    affected = 0
    with closing(_get_conn()) as conn:
        cur = conn.execute(
            "UPDATE memories SET deleted=1 WHERE id=? AND project_id=?",
            (entry_id, project_id),
        )
        affected = cur.rowcount
        conn.commit()

    if _chroma_ok():
        for _attempt in range(3):
            try:
                _get_collection(project_id).delete(ids=[str(entry_id)])
                break
            except Exception as exc:
                logger.warning(f"[Memory] ChromaDB delete attempt {_attempt+1} failed id={entry_id}: {exc}")
        else:
            logger.error(f"[Memory] ChromaDB delete PERMANENTLY failed id={entry_id} after 3 attempts")

    return affected > 0


def _get_stats(project_id: str, session_id: str) -> dict:
    """Get memory statistics (zero API calls). PERF 8: consolidated query."""
    try:
        with closing(_get_conn()) as conn:
            stats_row = conn.execute("""
                SELECT
                    SUM(CASE WHEN deleted=0 AND superseded=0 THEN 1 ELSE 0 END)        AS active,
                    SUM(CASE WHEN deleted=0 AND pinned=1     THEN 1 ELSE 0 END)        AS pinned,
                    SUM(CASE WHEN source='reduce'            THEN 1 ELSE 0 END)        AS compressed,
                    COALESCE(SUM(CASE WHEN deleted=0 AND superseded=0
                                  THEN token_count ELSE 0 END), 0)                     AS total_tokens
                FROM memories WHERE project_id=?
            """, (project_id,)).fetchone()
            active = stats_row[0] or 0
            pinned = stats_row[1] or 0
            compressed = stats_row[2] or 0
            total_tokens = stats_row[3] or 0

            last_save = conn.execute(
                "SELECT created_at, text FROM memories "
                "WHERE project_id=? AND deleted=0 AND superseded=0 "
                "ORDER BY created_at DESC LIMIT 1",
                (project_id,),
            ).fetchone()

            session_history = conn.execute(
                "SELECT session_id, COUNT(*), MIN(created_at) "
                "FROM memories "
                "WHERE project_id=? AND deleted=0 "
                "GROUP BY session_id "
                "ORDER BY MIN(created_at) DESC",
                (project_id,),
            ).fetchall()

            try:
                real_msg_rows = conn.execute(
                    "SELECT session_id, msg_count FROM _session_msg_counts "
                    "WHERE project_id=?",
                    (project_id,),
                ).fetchall()
                real_msg_counts = {r[0]: r[1] for r in real_msg_rows}
            except Exception:
                real_msg_counts = {}

            all_tags_raw = conn.execute(
                "SELECT tags FROM memories "
                "WHERE project_id=? AND deleted=0 AND superseded=0",
                (project_id,),
            ).fetchall()

        tag_counts: dict[str, int] = {}
        for row in all_tags_raw:
            try:
                tags = json.loads(row[0]) if row[0] else []
                for t in tags:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            except Exception:
                pass

        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:5]
        top_tags_str = " ".join(f"{t}({c})" for t, c in top_tags) if top_tags else "—"

        last_save_text = ""
        last_save_age = "—"
        if last_save:
            last_save_age = _age_label(last_save[0])
            last_save_text = last_save[1][:60] + "..." if len(last_save[1]) > 60 else last_save[1]

        chroma_size = "N/A"
        try:
            chroma_path = Path(MEMORY_CHROMA_PATH)
            if chroma_path.exists():
                total_size = sum(f.stat().st_size for f in chroma_path.rglob("*") if f.is_file())
                chroma_size = f"{total_size / (1024*1024):.1f} MB"
        except Exception:
            pass

        estimated_inject = min(total_tokens, MEMORY_TOKEN_BUDGET)
        pct = int((estimated_inject / MEMORY_TOKEN_BUDGET) * 100) if MEMORY_TOKEN_BUDGET > 0 else 0

        return {
            "active": active,
            "pinned": pinned,
            "compressed": compressed,
            "tokens_injected": estimated_inject,
            "token_budget": MEMORY_TOKEN_BUDGET,
            "budget_pct": pct,
            "last_save_age": last_save_age,
            "last_save_text": last_save_text,
            "top_tags": top_tags_str,
            "chroma_size": chroma_size,
            "session_history": session_history,
            "real_msg_counts": real_msg_counts,
        }
    except Exception as exc:
        logger.error(f"[Memory] Stats failed: {exc}")
        return {}


_ACON_STRIP_SYSTEM = (
    "Remove filler words only. Keep all facts. Output the compressed text, nothing else."
)

_ACON_COMPRESS_SYSTEM = (
    "Rewrite in 1 dense line. "
    "MUST PRESERVE: variable names, function names, class names, file paths, "
    "line numbers, error types, error messages, import paths, test names, "
    "and any decision or conclusion reached. "
    "Summarize prose aggressively. Never summarize or paraphrase code, errors, or file references. "
    "Output the compressed line only."
)

_ACON_DISTILL_SYSTEM = (
    "These entries are about the same topic. Merge into 1 dense line. "
    "Keep all distinct facts. Output the merged line only."
)

_ACON_ENABLED = os.getenv("ACON_ENABLED", "true").lower() != "false"
_ACON_PRESSURE_THRESHOLD = float(os.getenv("ACON_PRESSURE_THRESHOLD", "0.85"))


def _acon_estimate_tokens(entries: list[dict]) -> int:
    return sum(len(e["text"]) // 3 for e in entries)


def _acon_strip_filler(text: str) -> str:
    """Level 1: smart filler removal via local LLM, regex fallback.
    
    Replaces the hardcoded filler list with a direct LLM instruction.
    The model knows far more filler patterns than any static list.
    """
    if _call_llm and len(text) > 60:
        stripped = _call_llm(
            system="Remove all filler/hedge words. Keep only technical facts. Output cleaned text only.",
            user=text[:800],
            max_tokens=len(text) // 3 + 20,
            timeout=2.5,
            task="acon_strip",
        )
        if stripped and 0.4 < len(stripped) / len(text) < 1.0:
            return stripped.strip()
    result = re.sub(r"\b(Note that|Please note|Basically|Essentially|As mentioned|In order to)\b[,:]?\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r" {2,}", " ", result).strip()


def _acon_compress_entry(entry: dict) -> dict:
    """Level 2: compress a single entry to 1 line via local LLM."""
    compressed = _call_llm(
        system=_ACON_COMPRESS_SYSTEM,
        user=entry["text"],
        max_tokens=60,
        timeout=3.0,
        task="acon_compress",
    )
    if compressed and len(compressed) < len(entry["text"]) * 0.9:
        return {**entry, "text": compressed, "_acon_compressed": True}
    return entry


def _acon_distill_group(entries: list[dict]) -> dict | None:
    """Level 3: merge a group of semantically close entries into one via local LLM."""
    if len(entries) < 2:
        return None
    combined = "\n".join(f"- {e['text']}" for e in entries)
    distilled = _call_llm(
        system=_ACON_DISTILL_SYSTEM,
        user=combined,
        max_tokens=80,
        timeout=4.0,
        task="acon_distill",
    )
    if not distilled:
        return None
    best = max(entries, key=lambda e: e.get("score", 0))
    return {
        **best,
        "text": distilled,
        "_acon_distilled": True,
        "_acon_source_count": len(entries),
    }


def _acon_group_by_component(entries: list[dict]) -> dict[str, list[dict]]:
    """Group entries by component for level-3 distillation."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        comp = e.get("component", "general")
        groups.setdefault(comp, []).append(e)
    return groups


def _acon_strip_filler_batch(entries: list[dict]) -> list[dict]:
    """
    PERF 4: Batch filler removal — one LLM call for all entries instead of N serial calls.
    Fallback: return entries unchanged if LLM unavailable or parse fails.
    """
    if not _call_llm or not entries:
        return entries

    numbered = "\n".join(f"[{i}] {e['text'][:500]}" for i, e in enumerate(entries))

    result = _call_llm(
        system=(
            "You receive a numbered list of memory entries. "
            "Remove all filler/hedge words from each. Keep only technical facts. "
            "Return ONLY a JSON array of cleaned strings, same count and order. "
            'Example input: "[0] Note that the auth uses JWT\n[1] Basically it runs on port 4000"\n'
            'Example output: ["auth uses JWT", "runs on port 4000"]'
        ),
        user=numbered,
        max_tokens=sum(len(e["text"]) // 3 + 20 for e in entries),
        timeout=6.0,
        task="acon_strip_batch",
    )

    if not result:
        return entries

    try:
        import json as _json
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        cleaned = _json.loads(clean.strip())
        if isinstance(cleaned, list) and len(cleaned) == len(entries):
            out = []
            for e, c in zip(entries, cleaned):
                if c and isinstance(c, str) and 0.4 < len(c) / max(len(e["text"]), 1) < 1.0:
                    out.append({**e, "text": c.strip()})
                else:
                    out.append(e)
            return out
    except Exception as ex:
        logger.debug(f"[ACON] Batch strip parse failed: {ex} — falling back to serial")

    return entries


def _acon_prune(entries: list[dict], budget: int) -> list[dict]:
    """
    ACON hierarchical pruning. Tries 3 levels in order until entries fit budget.
    Returns pruned entry list. Never calls external API — local LLM only.

    Level 1: Strip filler (regex, free)
    Level 2: Compress each entry to 1 line (1 LLM call per entry)
    Level 3: Distill groups of same-component entries (1 LLM call per group)
    """
    if not _ACON_ENABLED or not entries:
        return entries

    current_tokens = _acon_estimate_tokens(entries)
    pressure = current_tokens / budget if budget > 0 else 0

    if pressure < _ACON_PRESSURE_THRESHOLD:
        return entries

    logger.info(
        f"[ACON] Budget pressure {pressure:.0%} ({current_tokens}/{budget} tokens) — pruning"
    )

    pruned = _acon_strip_filler_batch(entries)

    current_tokens = _acon_estimate_tokens(pruned)
    if current_tokens <= budget:
        logger.info(f"[ACON] Level 1 sufficient: {current_tokens}/{budget} tokens")
        return pruned

    if _call_llm:
        compressed = []
        for e in pruned:
            if not e.get("pinned"):
                compressed.append(_acon_compress_entry(e))
            else:
                compressed.append(e)
        pruned = compressed

    current_tokens = _acon_estimate_tokens(pruned)
    if current_tokens <= budget:
        saved = sum(1 for e in pruned if e.get("_acon_compressed"))
        logger.info(f"[ACON] Level 2 sufficient: compressed {saved} entries, {current_tokens}/{budget} tokens")
        return pruned

    if _call_llm:
        groups = _acon_group_by_component([e for e in pruned if not e.get("pinned")])
        pinned = [e for e in pruned if e.get("pinned")]
        distilled_entries = []
        distill_count = 0

        for comp, group in groups.items():
            if len(group) >= 2:
                merged = _acon_distill_group(group)
                if merged:
                    distilled_entries.append(merged)
                    distill_count += len(group)
                    continue
            distilled_entries.extend(group)

        pruned = pinned + distilled_entries

    current_tokens = _acon_estimate_tokens(pruned)
    logger.info(
        f"[ACON] Level 3 complete: {current_tokens}/{budget} tokens "
        f"({len(entries)}→{len(pruned)} entries)"
    )
    return pruned


def _llm_select_and_build(
    query: str,
    candidates: list[dict],
    budget: int = MEMORY_TOKEN_BUDGET,
    project_id: str = "default",
    session_id: str = "default",
) -> str:
    """
    Rerank candidates using local LLM to pick semantically relevant ones.
    Builds a topic-grouped injection block directly.
    """
    if not LLM_RERANK_ENABLED or not candidates:
        return _build_memory_block(candidates[:MEMORY_TOP_K], budget, project_id, session_id)

    candidates_text = []
    for i, c in enumerate(candidates):
        cid = c.get("id", i)
        text = c.get("text", "")
        sample = text[:300]
        candidates_text.append(f"[{i}] {sample}")

    prompt_user = (
        f"Query: \"{query}\"\n\n"
        f"Candidates:\n" + "\n".join(candidates_text) + "\n\n"
        "Instructions:\n"
        "1. Identify entries that contain specific technical details, decisions, or context TRULY RELEVANT to the query.\n"
        "2. Exclude topically unrelated entries even if they share keywords.\n"
        "3. Group the selected entries by topic (e.g. 'auth:', 'api:', 'infra:').\n"
        "4. Return a valid JSON object: {\"selected_indices\": [0, 2...], \"grouped_context\": \"topic1: text...\\ntopic2: text...\"}\n"
        "5. Output ONLY the raw JSON block."
    )

    try:
        raw_json = _call_llm(
            system="You are a context relevance judge. Your job is to select and group relevant memories.",
            user=prompt_user,
            max_tokens=800,
            task="rerank_selection"
        )
        if not raw_json:
            raise ValueError("Empty LLM response")

        clean_json = raw_json.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]

        data = json.loads(clean_json.strip())
        indices = data.get("selected_indices", [])
        grouped_text = data.get("grouped_context", "").strip()

        if not grouped_text or not indices:
            logger.debug(f"[LLM_Rerank] No relevant entries selected for query: {query[:50]}...")
            return ""

        valid_indices = [idx for idx in indices if 0 <= idx < len(candidates)]
        if not valid_indices:
             return ""

        logger.info(
            f"LLM_RERANK: selected {len(valid_indices)}/{len(candidates)} entries "
            f"for query='{query[:50]}...' session={session_id}"
        )

        return (
            f'<memory_context project="{project_id}" reranked="true" entries="{len(valid_indices)}">\n'
            f"{grouped_text}\n"
            f"</memory_context_end>"
        )

    except Exception as exc:
        logger.warning(f"[LLM_Rerank] Failed, falling back to cosine: {exc}")
        return _build_memory_block(candidates[:MEMORY_TOP_K], budget, project_id, session_id)


def _build_memory_block(
    entries: list[dict],
    budget: int = MEMORY_TOKEN_BUDGET,
    project_id: str = "default",
    session_id: str = "default",
) -> str:
    """Build <memory_context> block with split budget + pruning."""
    _INJECTION_BUDGET = int(os.getenv("INJECTION_BUDGET", "3000"))
    _INJECTION_TTL_DAYS = int(os.getenv("INJECTION_TTL_DAYS", "14"))

    layer2_entries = []
    regular_entries = []
    for e in entries:
        src = (e.get("source") or "")
        if src in ("smart_compact", "session_end"):
            layer2_entries.append(e)
        else:
            regular_entries.append(e)

    if layer2_entries:
        import time as _ttl_time
        _now = _ttl_time.time()
        try:
            _last_access_key = f"_last_project_access_{project_id}"
            _newest = max(
                (e.get("created_at", "") for e in layer2_entries),
                default=""
            )
            if _newest:
                from datetime import datetime
                try:
                    _created = datetime.fromisoformat(_newest.replace("Z", "+00:00"))
                    _age_days = (_now - _created.timestamp()) / 86400
                    if _age_days > _INJECTION_TTL_DAYS:
                        layer2_entries = [
                            e for e in layer2_entries
                            if float(e.get("score", 0)) >= 0.70
                        ]
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass

        l2_tokens = sum(len((e.get("text") or "")) // 3 for e in layer2_entries)
        if l2_tokens > _INJECTION_BUDGET and len(layer2_entries) > 2:
            layer2_entries.sort(
                key=lambda e: e.get("created_at", ""), reverse=True
            )
            keep_recent = layer2_entries[:2]
            to_merge = layer2_entries[2:]

            merge_parts = []
            _merge_tokens = 0
            for e in to_merge:
                txt = (e.get("text") or "")[:500]
                _merge_tokens += len(txt) // 3
                if _merge_tokens > 4000:
                    break
                merge_parts.append(txt)

            if merge_parts:
                merge_input = "\n".join(merge_parts)
                try:
                    merged = _call_llm(
                        system="Merge these session summaries into one concise summary. Keep all file paths, decisions, and technical details. Max 100 words.",
                        user=merge_input,
                        max_tokens=150,
                        timeout=4.0,
                        task="accumulation_merge",
                    )
                    if merged:
                        merged_entry = {
                            "text": f"[Merged sessions] {merged.strip()}",
                            "score": 0.75,
                            "source": "smart_compact",
                            "pinned": False,
                            "age": "",
                            "tags": "[]",
                        }
                        layer2_entries = keep_recent + [merged_entry]
                    else:
                        layer2_entries = keep_recent
                except Exception:
                    layer2_entries = keep_recent

    entries = regular_entries + layer2_entries

    total_entry_tokens = sum(len((e.get("text") or ""))//3 for e in entries)
    if total_entry_tokens > int(budget * 0.85) and len(entries) > 1:
        scored = sorted(entries, key=lambda e: float(e.get("score") or 0), reverse=True)
        kept = []
        used = 0
        for e in scored:
            tc = len((e.get("text") or ""))//3
            if used + tc <= budget:
                kept.append(e)
                used += tc
            elif e.get("pinned"):
                kept.append(e)
                used += tc
        entries = kept

    BUDGET_PINNED  = int(budget * 0.30)
    BUDGET_CURRENT = int(budget * 0.50)
    BUDGET_PAST    = int(budget * 0.20)

    def _format_entry(entry: dict) -> str:
        tags_str = ""
        try:
            tags = json.loads(entry.get("tags", "[]")) if isinstance(entry.get("tags"), str) else entry.get("tags", [])
            tags_str = " ".join(tags) + " " if tags else ""
        except Exception:
            pass
        age = entry.get("age", "")
        return f"{age} {tags_str}{entry['text']}" if age else f"{tags_str}{entry['text']}"

    lines = []
    total_tokens = 0
    past_count = 0
    used_pinned = 0
    used_current = 0
    used_past = 0

    _dropped_pinned = 0
    for entry in entries:
        if not entry.get("pinned"):
            continue
        tc = len(entry["text"]) // 3
        if used_pinned + tc > BUDGET_PINNED:
            _dropped_pinned += 1
            continue
        lines.append(_format_entry(entry))
        used_pinned += tc
        total_tokens += tc
    if _dropped_pinned:
        logger.warning(
            f"PINNED_OVERFLOW: {_dropped_pinned} pinned entry/entries dropped — "
            f"budget_pinned={BUDGET_PINNED} tokens exhausted. "
            f"Raise MEMORY_TOKEN_BUDGET in .env to fix (current={budget})."
        )

    for entry in entries:
        if entry.get("pinned"):
            continue
        if entry.get("past_session") or entry.get("tier") == 2:
            continue
        tc = len(entry["text"]) // 3
        if used_current + tc > BUDGET_CURRENT:
            continue
        lines.append(_format_entry(entry))
        used_current += tc
        total_tokens += tc

    for entry in entries:
        if entry.get("pinned"):
            continue
        if not (entry.get("past_session") or entry.get("tier") == 2):
            continue
        tc = len(entry["text"]) // 3
        if used_past + tc > BUDGET_PAST:
            continue
        lines.append(_format_entry(entry))
        used_past += tc
        total_tokens += tc
        past_count += 1

    if not lines:
        return ""

    return (
        f'<memory_context project="{project_id}" entries="{len(lines)}">\n'
        + "\n".join(lines)
        + "\n</memory_context_end>"
    )


def _build_tiered_block(
    candidates: list[dict],
    query: str,
    project_id: str,
    session_id: str,
    already_seen_ids: set,
) -> tuple[str, set]:
    """
    Score-tiered memory injection block builder.
    Returns (block_text, newly_injected_ids).

    FULL  (score >= TIER_FULL_THRESHOLD): complete text injected
    HEAD  (score >= TIER_HEAD_THRESHOLD): one-line headline only
    SKIP  (id in already_seen_ids):       suppressed, already in context window
    """
    full_entries: list[dict] = []
    head_entries: list[dict] = []
    skipped = 0
    newly_seen: set = set()
    used_tokens = 0

    for entry in candidates:
        eid = entry.get("id")
        score = float(entry.get("score") or 0.0)
        text = (entry.get("text") or "").strip()

        if eid is not None and eid in already_seen_ids:
            skipped += 1
            continue

        tc = len(text) // 3

        if score >= TIER_FULL_THRESHOLD:
            if used_tokens + tc > MEMORY_TOKEN_BUDGET:
                headline = text.split(".")[0][:80].strip()
                head_tc = len(headline) // 3
                if used_tokens + head_tc <= MEMORY_TOKEN_BUDGET:
                    head_entries.append({"id": eid, "score": score, "headline": headline})
                    used_tokens += head_tc
                    if eid is not None:
                        newly_seen.add(eid)
                continue
            full_entries.append(entry)
            used_tokens += tc
            if eid is not None:
                newly_seen.add(eid)
        elif score >= TIER_HEAD_THRESHOLD:
            headline = text.split(".")[0][:80].strip()
            head_tc = len(headline) // 3
            if used_tokens + head_tc > MEMORY_TOKEN_BUDGET:
                continue
            head_entries.append({"id": eid, "score": score, "headline": headline})
            used_tokens += head_tc
            if eid is not None:
                newly_seen.add(eid)

    if not full_entries and not head_entries:
        return "", newly_seen

    lines = [
        f'<memory_context project="{project_id}" '
        f'full="{len(full_entries)}" head="{len(head_entries)}" '
        f'suppressed="{skipped}">'
    ]

    if full_entries:
        lines.append("<!-- high relevance -->")
        for e in full_entries:
            lines.append((e.get("text") or "").strip())

    if head_entries:
        lines.append("<!-- available — ask to expand by ID -->")
        for h in head_entries:
            lines.append(f"- [{h['id']}] {h['headline']}")

    if skipped > 0:
        lines.append(f"<!-- {skipped} entries suppressed (already in context) -->")

    lines.append("</memory_context_end>")
    return "\n".join(lines), newly_seen


def _inject_system(request_data, extra: str) -> None:
    """Inject memory context at the end of system prompt.
    
    Order:
    [ Stable System Prompt ]  <-- cache_control breakpoint here
    [ Memory Context Block ]  <-- changes turn-by-turn
    """
    if not extra:
        return
    if request_data.system is None:
        request_data.system = extra
    elif isinstance(request_data.system, str):
        request_data.system = [
            {"type": "text", "text": request_data.system},
            {"type": "text", "text": extra}
        ]
    elif isinstance(request_data.system, list):
        try:
            from api.models.anthropic import SystemContent
            request_data.system.append(SystemContent(type="text", text=extra))
        except ImportError:
            request_data.system.append({"type": "text", "text": extra})


def _extract_memory_block(system) -> str | None:
    """Extract <memory_context> block from system prompt for persistence."""
    if system is None:
        return None
    raw = system if isinstance(system, str) else str(system)
    match = re.search(r"(<memory_context.*?>.*?</memory_context_end>)", raw, flags=re.DOTALL)
    return match.group(1) if match else None


COMMAND_RE = re.compile(
    r"^/(save!|save|search|remember|rollback|reduce|forget|status|pin|unpin|export|clear)\s*(.*)?$",
    re.IGNORECASE | re.DOTALL,
)


def _extract_last_user_text(messages: list) -> str:
    """Extract the text of the last user message."""
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "user":
            continue
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for b in content:
                if (getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)) == "text":
                    t = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
                    if t:
                        return str(t).strip()
    return ""


def _make_fast_response(model: str, text: str) -> dict:
    """Build a fast-path SSE response dict (zero API tokens)."""
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": len(text.split())},
    }


def auto_save_response(
    project_id: str,
    session_id: str,
    text: str,
    is_compressed: bool = False,
) -> None:
    """
    Smart autosave pipeline. Single _smart_classify() call drives all decisions.

    Old pipeline had 8 sequential hardcoded filters each with keyword lists.
    New pipeline: 1 LLM call → structured JSON → all decisions in one shot.

    Steps:
      0. Compressed bypass (from /reduce)
      1. Length gate (< 80 chars = not worth classifying)
      2. Code ratio gate (> 80% code lines = skip, no LLM needed)
      3. _smart_classify() → action / summary / tags / component / pin
      4. Save with LLM-generated summary + tags
    """
    if not MEMORY_ENABLED:
        return

    text = text.strip()
    logger.info(f"[Memory] Autosave START: text_len={len(text)} is_compressed={is_compressed}")

    if is_compressed:
        lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith("- ")]
        logger.info(f"[Memory] Autosave: Processing COMPRESSED output ({len(lines)} lines)")
        for line in lines:
            _save(project_id, session_id, line, pinned=False, source="reduce", check_dedup=True, is_manual=False)
        with closing(_get_conn()) as conn:
            conn.execute(
                "UPDATE memories SET deleted=1 "
                "WHERE project_id=? AND pinned=0 AND source != 'reduce' AND deleted=0",
                (project_id,),
            )
            conn.commit()
        logger.info(f"[Memory] /reduce COMPLETED: Saved {len(lines)} compressed entries")
        return

    if len(text) < 80:
        logger.info(f"[Memory] Autosave SKIP: too short ({len(text)} chars)")
        return

    lines = text.split("\n")
    if lines:
        code_lines = sum(1 for line in lines if CODE_INDICATORS.search(line))
        if len(lines) > 3 and code_lines / len(lines) > 0.8:
            logger.info(f"[Memory] Autosave SKIP: code ratio {code_lines}/{len(lines)}")
            return

    classification = _smart_classify(text)
    action = classification.get("action", "SKIP")
    confidence = classification.get("confidence", 0.5)
    summary = classification.get("summary", "").strip()
    tags = classification.get("tags", [])
    component = classification.get("component", "general")
    should_pin = (action == "SAVE_PIN")
    source = classification.get("_source", "auto")

    logger.info(
        f"[Classifier] action={action} conf={confidence:.2f} comp={component} "
        f"tags={tags} source={source}"
    )

    if action == "SKIP":
        logger.info(f"[Memory] Autosave SKIP: classifier decision (conf={confidence:.2f})")
        return

    save_text = summary if (summary and len(summary) > 10 and confidence >= 0.6) else text
    if save_text == text and len(text) > 200:
        try:
            fallback_summary = _summarize_before_save(text)
            if fallback_summary and len(fallback_summary) > 10:
                save_text = fallback_summary
        except Exception:
            pass

    result = _save(
        project_id,
        session_id,
        save_text,
        pinned=should_pin,
        source=source,
        check_dedup=True,
        is_manual=False,
        trigger=component,
    )

    if isinstance(result, dict):
        if result.get("error") == "duplicate":
            logger.info(f"[Memory] Autosave: duplicate skipped (similarity={result.get('similarity')})")
        elif result.get("error") == "contradiction":
            logger.info(f"[Memory] Autosave: contradiction detected, superseded id={result.get('superseded_id')}")
    else:
        logger.info(f"[Memory] Autosave SAVED: id={result} pin={should_pin} comp={component}")

    try:
        with closing(_get_conn()) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project_id=? AND deleted=0 AND superseded=0",
                (project_id,),
            ).fetchone()[0]
        if count > MEMORY_AUTO_REDUCE_THRESHOLD:
            logger.info(f"[Memory] Auto-reduce threshold hit ({count} entries) — run /reduce")
    except Exception as exc:
        logger.warning(f"[Memory] Auto-reduce check failed: {exc}")


def _handle_reduce(project_id: str, session_id: str, model: str, manual: bool = True) -> tuple:
    """Handle /reduce command — local compression (no LLM needed).
    
    Strategy:
      1. Deduplicate near-identical entries (keep newest).
      2. Merge short entries from the same session into combined entries.
      3. Mark originals as superseded, save merged entries with source='reduce'.
    
    Args:
        manual: True when user explicitly called /reduce (always runs).
                False when called from auto-reduce (gate by token budget).
    """
    rows = []
    with closing(_get_conn()) as conn:
        rows = conn.execute(
            "SELECT id, text, token_count, session_id, created_at FROM memories "
            "WHERE project_id=? AND deleted=0 AND superseded=0 AND pinned=0 "
            "ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()

    active_count = len(rows)

    estimated_tokens = sum(len(r["text"].split()) * 1.3 for r in rows) if rows else 0
    at_token_limit = estimated_tokens >= (MEMORY_TOKEN_BUDGET * 0.85)

    if not manual and not at_token_limit and active_count < 10:
        return None, _make_fast_response(model, "Nothing to reduce yet.")

    if active_count == 0:
        return None, _make_fast_response(model, "No entries to reduce.")

    before_count = active_count
    before_tokens = sum(r["token_count"] for r in rows)

    logger.info(
        f"[Reduce] {'Manual' if manual else 'Auto'} reduce: "
        f"{active_count} entries, ~{int(estimated_tokens)} est. tokens "
        f"(budget={MEMORY_TOKEN_BUDGET}, at_limit={at_token_limit})"
    )

    seen_texts: list[tuple[int, str]] = []
    ids_to_supersede: set[int] = set()

    for r in rows:
        is_dup = False
        for seen_id, seen_text in seen_texts:
            words_a = set(r["text"].lower().split())
            words_b = set(seen_text.lower().split())
            if words_a and words_b:
                overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                if overlap > 0.70:
                    ids_to_supersede.add(seen_id)
                    seen_texts = [(sid, st) if sid != seen_id else (r["id"], r["text"]) for sid, st in seen_texts]
                    is_dup = True
                    break
        if not is_dup:
            seen_texts.append((r["id"], r["text"]))

    remaining = [r for r in rows if r["id"] not in ids_to_supersede]
    session_groups: dict[str, list] = {}
    for r in remaining:
        sid = r["session_id"]
        if sid not in session_groups:
            session_groups[sid] = []
        session_groups[sid].append(r)

    merged_entries: list[tuple[str, str, str]] = []
    merge_source_ids: set[int] = set()

    for sid, entries in session_groups.items():
        short_entries = [e for e in entries if len(e["text"].split()) < 100]
        if len(short_entries) >= 2:
            merged_text = " | ".join(e["text"].strip()[:320] for e in short_entries)
            if len(merged_text) > 400:
                merged_text = merged_text[:400] + "..."
            original_tokens = sum(e["token_count"] for e in short_entries)
            merged_token_count = max(1, len(merged_text) // 3)
            if merged_token_count >= original_tokens:
                logger.info(
                    f"[Reduce] Merge skipped for session={sid}: "
                    f"merged={merged_token_count} >= original={original_tokens} tokens"
                )
                continue
            merged_entries.append((merged_text, sid, short_entries[-1]["created_at"]))
            for e in short_entries:
                merge_source_ids.add(e["id"])

    truncated_ids: set[int] = set()
    truncated_entries: list[tuple[int, str]] = []
    still_remaining = [r for r in remaining if r["id"] not in merge_source_ids]
    for r in still_remaining:
        words = r["text"].split()
        if len(words) > 100:
            short_text = " ".join(words[:80]) + "..."
            truncated_ids.add(r["id"])
            truncated_entries.append((r["id"], short_text))

    all_superseded = ids_to_supersede | merge_source_ids

    if not all_superseded and not truncated_ids:
        return None, _make_fast_response(
            model,
            f"Nothing to compress, {active_count} entries are already clean."
        )

    try:
        with closing(_get_conn()) as conn:
            if all_superseded:
                placeholders = ",".join("?" * len(all_superseded))
                conn.execute(
                    f"UPDATE memories SET superseded=1 WHERE id IN ({placeholders})",
                    list(all_superseded),
                )

            now_utc = datetime.now(UTC).isoformat()
            for merged_text, sid, created_at in merged_entries:
                token_count = max(1, len(merged_text) // 3)
                conn.execute(
                    "INSERT INTO memories "
                    "(project_id, session_id, text, tags, pinned, superseded, deleted, "
                    " source, created_at, token_count) "
                    "VALUES (?,?,?,?,0,0,0,?,?,?)",
                    (project_id, sid, merged_text, "[]", "reduce", now_utc, token_count),
                )

            for entry_id, short_text in truncated_entries:
                new_tc = max(1, len(short_text) // 3)
                conn.execute(
                    "UPDATE memories SET text=?, token_count=? WHERE id=?",
                    (short_text, new_tc, entry_id),
                )
                if _chroma_ok():
                    try:
                        emb = _encode(short_text)
                        if emb:
                            _get_collection(project_id).update(
                                ids=[str(entry_id)],
                                embeddings=[emb],
                                documents=[short_text],
                            )
                    except Exception:
                        pass

            conn.commit()

        if all_superseded and _chroma_ok():
            try:
                coll = _get_collection(project_id)
                str_ids = [str(i) for i in all_superseded]
                coll.delete(ids=str_ids)
                logger.info(f"[Reduce] ChromaDB: deleted {len(str_ids)} superseded vectors")
            except Exception as _ce:
                logger.warning(f"[Reduce] ChromaDB cleanup failed: {_ce}")

        if merged_entries:
            try:
                with closing(_get_conn()) as conn:
                    for merged_text, sid, _ in merged_entries:
                        row = conn.execute(
                            "SELECT id FROM memories WHERE project_id=? AND source='reduce' "
                            "AND text=? AND deleted=0 ORDER BY id DESC LIMIT 1",
                            (project_id, merged_text),
                        ).fetchone()
                        if row and _chroma_ok():
                            emb = _encode(merged_text)
                            if emb:
                                _get_collection(project_id).add(
                                    ids=[str(row["id"])],
                                    embeddings=[emb],
                                    metadatas=[{
                                        "id": row["id"],
                                        "project_id": project_id,
                                        "session_id": sid,
                                        "pinned": 0,
                                        "created_at": datetime.now(UTC).isoformat(),
                                        "token_count": max(1, len(merged_text) // 3),
                                        "tags": "[]",
                                    }],
                                    documents=[merged_text],
                                )
            except Exception as _ie:
                logger.warning(f"[Reduce] ChromaDB index of merged entries failed: {_ie}")

        with closing(_get_conn()) as conn:
            after_count = conn.execute(
                "SELECT COUNT(*) FROM memories "
                "WHERE project_id=? AND deleted=0 AND superseded=0",
                (project_id,),
            ).fetchone()[0]
            after_tokens_row = conn.execute(
                "SELECT COALESCE(SUM(token_count), 0) FROM memories "
                "WHERE project_id=? AND deleted=0 AND superseded=0",
                (project_id,),
            ).fetchone()
            after_tokens = after_tokens_row[0] if after_tokens_row else 0

        deduped = len(ids_to_supersede)
        merged = len(merge_source_ids)
        new_merged = len(merged_entries)
        truncated = len(truncated_ids)

        summary = (
            f"Reduction complete. "
            f"Before: {before_count} entries (~{before_tokens} tokens). "
            f"Deduplicated {deduped}, merged {merged} into {new_merged}, truncated {truncated}. "
            f"After: {after_count} entries (~{after_tokens} tokens)."
        )
        logger.info(f"[Reduce] {summary}")

        try:
            from api.hot_cache import hot_cache as _hc
            from api.routes import _get_memory_hash
            hits = _search("context", project_id=project_id, session_id=session_id, top_k=10)
            if hits:
                new_block = _build_memory_block(hits, project_id=project_id, session_id=session_id)
                if new_block:
                    new_hash = _get_memory_hash(project_id)
                    _hc.set(project_id, new_hash, new_block)
                    logger.info(
                        f"[Reduce] Cache pre-warmed: project={project_id} "
                        f"hash={new_hash} block_len={len(new_block)}"
                    )
            else:
                _hc.invalidate(project_id)
                logger.info("[Reduce] Cache invalidated (empty) — will rebuild next turn")
        except Exception as _ce:
            logger.debug(f"[Reduce] Cache pre-warm skipped: {_ce}")

        return None, _make_fast_response(model, summary)

    except Exception as exc:
        logger.error(f"[Reduce] Failed: {exc}")
        return None, _make_fast_response(model, f"Reduce failed: {exc}")


def _auto_reindex_if_needed(project_id: str) -> None:
    """
    Silently reindex ChromaDB if it's out of sync with SQLite.
    Runs once per project per process lifetime.
    """
    global _chroma_checked
    if project_id in _chroma_checked:
        return
    _chroma_checked.add(project_id)

    if not _chroma_ok():
        return

    try:
        with closing(_get_conn()) as conn:
            sqlite_count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project_id=? AND deleted=0 AND superseded=0",
                (project_id,),
            ).fetchone()[0]

        if sqlite_count == 0:
            return

        safe_pid = re.sub(r"[^a-zA-Z0-9]", "_", project_id).strip("_") or "default"
        coll_name = f"memories_{safe_pid}"[:63]
        try:
            coll = _get_client().get_collection(coll_name)
            chroma_count = coll.count()
        except Exception:
            chroma_count = 0

        if chroma_count >= sqlite_count:
            return

        logger.warning(
            f"AUTO_REINDEX: project={project_id} sqlite={sqlite_count} chroma={chroma_count} — rebuilding"
        )

        with closing(_get_conn()) as conn:
            rows = conn.execute(
                "SELECT id, text, session_id, pinned, created_at, token_count, tags, source "
                "FROM memories WHERE project_id=? AND deleted=0 AND superseded=0",
                (project_id,),
            ).fetchall()

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
                        "project_id": project_id,
                        "session_id": row["session_id"],
                        "pinned": row["pinned"],
                        "created_at": now_str,
                        "token_count": row["token_count"] or 0,
                        "tags": row["tags"] or "[]",
                        "source": row["source"] or "user",
                    }],
                    documents=[row["text"]],
                )
                indexed += 1
            except Exception as exc:
                logger.debug(f"AUTO_REINDEX id={row['id']} failed: {exc}")

        logger.info(f"AUTO_REINDEX: rebuilt {indexed}/{len(rows)} entries for project={project_id}")

    except Exception as exc:
        logger.debug(f"AUTO_REINDEX check failed (non-fatal): {exc}")


def process_memory(
    request_data,
    project_id: str = "default",
    session_id: str = "default",
    already_seen_ids: set | None = None,
):
    """
    Called from routes.py before every provider call.
    Returns (request_data, fast_response | None).
    If fast_response is not None → return immediately (command handled).
    """
    if not MEMORY_ENABLED:
        return request_data, None

    _auto_reindex_if_needed(project_id)

    try:
        last_text = _extract_last_user_text(request_data.messages)
        if not last_text:
            return request_data, None

        match = COMMAND_RE.match(last_text)

        if not match:
            query = last_text[:150]
            search_k = LLM_RERANK_CANDIDATES if LLM_RERANK_ENABLED else MEMORY_TOP_K
            hits = _search_with_tiers(query, project_id, session_id, top_k=search_k)

            hits = [
                h for h in hits
                if h.get("pinned") or not _already_injected(session_id, h.get("id", -1))
            ]

            if hits:
                if already_seen_ids is not None:
                    block, newly_seen = _build_tiered_block(
                        hits, query, project_id, session_id, already_seen_ids
                    )
                    request_data._newly_seen_ids = newly_seen
                elif LLM_RERANK_ENABLED:
                    block = _llm_select_and_build(
                        query, hits, project_id=project_id, session_id=session_id
                    )
                else:
                    block = _build_memory_block(
                        hits, project_id=project_id, session_id=session_id
                    )

                if block:
                    _inject_system(request_data, block)

                    injected_ids = [int(str(h.get("id"))) for h in hits if h.get("id") is not None and h.get("id") != -1]
                    if injected_ids:
                        _track_injected(session_id, injected_ids)

                    inject_tokens = sum(len(h['text'])//3 for h in hits)
                    logger.info(
                        f"MEMORY_INJECT: project={project_id} session={session_id} "
                        f"entries={len(hits)} top_score={hits[0]['score']} "
                        f"tier1={sum(1 for h in hits if h.get('tier')==1)} "
                        f"tier2={sum(1 for h in hits if h.get('tier')==2)} "
                        f"tokens={inject_tokens} "
                        f"past_sessions={sum(1 for h in hits if h.get('past_session'))}"
                    )

                    _injection_history.insert(0, {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "project_id": project_id,
                        "session_id": session_id,
                        "count": len(hits),
                        "tokens": inject_tokens,
                        "top_score": round(hits[0]['score'], 3),
                        "tier1": sum(1 for h in hits if h.get('tier') == 1),
                        "tier2": sum(1 for h in hits if h.get('tier') == 2),
                    })
                    if len(_injection_history) > _INJECTION_HISTORY_MAX:
                        _injection_history[:] = _injection_history[:_INJECTION_HISTORY_MAX]
            return request_data, None

        cmd = match.group(1).lower()
        args = (match.group(2) or "").strip()

        if cmd == "save!":
            if not args:
                return request_data, _make_fast_response(
                    request_data.model, "Usage: /save! <text>"
                )
            save_text = _summarize_before_save(args)
            result = _save(project_id, session_id, save_text, pinned=True, is_manual=True)
            if isinstance(result, dict) and result.get("error") == "duplicate":
                return request_data, _make_fast_response(
                    request_data.model,
                    f"Already in memory (id={result['id']}, "
                    f"similarity={result['similarity']})",
                )
            if isinstance(result, dict):
                tags_str = " ".join(result.get("tags", []))
                msg = f"Saved + pinned (id={result['id']}) — tags: {tags_str}"
                if result.get("contradiction"):
                    c = result["contradiction"]
                    msg += f"\nReplaces id={c['superseded_id']} : {c['old_text']}"
                return request_data, _make_fast_response(request_data.model, msg)
            return request_data, _make_fast_response(
                request_data.model, "Save failed."
            )

        elif cmd == "save":
            if not args:
                return request_data, _make_fast_response(
                    request_data.model, "Usage: /save <text>"
                )
            save_text = _summarize_before_save(args)
            result = _save(project_id, session_id, save_text, pinned=False, is_manual=True)
            if isinstance(result, dict) and result.get("error") == "duplicate":
                return request_data, _make_fast_response(
                    request_data.model,
                    f"Already in memory (id={result['id']}, "
                    f"similarity={result['similarity']})",
                )
            if isinstance(result, dict):
                tags_str = " ".join(result.get("tags", []))
                pin_msg = " [pinned]" if result.get("pinned") else ""
                msg = f"Saved{pin_msg} (id={result['id']}) — tags: {tags_str}"
                if result.get("contradiction"):
                    c = result["contradiction"]
                    msg += f"\nReplaces id={c['superseded_id']} : {c['old_text']}"
                return request_data, _make_fast_response(request_data.model, msg)
            return request_data, _make_fast_response(
                request_data.model, "Save failed."
            )

        elif cmd == "search":
            query = args or "context"
            hits = _search(query, project_id=project_id, session_id=session_id, top_k=MEMORY_TOP_K)
            if not hits:
                return request_data, _make_fast_response(
                    request_data.model, "No results found."
                )
            lines = []
            for h in hits:
                tags_str = ""
                try:
                    tags = json.loads(h["tags"]) if isinstance(h["tags"], str) else h.get("tags", [])
                    tags_str = " ".join(tags) + " " if tags else ""
                except Exception:
                    pass
                pin = " [pinned]" if h.get("pinned") else ""
                degraded = " [degraded-search]" if h.get("degraded") else ""
                lines.append(
                    f"score:{h['score']:.2f} {h['age']} {tags_str}"
                    f"session:{h['session_id']} — {h['text'][:200]}"
                    f"{pin}{degraded}"
                )
            result_text = "Results:\n\n" + "\n".join(lines)
            return request_data, _make_fast_response(request_data.model, result_text)

        elif cmd == "forget":
            try:
                eid = int(args)
                ok = _soft_delete(eid, project_id)
                msg = f"Forgotten (id={eid})" if ok else f"Entry {eid} not found."
            except (ValueError, AttributeError):
                msg = "Usage: /forget <id>"
            return request_data, _make_fast_response(request_data.model, msg)

        elif cmd == "rollback":
            row = None
            with closing(_get_conn()) as conn:
                row = conn.execute(
                    "SELECT id, text, tags, created_at, pinned FROM memories "
                    "WHERE project_id=? AND deleted=0 "
                    "ORDER BY id DESC LIMIT 1",
                    (project_id,),
                ).fetchone()
            if row:
                tags_str = row["tags"] or "[]"
                pin = " [pinned]" if row["pinned"] else ""
                msg = (
                    f"Dernière entrée [id={row['id']}]{pin}\n"
                    f"Date: {row['created_at']}\n"
                    f"Tags: {tags_str}\n\n"
                    f"{row['text']}"
                )
            else:
                msg = "No memory yet."
            return request_data, _make_fast_response(request_data.model, msg)

        elif cmd == "pin":
            try:
                eid = int(args)
                affected = 0
                with closing(_get_conn()) as conn:
                    cur = conn.execute(
                        "UPDATE memories SET pinned=1 WHERE id=? AND project_id=?",
                        (eid, project_id),
                    )
                    affected = cur.rowcount
                    conn.commit()
                msg = f"Pinned (id={eid})" if affected > 0 else f"Entry {eid} not found."
            except (ValueError, AttributeError):
                msg = "Usage: /pin <id>"
            return request_data, _make_fast_response(request_data.model, msg)

        elif cmd == "unpin":
            try:
                eid = int(args)
                affected = 0
                with closing(_get_conn()) as conn:
                    cur = conn.execute(
                        "UPDATE memories SET pinned=0 WHERE id=? AND project_id=?",
                        (eid, project_id),
                    )
                    affected = cur.rowcount
                    conn.commit()
                msg = f"Unpinned (id={eid})" if affected > 0 else f"Entry {eid} not found."
            except (ValueError, AttributeError):
                msg = "Usage: /unpin <id>"
            return request_data, _make_fast_response(request_data.model, msg)

        elif cmd == "status":
            s = _get_stats(project_id, session_id)
            if not s:
                msg = "Unable to get memory stats."
            else:
                msg = (
                    f"Memory Status — project: {project_id} | session: {session_id}\n"
                    f"|-- Entrées actives    : {s.get('active', 0)}  "
                    f"(dont {s.get('pinned', 0)} pinnées)\n"
                    f"|-- Entrées compressées: {s.get('compressed', 0)}\n"
                    f"|-- Tokens injectés    : ~{s.get('tokens_injected', 0)} / "
                    f"budget {s.get('token_budget', MEMORY_TOKEN_BUDGET)} "
                    f"({s.get('budget_pct', 0)}%)\n"
                    f"|-- Dernière save      : {s.get('last_save_age', '—')} — "
                    f"\"{s.get('last_save_text', '—')}\"\n"
                    f"|-- Top tags           : {s.get('top_tags', '—')}\n"
                    f"|-- ChromaDB size      : {s.get('chroma_size', 'N/A')}"
                )
            return request_data, _make_fast_response(request_data.model, msg)

        elif cmd == "export":
            rows = []
            with closing(_get_conn()) as conn:
                rows = conn.execute(
                    "SELECT id, project_id, session_id, text, tags, pinned, "
                    "superseded, source, created_at, token_count "
                    "FROM memories "
                    "WHERE project_id=? AND deleted=0 "
                    "ORDER BY created_at ASC",
                    (project_id,),
                ).fetchall()

            entries = [dict(r) for r in rows]
            export_dir = Path("./exports")
            export_dir.mkdir(exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = export_dir / f"{project_id}_{date_str}.json"
            export_path.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            msg = f"Exported {len(entries)} entries -> {export_path}"
            return request_data, _make_fast_response(request_data.model, msg)

        elif cmd == "clear":
            target = args.strip().lower()

            if target == "session":
                count = 0
                with closing(_get_conn()) as conn:
                    cur = conn.execute(
                        "UPDATE memories SET deleted=1 "
                        "WHERE project_id=? AND session_id=? AND deleted=0",
                        (project_id, session_id),
                    )
                    count = cur.rowcount
                    conn.commit()
                msg = f"Session effacée ({count} entrées)"
                return request_data, _make_fast_response(request_data.model, msg)

            elif target == "project confirm":
                count = 0
                with closing(_get_conn()) as conn:
                    cur = conn.execute(
                        "UPDATE memories SET deleted=1 WHERE project_id=? AND deleted=0",
                        (project_id,),
                    )
                    count = cur.rowcount
                    conn.commit()
                if _chroma_ok():
                    try:
                        safe_pid = re.sub(r"[^a-zA-Z0-9]", "_", project_id).strip("_") or "default"
                        coll_name = f"memories_{safe_pid}"[:63]
                        _get_client().delete_collection(coll_name)
                    except Exception:
                        pass
                msg = f"Projet effacé ({count} entrées)"
                _clear_project_pending.pop(project_id, None)
                return request_data, _make_fast_response(request_data.model, msg)

            elif target == "project":
                _clear_project_pending[project_id] = time.time()
                msg = "Confirmer avec /clear project confirm"
                return request_data, _make_fast_response(request_data.model, msg)

            else:
                msg = "Usage: /clear session | /clear project"
                return request_data, _make_fast_response(request_data.model, msg)

        elif cmd == "reduce":
            reduce_prompt, fast_resp = _handle_reduce(project_id, session_id, request_data.model)
            if fast_resp:
                return request_data, fast_resp
            _inject_system(request_data, reduce_prompt)
            request_data._reduce_mode = True  # type: ignore[attr-defined]
            return request_data, None

        elif cmd == "remember":
            query = args or "context"
            hits = _search(
                query, project_id=project_id, session_id=session_id,
                top_k=10,
            )
            if hits:
                block = _build_memory_block(hits)
                if block:
                    _inject_system(request_data, block)
                    logger.info(
                        f"MEMORY_API_CALL: tokens_used=~{sum(len(h['text'])//3 for h in hits)} "
                        f"reason=remember"
                    )
                    logger.info(
                        f"[Memory] /remember injected {len(hits)} memories (query: {query})"
                    )
            return request_data, None

        else:
            return request_data, None

    except Exception as exc:
        logger.error(f"[Memory] Middleware error: {exc}")
        return request_data, None


TOOL_CAPTURE_RULES: dict[str, str] = {
    "Write":      "capture",
    "Edit":       "capture",
    "MultiEdit":  "capture",
    "Bash":       "ignore",
    "Read":       "ignore",
    "Glob":       "ignore",
    "Grep":       "ignore",
    "TodoWrite":  "capture",
    "WebSearch":  "filter",
}


def process_tool_output(
    tool_name: str, output: str, project_id: str, session_id: str
) -> None:
    """Process tool output from Claude Code hooks. Auto-save disabled."""
    return
