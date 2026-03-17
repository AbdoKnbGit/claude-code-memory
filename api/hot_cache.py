"""
api/hot_cache.py

Zero-latency injection cache — the "zero token overhead" layer.

PROBLEM:
  Every session start → query SQLite → query ChromaDB → build injection block
  This costs 10-50ms latency + CPU per session even when nothing changed.

SOLUTION:
  LRU cache keyed by (project_id, memory_hash).
  If memory hasn't changed → serve the exact same block from RAM in <1ms.
  Same block = same cache prefix = Anthropic cache HIT guaranteed.

ZERO TOKEN BENEFIT:
  The injection block tokens are STILL sent to Anthropic.
  But because the block is BYTE-FOR-BYTE IDENTICAL to the cached version,
  Anthropic serves them at cache read price ($0.30/MTok vs $3/MTok).
  Hot cache GUARANTEES this by serving the exact same bytes every time.

ALSO:
  - Tracks cache hit rate per project (exposed to dashboard)
  - Pre-warms on startup from last-known injection blocks in SQLite
  - Thread-safe via simple dict + lock

COST TRACKING:
  Tracks input/cache token counts from Anthropic responses.
  Used by dashboard to show real cost savings.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

from loguru import logger


MAX_ENTRIES  = 200
BLOCK_TTL    = 4500


def get_model_pricing(model: str) -> tuple[float, float, float, float]:
    """Return (rate_input, rate_cache_write_1h, rate_cache_read, rate_output) for a model.

    All cache_write rates use the 1-HOUR tier (cache_control: ephemeral = 1h).
    1h write = 2x base input. cache_read = 0.1x base input.
    Source: https://platform.claude.com/docs/en/about-claude/pricing
    """
    model_lower = (model or "").lower()
    if "claude-haiku-4-5" in model_lower:
        return 1.00, 2.00, 0.10, 5.00
    if "claude-haiku-4" in model_lower:
        return 1.00, 2.00, 0.10, 5.00
    if "claude-3-5-haiku" in model_lower:
        return 0.80, 1.60, 0.08, 4.00
    if "claude-3-haiku" in model_lower:
        return 0.25, 0.50, 0.03, 1.25
    if "claude-opus-4-1" in model_lower:
        return 15.00, 30.00, 1.50, 75.00
    if "claude-opus-4-0" in model_lower:
        return 15.00, 30.00, 1.50, 75.00
    if "claude-opus-4" in model_lower:
        return 5.00, 10.00, 0.50, 25.00
    if "claude-3-opus" in model_lower:
        return 15.00, 30.00, 1.50, 75.00
    if "claude-sonnet-4" in model_lower:
        return 3.00, 6.00, 0.30, 15.00
    if "claude-3-7-sonnet" in model_lower or "claude-3-5-sonnet" in model_lower or "claude-3-sonnet" in model_lower:
        return 3.00, 6.00, 0.30, 15.00
    if "haiku" in model_lower:
        return 1.00, 2.00, 0.10, 5.00
    if "opus" in model_lower:
        return 5.00, 10.00, 0.50, 25.00
    if "sonnet" in model_lower:
        return 3.00, 6.00, 0.30, 15.00
    logger.warning(
        f"[Pricing] Unknown model '{model}' — using Sonnet pricing as default."
    )
    return 3.00, 6.00, 0.30, 15.00


class _HitTracker:
    """Per-project cache hit/miss counters for dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: dict[str, dict] = {}
        self._token_stats: dict[str, dict] = {}

    def record(self, project_id: str, hit: bool) -> None:
        with self._lock:
            if project_id not in self._stats:
                self._stats[project_id] = {"hits": 0, "misses": 0}
            if hit:
                self._stats[project_id]["hits"] += 1
            else:
                self._stats[project_id]["misses"] += 1

    def record_tokens(self, project_id: str, input_tokens: int, cache_read: int, cache_write: int, output_tokens: int, model: str = "") -> None:
        """Record Anthropic token usage for model-aware cost tracking."""
        with self._lock:
            if project_id not in self._token_stats:
                self._token_stats[project_id] = {
                    "input_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "output_tokens": 0,
                    "api_calls": 0,
                    "sessions": 0,
                    "cost_usd": 0.0,
                    "saved_usd": 0.0,
                }
            s = self._token_stats[project_id]
            s["input_tokens"]       += input_tokens
            s["cache_read_tokens"]  += cache_read
            s["cache_write_tokens"] += cache_write
            s["output_tokens"]      += output_tokens
            s["api_calls"]          += 1

            rate_input, rate_cache_w, rate_cache_r, rate_output = get_model_pricing(model)

            cost = (
                (input_tokens  / 1_000_000) * rate_input +
                (cache_write   / 1_000_000) * rate_cache_w +
                (cache_read    / 1_000_000) * rate_cache_r +
                (output_tokens / 1_000_000) * rate_output
            )
            no_cache_cost = (
                ((input_tokens + cache_read + cache_write) / 1_000_000) * rate_input +
                (output_tokens / 1_000_000) * rate_output
            )
            s["cost_usd"]  += cost
            s["saved_usd"] += max(0.0, no_cache_cost - cost)

    def get_stats(self, project_id: str) -> dict:
        with self._lock:
            hits_data = self._stats.get(project_id, {"hits": 0, "misses": 0})
            total = hits_data["hits"] + hits_data["misses"]
            hit_rate = (hits_data["hits"] / total * 100) if total > 0 else 0.0
            token_data = self._token_stats.get(project_id, {
                "input_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
                "output_tokens": 0, "api_calls": 0, "sessions": 0,
                "cost_usd": 0.0, "saved_usd": 0.0,
            })
            return {
                "cache_hits": hits_data["hits"],
                "cache_misses": hits_data["misses"],
                "cache_hit_rate": round(hit_rate, 1),
                **token_data,
            }

    def get_all_stats(self) -> dict:
        with self._lock:
            result = {}
            for pid in set(list(self._stats.keys()) + list(self._token_stats.keys())):
                result[pid] = self.get_stats(pid)
            return result


hit_tracker = _HitTracker()


class HotCache:
    """
    LRU cache of injection blocks.
    Key: (project_id, memory_hash)
    Value: (injection_block: str, timestamp: float)
    """

    def __init__(self, max_size: int = MAX_ENTRIES):
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], tuple[str, float]] = OrderedDict()
        self._max_size = max_size

    def get(self, project_id: str, memory_hash: str) -> str | None:
        """Return cached injection block or None if miss/expired."""
        key = (project_id, memory_hash)
        with self._lock:
            if key not in self._cache:
                hit_tracker.record(project_id, hit=False)
                return None

            block, ts = self._cache[key]
            if time.time() - ts > BLOCK_TTL:
                del self._cache[key]
                hit_tracker.record(project_id, hit=False)
                return None

            self._cache.move_to_end(key)
            hit_tracker.record(project_id, hit=True)
            logger.debug(f"[HotCache] HIT project={project_id} hash={memory_hash[:8]}")
            return block

    def set(self, project_id: str, memory_hash: str, block: str) -> None:
        """Store injection block in cache."""
        key = (project_id, memory_hash)
        with self._lock:
            self._cache[key] = (block, time.time())
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
        logger.debug(f"[HotCache] SET project={project_id} hash={memory_hash[:8]} len={len(block)}")

    def invalidate(self, project_id: str) -> None:
        """Invalidate all cached blocks for a project (call on any save)."""
        with self._lock:
            to_delete = [k for k in self._cache if k[0] == project_id]
            for k in to_delete:
                del self._cache[k]
        if to_delete:
            logger.debug(f"[HotCache] INVALIDATED {len(to_delete)} entries for project={project_id}")

    def warm(self, project_id: str, memory_hash: str, block: str) -> None:
        """Pre-warm cache entry (called on startup from persisted hashes)."""
        self.set(project_id, memory_hash, block)
        logger.info(f"[HotCache] PRE-WARMED project={project_id}")

    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_entries": len(self._cache),
                "projects": list(set(k[0] for k in self._cache)),
                "hit_stats": hit_tracker.get_all_stats(),
            }


hot_cache = HotCache()


def extract_and_track_tokens(response_data: dict, project_id: str) -> None:
    """
    Parse Anthropic response JSON and record token usage.
    Call this from routes.py after each API response.
    """
    if not project_id or not response_data:
        return
    try:
        usage = response_data.get("usage", {})
        if not usage:
            return
        hit_tracker.record_tokens(
            project_id=project_id,
            input_tokens=usage.get("input_tokens", 0),
            cache_read=usage.get("cache_read_input_tokens", 0),
            cache_write=usage.get("cache_creation_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            model=response_data.get("model", ""),
        )
    except Exception as e:
        logger.debug(f"[HotCache] Token tracking failed: {e}")


class NotificationBus:
    """
    Simple in-process pub/sub for real-time dashboard notifications.

    Events pushed here appear in /api/notifications as SSE or polled JSON.
    Useful for:
    - "Memory saved: auth component (3 entries compressed)"
    - "Cache warmed for project ProjetBI"
    - "New session detected — injecting 136 tokens"
    - "Cost this session: $0.21 (saved $1.23)"
    """

    def __init__(self, max_events: int = 100):
        self._lock = threading.Lock()
        self._events: deque = deque(maxlen=max_events)
        self._max = max_events

    def push(self, event_type: str, message: str, project_id: str = "", data: dict = None) -> None:
        event = {
            "type": event_type,
            "message": message,
            "project_id": project_id,
            "ts": time.time(),
            "data": data or {},
        }
        with self._lock:
            self._events.appendleft(event)
        logger.debug(f"[Notify] {event_type}: {message}")

    def get_recent(self, project_id: str = "", limit: int = 20) -> list[dict]:
        with self._lock:
            events = list(self._events)
        if project_id:
            events = [e for e in events if not e["project_id"] or e["project_id"] == project_id]
        return events[:limit]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


notifications = NotificationBus()
