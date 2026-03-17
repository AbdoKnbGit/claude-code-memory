"""
Anthropic native provider — forwards requests directly via the anthropic SDK.

Since Claude Code already sends Anthropic-format requests, this provider
streams the response natively and re-emits SSE events as-is.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from loguru import logger

from providers.base import BaseProvider, ProviderConfig
from providers.rate_limit import GlobalRateLimiter


class AnthropicProvider(BaseProvider):
    """Native Anthropic API provider using the anthropic Python SDK."""

    _last_tool_fingerprints: dict[str, str] = {}

    def __init__(self, config: ProviderConfig, *, model_name: str | None = None):
        super().__init__(config)
        self._model_name = model_name
        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self._limiter = GlobalRateLimiter(
            rate_limit=config.rate_limit or 60,
            rate_window=config.rate_window or 60.0,
            max_concurrency=config.max_concurrency or 5,
        )
        logger.info(f"Anthropic provider initialized: model={model_name}")

    async def cleanup(self) -> None:
        """Release resources."""
        await self._client.close()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        async for chunk in self._stream_impl(request, input_tokens, request_id):
            yield chunk

    async def _stream_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
    ) -> AsyncIterator[str]:
        """Core streaming implementation."""
        rid = request_id or f"req_{uuid.uuid4().hex[:12]}"

        await self._limiter.wait_if_blocked()
        async with self._limiter.concurrency_slot():
            kwargs: dict[str, Any] = {
                "model": getattr(request, "model", None) or self._model_name or "claude-haiku-4-5-20251001",
                "max_tokens": getattr(request, "max_tokens", 8192),
            }

            messages = []
            for msg in request.messages:
                if hasattr(msg, "model_dump"):
                    messages.append(msg.model_dump(exclude_none=True))
                elif isinstance(msg, dict):
                    messages.append(msg)
                else:
                    messages.append({"role": msg.role, "content": msg.content})
            kwargs["messages"] = messages

            if request.system:
                if isinstance(request.system, str):
                    kwargs["system"] = [
                        {
                            "type": "text",
                            "text": request.system,
                            "cache_control":{"type": "ephemeral"}
                        }
                    ]
                elif isinstance(request.system, list):
                    sys_list = []
                    last_idx = len(request.system) - 1
                    target_idx = last_idx - 1 if last_idx > 0 else 0

                    for i, block in enumerate(request.system):
                        if isinstance(block, dict):
                            b = dict(block)
                        elif hasattr(block, "model_dump"):
                            b = block.model_dump(exclude_none=True)
                        else:
                            b = {"type": "text", "text": str(block)}

                        if i == target_idx:
                            b["cache_control"] = {"type": "ephemeral"}
                        sys_list.append(b)

                    if (
                        getattr(request, "_memory_block_stable", False)
                        and len(sys_list) >= 2
                    ):
                        sys_list[-1]["cache_control"] = {"type": "ephemeral"}

                    kwargs["system"] = sys_list

            if hasattr(request, "temperature") and request.temperature is not None:
                kwargs["temperature"] = request.temperature

            if hasattr(request, "top_p") and request.top_p is not None:
                kwargs["top_p"] = request.top_p

            if hasattr(request, "tools") and request.tools:
                tools = []
                for t in request.tools:
                    if hasattr(t, "model_dump"):
                        tools.append(t.model_dump(exclude_none=True))
                    elif isinstance(t, dict):
                        tools.append(t)
                if tools:
                    tools = self._filter_tools(tools, messages)
                    cleaned = []
                    for t in tools:
                        bt = dict(t)
                        if "input_schema" not in bt and "parameters" in bt:
                            bt["input_schema"] = bt.pop("parameters")
                        cleaned.append(bt)
                    import hashlib as _hashlib_tools
                    _session_id_for_tools = (
                        getattr(request, "metadata", {}) or {}
                    ).get("session_id", "default") if hasattr(request, "metadata") else "default"
                    _tool_fp = _hashlib_tools.md5(
                        ",".join(sorted(t.get("name", "") for t in cleaned)).encode()
                    ).hexdigest()[:12]
                    _prev_tool_fp = self._last_tool_fingerprints.get(_session_id_for_tools, "")
                    _tools_stable = (_tool_fp == _prev_tool_fp) if _prev_tool_fp else False
                    self._last_tool_fingerprints[_session_id_for_tools] = _tool_fp

                    if _tools_stable and cleaned:
                        cleaned[-1]["cache_control"] = {"type": "ephemeral"}

                    kwargs["tools"] = cleaned

            if hasattr(request, "metadata") and request.metadata:
                kwargs["metadata"] = request.metadata

            kwargs["extra_headers"] = {
                "anthropic-beta": ",".join([
                    "prompt-caching-2024-07-31",
                    "context-management-2025-06-27",
                    "compact-2026-01-12",
                    "token-efficient-tools-2025-02-19",
                ]),
            }

            kwargs["extra_body"] = {
                "context_management": {
                    "strategies": [
                        {"type": "clear_tool_uses_20250919"},
                        {"type": "clear_thinking_20251015", "keep_last_n": 2},
                    ]
                },
                "compaction": {
                    "instructions": (
                        "Focus on preserving code snippets, variable names, "
                        "and technical decisions."
                    ),
                },
            }

            _actual_model = kwargs["model"]
            logger.info(
                f"ANTHROPIC_REQUEST: rid={rid} model={_actual_model} "
                f"messages={len(messages)} tools={len(kwargs.get('tools', []))} "
                f"input_tokens≈{input_tokens}"
            )

            _msgs = kwargs.get("messages", [])
            _is_tool_result_only = bool(
                _msgs and
                isinstance(_msgs[-1], dict) and
                _msgs[-1].get("role") == "user" and
                isinstance(_msgs[-1].get("content"), list) and
                _msgs[-1]["content"] and
                all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in _msgs[-1]["content"]
                )
            )
            if _is_tool_result_only and kwargs.get("thinking"):
                kwargs["thinking"] = {"type": "disabled"}
                logger.debug("[Client] Thinking disabled for tool-result-only turn")

            import asyncio as _asyncio
            import random as _random

            _RETRYABLE_STATUS = {429, 503, 529}
            _MAX_ATTEMPTS = 3

            for _attempt in range(_MAX_ATTEMPTS):
                try:
                    async with self._client.messages.stream(**kwargs) as stream:
                        async for event in stream:
                            sse_line = self._event_to_sse(event)
                            if sse_line:
                                yield sse_line
                    break
                except anthropic.APIStatusError as e:
                    if e.status_code not in _RETRYABLE_STATUS:
                        logger.error(f"Anthropic API error: {e.status_code} {e.message}")
                        error_event = {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(e.message),
                            },
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                        break

                    if _attempt == _MAX_ATTEMPTS - 1:
                        logger.error(f"Anthropic API error after {_MAX_ATTEMPTS} attempts: {e.status_code} {e.message}")
                        error_event = {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": str(e.message),
                            },
                        }
                        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                        break

                    _wait = (2 ** _attempt) + _random.uniform(0, 0.5)
                    logger.warning(
                        f"ANTHROPIC_RETRY: status={e.status_code}, "
                        f"attempt={_attempt+1}/{_MAX_ATTEMPTS}, "
                        f"waiting {_wait:.1f}s"
                    )
                    await _asyncio.sleep(_wait)
                except Exception as e:
                    logger.error(f"Anthropic stream error: {e}")
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"Stream error: {e!s}",
                        },
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                    break

    def _event_to_sse(self, event: Any) -> str | None:
        """Convert an anthropic stream event to SSE string."""
        event_type = getattr(event, "type", None)
        if not event_type:
            return None

        try:
            if hasattr(event, "model_dump"):
                data = event.model_dump()
            elif hasattr(event, "to_dict"):
                data = event.to_dict()
            else:
                data = {"type": event_type}

            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:
            logger.debug(f"Failed to serialize event {event_type}: {exc}")
            return None


    _CORE_TOOLS: set[str] = {
        "Read", "Write", "Edit", "MultiEdit",
        "Bash", "Glob", "Grep", "LS",
        "TodoRead", "TodoWrite",
        "Task", "BatchTool",
    }

    @staticmethod
    def _filter_tools(
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        *,
        lookback: int = 4,
    ) -> list[dict[str, Any]]:
        """Filter tool definitions to recently-used + core tools.

        Scans the last *lookback* assistant messages for tool_use blocks,
        collects their names, unions with _CORE_TOOLS, and strips everything
        else.  Falls back to returning all tools if TOOL_FILTER_ENABLED=0.
        """
        import os
        if os.getenv("TOOL_FILTER_ENABLED", "1") == "0":
            return tools

        recent_names: set[str] = set(AnthropicProvider._CORE_TOOLS)
        count = 0
        for msg in reversed(messages):
            if count >= lookback:
                break
            if msg.get("role") != "assistant":
                continue
            count += 1
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name:
                        recent_names.add(name)

        filtered = [t for t in tools if t.get("name", "") in recent_names]

        if not filtered and tools:
            return tools

        if len(filtered) < len(tools):
            logger.info(
                f"TOOL_FILTER: {len(tools)} → {len(filtered)} tools "
                f"(cut {len(tools) - len(filtered)}, "
                f"recent={recent_names - AnthropicProvider._CORE_TOOLS})"
            )

        return filtered

