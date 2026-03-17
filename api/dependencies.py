"""Dependency injection for FastAPI."""

import os

from fastapi import HTTPException, Request
from loguru import logger

from config.settings import Settings
from config.settings import get_settings as _get_settings
from providers.anthropic import AnthropicProvider
from providers.base import BaseProvider, ProviderConfig
from providers.common import get_user_facing_error_message
from providers.exceptions import AuthenticationError


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_provider: BaseProvider | None = None


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def _create_provider(settings: Settings) -> BaseProvider:
    """Construct and return a new provider instance from settings."""
    ptype = settings.provider_type

    if ptype == "nvidia_nim":
        from providers.nvidia_nim import NVIDIA_NIM_BASE_URL, NvidiaNimProvider
        if not settings.nvidia_nim_api_key or not settings.nvidia_nim_api_key.strip():
            raise AuthenticationError(
                "NVIDIA_NIM_API_KEY is not set. Add it to your .env file."
            )
        config = ProviderConfig(
            api_key=settings.nvidia_nim_api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        provider = NvidiaNimProvider(config, nim_settings=settings.nim)

    elif ptype in ("open_router", "openrouter", "gemini", "groq", "deepseek", "grok"):
        from providers.open_router import OPENROUTER_BASE_URL, OpenRouterProvider
        from providers.openai_compat import OpenAIProvider
        if ptype in ("open_router", "openrouter"):
            key = settings.open_router_api_key
            url = OPENROUTER_BASE_URL
            key_name = "OPENROUTER_API_KEY"
        elif ptype == "gemini":
            key = settings.gemini_api_key
            url = GEMINI_BASE_URL
            key_name = "GEMINI_API_KEY"
        elif ptype == "groq":
            key = settings.groq_api_key
            url = GROQ_BASE_URL
            key_name = "GROQ_API_KEY"
        elif ptype == "deepseek":
            key = os.getenv("DEEPSEEK_API_KEY", "")
            url = "https://api.deepseek.com/v1"
            key_name = "DEEPSEEK_API_KEY"
        else:
            key = os.getenv("GROK_API_KEY", "")
            url = "https://api.x.ai/v1"
            key_name = "GROK_API_KEY"

        if not key or not key.strip():
            raise AuthenticationError(f"{key_name} is not set. Add it to your .env file.")

        config = ProviderConfig(
            api_key=key,
            base_url=url,
            rate_limit=settings.provider_rate_limit or 40,
            rate_window=settings.provider_rate_window or 60,
            max_concurrency=settings.provider_max_concurrency or 5,
            http_read_timeout=settings.http_read_timeout or 300.0,
            http_write_timeout=settings.http_write_timeout or 10.0,
            http_connect_timeout=settings.http_connect_timeout or 2.0,
        )
        if ptype in ("open_router", "openrouter"):
            provider = OpenRouterProvider(config)
        else:
            provider = OpenAIProvider(
                config, provider_name=ptype.upper(), base_url=url, api_key=key
            )

    elif ptype == "lmstudio":
        from providers.lmstudio import LMStudioProvider
        config = ProviderConfig(
            api_key="lm-studio",
            base_url=settings.lm_studio_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        provider = LMStudioProvider(config)

    elif ptype == "anthropic":
        if not settings.anthropic_api_key or not settings.anthropic_api_key.strip():
            raise AuthenticationError("ANTHROPIC_API_KEY is not set.")
        config = ProviderConfig(
            api_key=settings.anthropic_api_key,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        provider = AnthropicProvider(config, model_name=settings.model_name)
    else:
        logger.error(f"Unknown provider_type: '{ptype}'")
        raise ValueError(f"Unknown provider_type: '{ptype}'")

    logger.info("Provider initialized: {}", ptype)
    return provider


def get_provider() -> BaseProvider:
    """Get or create the provider instance based on settings.provider_type."""
    global _provider
    if _provider is None:
        try:
            _provider = _create_provider(get_settings())
        except AuthenticationError as e:
            raise HTTPException(
                status_code=503, detail=get_user_facing_error_message(e)
            ) from e
    return _provider


def get_provider_for_request(raw_request: Request) -> BaseProvider:
    """
    Smart provider resolution:

    CASE 1 — Pro/Max user (no API key, OAuth login):
      MODEL= not set in .env
      → reads OAuth token from incoming x-api-key header
      → creates AnthropicProvider with that token per-request
      → memory injection done by MEMORY_MODEL (Groq/Gemini/etc.) before forwarding
      → forwards enriched request to real Anthropic transparently

    CASE 2 — API key user with explicit MODEL= set:
      MODEL=groq/llama-3.3-70b-versatile (or any non-anthropic)
      → uses that provider as the brain (fake-key mode)
      → memory injection still done by MEMORY_MODEL

    CASE 3 — API key user, MODEL=anthropic/...
      → uses ANTHROPIC_API_KEY from .env
    """
    settings = get_settings()

    model_env = os.getenv("MODEL", "").strip()
    if model_env and "/" in model_env:
        provider_type = model_env.split("/", 1)[0].lower()
        if provider_type != "anthropic":
            logger.info(f"[Provider] MODEL={model_env} → using as brain")
            return get_provider()
        else:
            return get_provider()

    token = (
        raw_request.headers.get("x-api-key", "")
        or raw_request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    )

    if not token or token.lower() in ("fake-key", "dummy", "test", "", "sk-placeholder-key-for-proxy"):
        logger.warning("[Provider] No valid token found — falling back to .env provider")
        return get_provider()

    config = ProviderConfig(
        api_key=token,
        rate_limit=settings.provider_rate_limit,
        rate_window=settings.provider_rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
    )
    logger.info("[Passthrough] OAuth/API token → real Anthropic (model from request body)")
    return AnthropicProvider(config, model_name=None)


async def cleanup_provider():
    """Cleanup provider resources."""
    global _provider
    if _provider:
        await _provider.cleanup()
    _provider = None
    logger.debug("Provider cleanup completed")
