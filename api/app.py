from __future__ import annotations

"""FastAPI application factory and configuration."""

from loguru import logger
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import asyncio
import os
import traceback
from pathlib import Path

import httpx

from config.logging_config import configure_logging
from config.settings import get_settings
from providers.exceptions import ProviderError

from .dependencies import cleanup_provider
from .routes import router

_settings = get_settings()
configure_logging(_settings.log_file)


_SHUTDOWN_TIMEOUT_S = 5.0

_prewarm_client: httpx.AsyncClient | None = None

def get_prewarm_client() -> httpx.AsyncClient:
    """Return the shared prewarm HTTP client (initialized at startup)."""
    if _prewarm_client is None:
        raise RuntimeError("Prewarm client not initialized — lifespan not run")
    return _prewarm_client


async def _best_effort(
    name: str, awaitable, timeout_s: float = _SHUTDOWN_TIMEOUT_S
) -> None:
    """Run a shutdown step with timeout; never raise to callers."""
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_s)
    except TimeoutError:
        logger.warning(f"Shutdown step timed out: {name} ({timeout_s}s)")
    except Exception as e:
        logger.warning(f"Shutdown step failed: {name}: {type(e).__name__}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    wc = int(os.getenv("WEB_CONCURRENCY", "1"))
    if wc > 1:
        raise RuntimeError(
            f"WEB_CONCURRENCY={wc} is not supported. "
            "This proxy uses in-process OrderedDicts for session state. "
            "Multi-worker mode produces silent split-brain corruption. "
            "Set WEB_CONCURRENCY=1 — asyncio handles concurrency within one process."
        )
    logger.info("Starting Claude Code Proxy...")

    global _prewarm_client
    _prewarm_client = httpx.AsyncClient(
        timeout=10.0,
        http2=True,
    )
    logger.info("[Startup] Shared prewarm HTTP client initialized (HTTP/2)")
    
    try:
        from memory import _get_embed_model
        logger.info("[Pre-warm] Loading embedding model...")
        _get_embed_model()
        
    except Exception as e:
        logger.warning(f"Pre-warming skipped or failed: {e}")

    try:
        from memory import _call_llm
        logger.info("[Startup] Testing memory model connectivity...")
        test_result = _call_llm(
            system="Reply with OK only.",
            user="test",
            max_tokens=3,
            timeout=8.0,
            task="startup_health_check"
        )
        if not test_result:
            raise RuntimeError("Memory model returned empty response")
        logger.info(f"[Startup] Memory model health check passed: {test_result!r}")
    except SystemExit:
        raise
    except Exception as e:
        logger.critical(
            f"MEMORY_MODEL_UNAVAILABLE: {e}. "
            f"Proxy cannot start without a working memory model. "
            f"Check MEMORY_MODEL env var and API key in .env. "
            f"A session without compression accumulates history without bound, "
            f"corrupts your memory DB permanently, and costs 10x normal price by turn 30."
        )
        raise SystemExit(1)

    async def _memory_model_keepalive_loop():
        """Ping MEMORY_MODEL provider every 4 minutes to prevent cold start latency."""
        while True:
            await asyncio.sleep(240)
            try:
                from memory import _call_llm
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, _call_llm,
                    "keepalive", "ping", 1, 5.0, "memory_model_keepalive",
                )
                logger.debug("[MemoryModel] Keepalive ping sent")
            except Exception as e:
                logger.debug(f"[MemoryModel] Keepalive failed (non-critical): {e}")

    asyncio.create_task(_memory_model_keepalive_loop())

    yield
    if _prewarm_client:
        await _best_effort("close_prewarm_client", _prewarm_client.aclose())
        logger.info("[Shutdown] Prewarm HTTP client closed")
    await _best_effort("cleanup_provider", cleanup_provider())
    logger.info("Server shut down cleanly")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Claude Code Proxy",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="web")
        logger.info("Web UI available at http://localhost:8082/ui")

        @app.get("/favicon.ico", include_in_schema=False)
        async def favicon():
            return FileResponse(web_dir / "Logo.png")

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        """Handle provider-specific errors and return Anthropic format."""
        logger.error(f"Provider Error: {exc.error_type} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_anthropic_format(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle general errors and return Anthropic format."""
        logger.error(f"General Error: {exc!s}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "An unexpected error occurred.",
                },
            },
        )

    return app


app = create_app()
