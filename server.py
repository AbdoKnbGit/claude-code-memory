"""
Claude Code Proxy - Entry Point

Minimal entry point that imports the app from the api module.
Run with: uv run uvicorn server:app --host 0.0.0.0 --port 8082 --timeout-graceful-shutdown 5
"""

import os
import shutil
import threading
import time as _time
from datetime import datetime
from pathlib import Path

from api.app import app, create_app

VERSION = Path("VERSION").read_text(encoding="utf-8").strip() if Path("VERSION").exists() else "dev"

__all__ = ["VERSION", "app", "create_app"]


def _start_backup_scheduler():
    """Copy memory.db → data/backups/ every 6 hours, cleanup old backups."""
    from loguru import logger

    db_path = os.getenv("MEMORY_DB_PATH", "./data/memory.db")
    keep_days = int(os.getenv("BACKUP_KEEP_DAYS", "7"))

    while True:
        _time.sleep(21600)
        try:
            src = Path(db_path)
            if not src.exists():
                continue
            dst_dir = Path("./data/backups")
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(src, dst)
            size_kb = dst.stat().st_size // 1024
            logger.info(f"[Backup] memory.db → {dst.name} ({size_kb}KB)")

            cutoff = _time.time() - keep_days * 86400
            for f in dst_dir.glob("memory_*.db"):
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    logger.info(f"[Backup] Deleted old backup: {f.name}")
        except Exception:
            pass


if os.getenv("BACKUP_ENABLED", "false").lower() in ("true", "1", "yes"):
    threading.Thread(target=_start_backup_scheduler, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    from loguru import logger

    from cli.process_registry import kill_all_best_effort
    from config.settings import get_settings

    settings = get_settings()
    logger.info(f"cc-memory v{VERSION} starting")
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level="debug",
            timeout_graceful_shutdown=5,
        )
    finally:
        kill_all_best_effort()
