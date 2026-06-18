"""
app/core/logging.py
────────────────────
Structured logging via Loguru.
Replaces Python's default logging with coloured console output
in dev and JSON-serialisable lines in production.
"""
import sys
from loguru import logger
from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    logger.remove()  # drop default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format=fmt,
        level="DEBUG" if settings.DEBUG else "INFO",
        colorize=True,
        enqueue=True,   # thread-safe async queue
    )
    logger.add(
        "logs/cinematch.log",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        level="INFO",
        enqueue=True,
    )
