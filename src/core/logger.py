"""
src/core/logger.py
------------------
Centralised logger factory for the Claim Denial System.

WHY ONE FACTORY?
----------------
Python's logging.basicConfig() is a ONE-TIME setup call. If two modules both
call it, only the first one takes effect and the second is silently ignored.
This caused the DEBUG log to be invisible in earlier code.

The fix: every module calls get_logger(__name__) from this file.
This file configures the ROOT logger ONCE, and all module loggers inherit
from it automatically. No module ever calls basicConfig() directly again.

SINGLE LOG FILE
---------------
All layers write to logs/app.log. This means:
    - One grep command finds errors across all layers
    - Error codes (e.g. "ING-1001") can be searched system-wide
    - Operations team has one file to monitor in CloudWatch

ROTATING HANDLER
----------------
RotatingFileHandler replaces the plain FileHandler. It:
    - Caps log file size at MAX_BYTES (10 MB)
    - Keeps BACKUP_COUNT (5) old files before discarding
    - Prevents disk from filling up in production

LOG LEVEL FROM ENVIRONMENT
--------------------------
Set LOG_LEVEL=DEBUG in your .env or shell to see debug messages.
Set LOG_LEVEL=WARNING in production to reduce noise and cost.
Default is DEBUG (show everything during development).

Usage
-----
    from src.core.logger import get_logger

    logger = get_logger(__name__)

    # Good: pass values as arguments, not inside the format string
    # This avoids string formatting overhead when the log level is disabled
    count = len(df)
    logger.info("Silver pipeline complete. rows=%d", count)

    # With error code — so errors are searchable by code
    logger.error("[%s] File not found: path=%s", ErrorCode.ING_FILE_NOT_FOUND, path)
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Constants                                                                    #
# --------------------------------------------------------------------------- #

# Resolved once at import time — all loggers write to the same file
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_LOG_DIR:      Path = _PROJECT_ROOT / "logs"
_LOG_FILE:     Path = _LOG_DIR / "app.log"

# Rotate at 10 MB, keep 5 backup files (app.log.1 … app.log.5)
_MAX_BYTES:     int = 10 * 1024 * 1024   # 10 MB
_BACKUP_COUNT:  int = 5

# Single format used by ALL handlers — timestamp | level | module | message
# The calling code is responsible for embedding the error code in the message.
_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# Track whether the root logger has been configured yet in this process
_configured: bool = False


# --------------------------------------------------------------------------- #
#  Internal setup                                                               #
# --------------------------------------------------------------------------- #

def _configure_root_logger() -> None:
    """
    Configure the root logger ONCE per process.

    Called automatically by get_logger() on first use. Subsequent calls
    are no-ops because the _configured flag is checked first.

    Reads LOG_LEVEL from the environment (default: DEBUG).
    Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL (case-insensitive).
    """
    global _configured
    if _configured:
        # Root logger already has handlers — do nothing.
        # This is the key fix: we never call basicConfig() again.
        return

    # Ensure the logs directory exists
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Read log level from environment — allows changing level without code edits
    raw_level: str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    level: int | None = getattr(logging, raw_level, None)

    if not isinstance(level, int):
        # Invalid level string — fall back to DEBUG and warn
        level = logging.DEBUG
        print(
            f"[CFG-8002] WARNING: Invalid LOG_LEVEL='{raw_level}' in environment. "
            f"Falling back to DEBUG. Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    # Shared formatter used by both handlers
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Handler 1: Console (StreamHandler) — shows logs in terminal during dev
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Handler 2: Rotating file — writes to logs/app.log, rotates at 10 MB
    file_handler = RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Configure the ROOT logger — all module loggers inherit from this
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _configured = True


# --------------------------------------------------------------------------- #
#  Public API                                                                   #
# --------------------------------------------------------------------------- #

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger for the calling module.

    This is the ONLY function every module should use. It ensures the root
    logger is configured before returning the module-level logger.

    Args:
        name: Logger name — always pass __name__ from the calling module.
              This makes the log line show which file emitted it, e.g.:
              "src.silver.cleaner" or "src.gold.feature_engineer".

    Returns:
        A configured logging.Logger instance for the given name.

    Example:
        from src.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Pipeline started. rows=%d", len(df))
    """
    _configure_root_logger()
    return logging.getLogger(name)
