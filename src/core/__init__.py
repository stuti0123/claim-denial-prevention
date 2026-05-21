"""
src/core — Cross-cutting infrastructure for the Claim Denial System.

Modules
-------
error_codes : Centralised catalogue of all error codes used across the system.
logger      : Factory function that every module uses to get a configured logger.

Usage
-----
    from src.core.logger import get_logger
    from src.core.error_codes import ErrorCode

    logger = get_logger(__name__)
"""
