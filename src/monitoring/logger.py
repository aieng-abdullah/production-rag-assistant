"""Structured JSON logging with Loguru."""

import sys
from loguru import logger

from src.config import Config


def get_logger(name: str = None):
    """Get configured logger instance."""
    # Remove default handler
    logger.remove()
    
    # Add structured console handler
    logger.add(
        sys.stdout,
        level=Config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        serialize=False
    )
    
    # Add file handler for production
    if Config.CHROMA_HOST:  # Production indicator
        logger.add(
            "logs/app.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra} | {message}",
            serialize=True,
            rotation="10 MB",
            retention="7 days"
        )
    
    if name:
        return logger.bind(name=name)
    return logger
