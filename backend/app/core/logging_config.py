"""
Logging configuration for security and application logging
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

from app.config import get_settings

settings = get_settings()


def setup_logging():
    """Setup application logging"""
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO" if not settings.DEBUG else "DEBUG",
        colorize=True
    )
    
    # File handler for all logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        compression="zip"
    )
    
    # Security log file
    logger.add(
        log_dir / "security_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="90 days",
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=lambda record: "security" in record["message"].lower() or record["level"].name in ["WARNING", "ERROR", "CRITICAL"],
        compression="zip"
    )
    
    return logger


# Initialize logging
app_logger = setup_logging()

