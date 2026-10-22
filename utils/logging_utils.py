"""
Logging configuration setup using Loguru or Python logging fallback.
"""

import sys
from pathlib import Path

try:
    from loguru import logger
    
    def setup_logger(log_dir: str = "logs", debug: bool = False):
        logger.remove()
        level = "DEBUG" if debug else "INFO"
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
        )
        Path(log_dir).mkdir(exist_ok=True)
        logger.add(f"{log_dir}/analyser.log", rotation="10 MB", retention="7 days", level="DEBUG")
        return logger

    logger = setup_logger()

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s - %(message)s")
    logger = logging.getLogger("sports_analyser")
