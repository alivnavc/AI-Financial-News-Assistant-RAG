import logging
import traceback
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "rag_chat.log")

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a logger that writes to both file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def log_exception_sync(logger: logging.Logger, msg: str, exc: Exception):
    """Log exceptions with stack trace to file and console."""
    logger.error(f"{msg}: {exc}")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(tb)
