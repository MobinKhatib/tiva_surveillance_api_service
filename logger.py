import logging
import os
from logging.handlers import RotatingFileHandler
import sys
import atexit
from contextlib import contextmanager
import config
atexit.register(logging.shutdown)  # Ensures logs are saved before exit

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

# Get the string level from config.py and convert it to the logging constant
level_str = config.LEVEL.upper()  # Ensure it's in uppercase
level = getattr(logging, level_str, logging.INFO)  # Default to INFO if invalid

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(level)

# Configure logging with rotation
logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(log_file, mode="a", maxBytes=10*1024*1024, backupCount=5),  # 10MB per file, keep 5 backups (when all become full the first one is deleted)
        # Windows: Some programs lock files, preventing logging from writing to them. Solution: Use "a" mode explicitly in RotatingFileHandler:
        logging.StreamHandler()  # Print logs to console
    ]
)

# global handler without try/except
def global_exception_handler(exctype, value, tb):
    logging.exception("Unhandled exception", exc_info=(exctype, value, tb))

sys.excepthook = global_exception_handler

logger = logging.getLogger(__name__)  # Global logger
logger.info("Logging system initialized.")
# ------------------------------
# Global handler for unchecked errors
# ------------------------------
def global_exception_handler(exctype, value, tb):
    logger.exception("Unhandled exception", exc_info=(exctype, value, tb))

sys.excepthook = global_exception_handler

# ------------------------------
# Decorator for logging errors of specific functions
# ------------------------------
def log_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception occurred in {func.__name__}: {e}")
            raise
    return wrapper

# ------------------------------
# Context manager To log the start, end, or error of specific blocks
# ------------------------------
@contextmanager
def log_context(msg):
    logger.info(f" Starting: {msg}")
    try:
        yield
        logger.info(f" Finished: {msg}")
    except Exception as e:
        logger.error(f" Failed during: {msg} | Error: {e}")
        raise
# ------------------------------
# Decorator to log all methods of a class
# ------------------------------
def log_exceptions_all_methods(cls):
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            setattr(cls, attr_name, log_exceptions(attr))
    return cls
