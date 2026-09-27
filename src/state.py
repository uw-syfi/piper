import sys
import logging

LOG_LEVEL = "INFO"

"""
Logger utility
"""

def create_logger(name: str, log_level: str):
    match log_level:
        case "DEBUG":
            log_level = logging.DEBUG
        case "INFO":
            log_level = logging.INFO
        case "WARNING":
            log_level = logging.WARNING
        case "ERROR":
            log_level = logging.ERROR

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
