import sys
import logging
from typing import Optional

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




"""
Piper thread local storage for tracking Piper actors, stages, and microbatches
"""

class PiperMetadata:
    actors = dict()
    visualize_dag: bool = False  # Whether to render per-rank DAG PNGs after compilation
    artifact_dir: str = "out"  # Directory for debug artifacts emitted during runs
    training_dag = None  # DAG of annotated model segments and transform-inserted nodes
    per_pp_training_dags = None  # One local DAG per schedule ActorGroup
    schedule_directives: list = []  # Program of DAG transform directives (e.g., place(...))
    schedule_directives_file: Optional[str] = None  # JSON source for schedule_directives
    schedule_info: dict = {}  # Derived ActorGroup and microbatch schedule facts
    profiled_training_plan = None  # Latest merged interference-free profile artifact

piper_metadata = PiperMetadata()
