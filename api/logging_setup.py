from __future__ import annotations
import logging
import sys
from pythonjsonlogger import jsonlogger

def setup_json_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("mlops_api")
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(fmt)
    logger.handlers = [handler]
    logger.propagate = False
    return logger
