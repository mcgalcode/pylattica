"""Tools for logging."""

import logging

__all__ = ["initialize_logger"]


LOGGER_NAME = "rxn-ca"

def get_logger():
    return logging.getLogger(LOGGER_NAME)

def initialize_logger(level: int = logging.INFO) -> logging.Logger:
    """Initialize the default logger.
    Parameters
    ----------
    level
        The log level.
    Returns
    -------
    Logger
        A logging instance with customized formatter and handlers.
    """
    import sys

    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(level)
    log.handlers = []  # reset logging handlers if they already exist

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(fmt)
    log.addHandler(screen_handler)
    return log