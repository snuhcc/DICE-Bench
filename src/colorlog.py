import logging
from colorlog import ColoredFormatter


# ---------------------------------------------------------------------------
# Shared ColoredFormatter (created once and reused by all project loggers)
# ---------------------------------------------------------------------------
_LOG_FORMAT = (
    "%(log_color)s[%(levelname)s]%(reset)s %(blue)s%(name)s:%(reset)s %(message)s"
)

_COLOR_FORMATTER = ColoredFormatter(
    _LOG_FORMAT,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)


def _add_stream_handler(logger: logging.Logger, level: int) -> None:
    """Attach a stream handler with the shared color formatter (idempotent)."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_COLOR_FORMATTER)
    logger.addHandler(handler)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a project-wide colorized logger.

    If a logger with the given *name* already has handlers attached, we simply
    return it. Otherwise, we configure it with a single colored StreamHandler.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        _add_stream_handler(logger, level)
    return logger


# Create a module-level logger for this utility module itself
logger = get_logger(__name__)
