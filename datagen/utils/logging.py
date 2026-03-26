import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logger = logging.getLogger("mum")
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def get_logger(name: str = "mum") -> logging.Logger:
    return logging.getLogger(name)
