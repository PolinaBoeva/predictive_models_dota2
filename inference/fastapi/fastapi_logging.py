import logging

from config import get_config


def get_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(get_config().log_config.log_level)
    handler.setFormatter(logging.Formatter(get_config().log_config.log_format))

    return handler


def get_rotating_file_handler() -> logging.FileHandler:
    handler = logging.handlers.RotatingFileHandler(
        get_config().log_config.log_file,
        maxBytes=get_config().log_config.log_max_bytes,
        backupCount=get_config().log_config.log_backup_count,
    )
    handler.setLevel(get_config().log_config.log_level)
    handler.setFormatter(logging.Formatter(get_config().log_config.log_format))

    return handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(get_config().log_config.log_level)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_rotating_file_handler())

    return logging.getLogger(name)

def close_logger(logger: logging.Logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)