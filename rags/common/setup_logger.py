from loguru import logger


def setup_logger(log_level: str = "INFO"):
    """
    Set up the logger with the specified log level.

    Args:
        log_level (str): The log level to set. Default is "INFO".
    """
    logger.remove()  # Remove default logger
    logger.add(
        sink=lambda msg: print(msg, end=''),
        level=log_level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
