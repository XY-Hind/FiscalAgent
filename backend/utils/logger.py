import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str = "fiscal_agent"):
    """
    Configures a robust logger with console and rotating file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if logger is already configured
    if logger.handlers:
        return logger

    # Log format: [TIMESTAMP] [LEVEL] [LOGGER] - MESSAGE
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] - %(message)s"
    )

    # 1. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Optional: useful for cloud logs persistence)
    log_dir = "logs"
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError:
            # Fallback if we can't create directory (e.g. read-only env)
            return logger

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10*1024*1024, # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Create a default logger instance
logger = setup_logger()
