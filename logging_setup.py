import logging
import threading

def setup_logging():
    logger = logging.getLogger('TradingSystem')
    logger.setLevel(logging.INFO)
    
    # Create a file handler for logging
    file_handler = logging.FileHandler('trading_system.log')
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Thread-safe logging setup
logging_lock = threading.Lock()
logger = setup_logging()