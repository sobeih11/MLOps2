import logging
import os

def get_logger(name: str, log_file: str = "logs/train.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
