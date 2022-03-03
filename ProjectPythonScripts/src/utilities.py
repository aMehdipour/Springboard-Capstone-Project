import logging
import tensorflow as tf
from pathlib import Path


def set_logger(log_path):
    """
    Purpose:
        Creates a logger and sets the path to log files to.
    Arguments:
        log_path [str]: File path where logging files are to be saved
    Returns:
        logger
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger


def load_model(model_path):
    """
    Purpose:
        Load in an already trained neural net
    Arguments:
        model_path [str]: File path of saved, trained model
    Returns:
        model: Loaded trained model
    """
    model_path = Path(model_path)
    model = tf.keras.models.load_model(model_path)
    return model
