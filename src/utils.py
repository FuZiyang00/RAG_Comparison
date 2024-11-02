import logging
import re
from typing import List
from src.constants import LOG_FILE_PATH

def setup_logging() -> None:
    """
    Configures logging settings for the application, specifying log file, format, and level.
    """
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )