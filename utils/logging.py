import logging


def setup_logging(log_file=None):
    logger = logging.getLogger()
    logger.handlers.clear() # clear existing handlers
    logger.setLevel(logging.INFO)

    # Formatting
    formatter = logging.Formatter("[%(asctime)s](%(levelname)s) %(message)s", datefmt="%H:%M:%S")

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
