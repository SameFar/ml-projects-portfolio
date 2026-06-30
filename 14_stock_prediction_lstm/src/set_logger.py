import logging

def make_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Terminal — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
    ))

    logger.addHandler(ch)
    
