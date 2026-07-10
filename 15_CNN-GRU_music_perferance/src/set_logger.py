import logging


def make_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Terminal — DEBUG and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )

    logger.addHandler(ch)

    # SILENCE NUMBA HERE:
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("torchaudio").setLevel(logging.WARNING)
