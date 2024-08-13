from afqmc.logging import Logger


def main():
    logger = Logger()
    logger.info("Begin AFQMC")
    logger.debug("Print a debug message")
    logger.warning("Print a warning message")
    logger.error("Print an error message")
    logger.critical("Print message for critical error")
    logger.info("End AFQMC")
