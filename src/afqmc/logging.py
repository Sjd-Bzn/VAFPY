import logging


class Logger:
    """Deal consistently with output produced by AFQMC."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename="afqmc.log", filemode="w", level=logging.INFO)

    def debug(self, message):
        "Debug messages should typically not be needed except to trace the origin of errors."
        self.logger.debug(message)

    def info(self, message):
        "Info messages inform about correct working of the code."
        self.logger.info(message)

    def warning(self, message):
        "Warning messages indicate possible issues with the code that should not affect the result."
        self.logger.warning(message)

    def error(self, message):
        "Error messages inform that a certain part of the code failed but the code can still continue."
        self.logger.error(message)

    def critical(self, message):
        "Critical errors require to stop the program because the results would be incorrect."
        self.logger.error(message)
