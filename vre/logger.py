"""Data store global logger."""

import os
import logging

# Usage: loglevel=0 (none), loglevel=1 (info), loglevel=2 (debug), loglevel=3 (debug verbose)
logging.DEBUG2 = 3
logging.addLevelName(logging.DEBUG2, "DBG2")
all_levels = {0: logging.NOTSET, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG2}


# pylint: disable=protected-access
def _debug2(self, message, *args, **kws):
    if self.isEnabledFor(logging.DEBUG2):
        self._log(logging.DEBUG2, message, args, **kws)


logging.Logger.debug2 = _debug2


class CustomFormatter(logging.Formatter):
    """Custom formatting for the logger."""

    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    pink = "\x1b[35;20m"

    pre = "[%(asctime)s-%(name)s-%(levelname)s]"
    post = "(%(filename)s:%(funcName)s:%(lineno)d)"

    # Example [TIME:LEVEL:NAME] Message [FILE:FUNC:LINE]
    FORMATS = {
        logging.DEBUG: f"{cyan}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.DEBUG2: f"{cyan}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.INFO: f"{green}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.WARNING: f"{pink}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.ERROR: f"{red}{pre}{reset} %(message)s {yellow}{post}{reset}",
        logging.CRITICAL: f"{bold_red}{pre}{reset} %(message)s {yellow}{post}{reset}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatter.formatTime = self.formatTime
        return formatter.format(record)

    def formatTime(self, record, datefmt=None):
        return super().formatTime(record, "%Y%m%d %H:%M:%S")


def build_logger(key: str) -> logging.Logger:
    """Returns a custom logger with the specified key."""
    # instantiate logger and set log level
    _logger = logging.getLogger(key)
    # default log level is INFO, change it via the env variable
    log_level = int(os.environ.get(f"{key}_LOGLEVEL", 1))
    _logger.setLevel(all_levels[log_level])
    # add custom formatter to logger
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    _logger.addHandler(handler)
    return _logger

# exported module logger
logger = build_logger("VRE")
