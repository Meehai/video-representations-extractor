"""
Python logger settings.
Uses ENv variables to control the log level:

YOUR_KEY_LOGLEVEL=4 python blabla.py
and logger.debug4("message")

"""

import os
import sys
import logging
from colorama import Fore, Back, Style

KEY = "VRE"
ENV_KEY = f"{KEY}_LOGLEVEL"
# defaults to -1 (no logger!).
env_var = int(os.environ[ENV_KEY]) if ENV_KEY in os.environ else -1

# we need numbers below 5 (last logging module used number)
logging.DEBUG2 = 3
logging.DEBUG3 = 2
logging.DEBUG4 = 1
try:
    loglevel = {
        -1: logging.NOTSET,
        0: logging.INFO,
        1: logging.DEBUG,
        2: logging.DEBUG2,
        3: logging.DEBUG3,
        4: logging.DEBUG4,
    }[env_var]
except KeyError:
    sys.stderr.write(f"You tried to use {KEY}_LOGLEVEL={env_var}. You need to set it between -1 and 4\n")
    sys.exit(1)
# add the custom ones in the logger
logging.addLevelName(logging.DEBUG2, "DGB2")
logging.addLevelName(logging.DEBUG3, "DGB3")
logging.addLevelName(logging.DEBUG4, "DGB4")


class CustomFormatter(logging.Formatter):
    """Custom formatting for logger."""

    reset = Style.RESET_ALL
    pre = "[%(asctime)s-%(name)s-%(levelname)s]"
    post = "(%(filename)s:%(funcName)s:%(lineno)d)"

    # Example [TIME:LEVEL:NAME] Message [FILE:FUNC:LINE]. We can update some other format here easily
    FORMATS = {
        logging.DEBUG: f"{Fore.CYAN}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
        logging.DEBUG2: f"{Fore.CYAN}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
        logging.DEBUG3: f"{Fore.MAGENTA}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
        logging.DEBUG4: f"{Back.RED}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
        logging.INFO: f"{Fore.GREEN}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
        logging.WARNING: f"{Fore.YELLOW}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
        logging.ERROR: f"{Fore.RED}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
        logging.CRITICAL: f"{Back.RED}{pre}{reset} %(message)s {Fore.YELLOW}{post}{reset}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        formatter.formatTime = self.formatTime
        return formatter.format(record)

    # here we define the time format.
    def formatTime(self, record, datefmt=None):
        return super().formatTime(record, "%Y%m%d %H:%M:%S")


# instantiate logger and set log level


def _debug2(self, message, *args, **kws):
    if self.isEnabledFor(logging.DEBUG2):
        self._log(logging.DEBUG2, message, args, **kws)


def _debug3(self, message, *args, **kws):
    if self.isEnabledFor(logging.DEBUG3):
        self._log(logging.DEBUG3, message, args, **kws)


def _debug4(self, message, *args, **kws):
    if self.isEnabledFor(logging.DEBUG4):
        self._log(logging.DEBUG4, message, args, **kws)


logger = logging.getLogger(KEY)
logger.setLevel(loglevel)
logging.Logger.debug2 = _debug2
logging.Logger.debug3 = _debug3
logging.Logger.debug4 = _debug4

# add custom formatter to logger
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)
