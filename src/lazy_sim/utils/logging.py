"""Ersilia-style logging: a stdlib logger printed through Rich, plus a success level."""

import logging

from rich.console import Console
from rich.logging import RichHandler

SUCCESS = 25
console = Console(stderr=True)
logging.addLevelName(SUCCESS, "SUCCESS")


class Logger:
    """Singleton logger for lazy-sim.

    Messages go to stderr through Rich, so the standard output stays clean.

    Methods
    -------
    debug(msg, *args), info(msg, *args), warning(msg, *args), error(msg, *args)
        Log at the usual levels, with %-style arguments.
    success(msg, *args)
        Log at the SUCCESS level, between INFO and WARNING.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            log = logging.getLogger("lazysim")
            log.setLevel(logging.INFO)
            log.propagate = False
            if not log.handlers:
                log.addHandler(RichHandler(console=console, show_path=False))
            cls._instance._log = log
        return cls._instance

    def debug(self, msg, *args):
        self._log.debug(msg, *args)

    def info(self, msg, *args):
        self._log.info(msg, *args)

    def warning(self, msg, *args):
        self._log.warning(msg, *args)

    def error(self, msg, *args):
        self._log.error(msg, *args)

    def success(self, msg, *args):
        self._log.log(SUCCESS, msg, *args)


logger = Logger()
