import logging
import logging.config


# Logging Config Dict

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "()": "core.logging.ColouredFormatter",
            "datefmt": "%I:%M:%S",
            "style": "{",
            "format": "{log_color}[{levelname:8s}]: [{name:<20s}:{lineno:d}]: {message}",
            "log_colors": {
                "INFO": "light_grey",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red",
            },
        },
    },
    "filters": {
        "under_level": {
            "()": "core.logging.UnderLevelFilter",
            "upper_level": "WARNING",
        }
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "urllib3": {
            "disabled": True,
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "matplotlib": {"disabled": True, "level": "INFO"},
        "faker": {"disabled": True, "level": "INFO"},
        "": {"level": "INFO", "handlers": ["console"], "propagate": False},
    },
}


class UnderLevelFilter(object):
    """Logging filter to ignore all the logs with level above or equal to
    `self.upper_level`.
    """

    def __init__(self, upper_level):
        self.upper_level = upper_level

    def filter(self, record):
        """Filters out logs with level above or equal to `self.upper_level`."""
        return self.get_log_level_name(record.levelno) < self.upper_level

    def get_log_level_name(self, levelno):
        """Converts log level string to it's corresponding int value as
        defined in the logging package.
        """
        switcher = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}
        return switcher.get(levelno, "INVALID")


COLORS = {
    "dark_grey": "\x1b[90m",
    "light_grey": "\x1b[37m",
    "cyan": "\x1b[36;21m",
    "white": "\x1b[97m",
    "grey": "\x1b[38;21m",
    "yellow": "\x1b[33;21m",
    "red": "\x1b[31;21m",
    "bold_red": "\x1b[31;1m",
}


class ColouredFormatter(logging.Formatter):
    """Colored logging formatter"""

    # Colors escape sequences
    DEFAULT_COLOR = "\x1b[39;21m"
    DEFAULT_COLORS = {
        "DEBUG": DEFAULT_COLOR,
        "INFO": COLORS["light_grey"],
        "WARNING": COLORS["yellow"],
        "ERROR": COLORS["red"],
        "CRITICAL": COLORS["red"],
    }
    RESET_SEQUENCE = "\x1b[0m"

    def __init__(
        self,
        fmt=None,
        datefmt=None,
        style="%",
        log_colors=None,
        reset=True,
    ):
        super().__init__(fmt, datefmt, style)
        self.log_colors = self.DEFAULT_COLORS
        if log_colors is not None:
            self.log_colors = {
                level: COLORS.get(color_name, self.DEFAULT_COLOR)
                for level, color_name in log_colors.items()
            }

    def format(self, record):
        record.log_color = self.log_colors.get(record.levelname, self.DEFAULT_COLOR)
        message = logging.Formatter.format(self, record)
        if not message.endswith(self.RESET_SEQUENCE):
            message += self.RESET_SEQUENCE
        return message


def init_logger():
    logging.config.dictConfig(LOGGING)
