# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import sys

import colorlog

# Logging conf
FORMAT = "%(asctime)-15s %(levelname)-9s%(threadName)-20s %(name)-15s: %(funcName)-20s: %(message)s"


COLOR_FORMAT = (
    "%(asctime_log_color)s%(asctime)-15s%(reset)s "
    "%(log_color)s%(levelname)-9s%(reset)s "
    "%(threadName_log_color)s%(threadName)-20s %(name)-15s: "
    "%(funcName)-20s%(reset)s: "
    "%(log_color)s%(message)s"
)


def uniform_color(color: str) -> dict:
    return dict.fromkeys(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], color)


def configure_logging() -> None:

    # Create a root logger
    import os

    logger = logging.getLogger()
    LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
    logging.basicConfig(level=LOGLEVEL, format=FORMAT)

    color_formatter = colorlog.ColoredFormatter(
        fmt=COLOR_FORMAT,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={
            "asctime": uniform_color("bold_purple"),
            "threadName": uniform_color("bold_white"),
            "name": uniform_color("bold_white"),
            "funcName": uniform_color("bold_white"),
        },
        style="%",
    )

    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with color only if output is a TTY
    console_handler = logging.StreamHandler(sys.stderr)

    # Since the logs of a remote ray process can be streamed to a terminal,
    # but it won't know if stderr is connected to a tty
    # we can't use stderr.isatty to determine if color should be on or off
    # instead we use the NO_COLOR envvar
    if not os.environ.get("NO_COLOR", None):
        console_handler.setFormatter(color_formatter)
    else:
        # Fallback to plain formatter if not a TTY or NO_COLOR has been requested
        console_handler.setFormatter(logging.Formatter(fmt=FORMAT))

    logger.addHandler(console_handler)
