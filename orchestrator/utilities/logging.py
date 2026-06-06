# Copyright IBM Corporation 2025, 2026
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

# Guard flag to make configure_logging() idempotent within a process.
# Many modules call configure_logging() at module level so that they work
# correctly when imported directly (e.g. inside Ray workers).  In the test
# suite those same modules may be lazily imported for the first time inside a
# typer CliRunner.invoke() call, at which point sys.stderr has been replaced by
# the runner's captured-output buffer.  Without this guard the root logger
# handler would point to that temporary buffer; once the invoke() returns and
# the buffer is closed, every subsequent log call raises
# "ValueError: I/O operation on closed file." and pollutes following invocations
# with "--- Logging error ---" text captured via Python's handleError() fallback.
_logging_configured = False


def uniform_color(color: str) -> dict:
    """Return a dict mapping every log level name to *color*."""
    return dict.fromkeys(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], color)


def configure_logging() -> None:
    """Configure the root logger with a colorized (or plain) stream handler.

    Idempotent: the full handler setup only runs once per process.  Subsequent
    calls merely update the root logger's effective level from the ``LOGLEVEL``
    environment variable, which is safe to do at any time.
    """
    global _logging_configured

    import os

    logger = logging.getLogger()
    LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()

    if _logging_configured:
        logger.setLevel(LOGLEVEL)
        return

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
    _logging_configured = True
