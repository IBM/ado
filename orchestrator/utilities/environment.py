# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import logging
import os


def enable_ray_actor_coverage(identifier: str) -> None:
    """For coverage to work with distributed ray actors they need to call this function in their init

    If coverage module is not installed or COVERAGE_PROCESS_START is not defined in the environment
    this function does nothing"""

    if "COVERAGE_PROCESS_START" in os.environ:
        # Don't start multiple times in the same process
        if not globals().get("_coverage_started", False):
            try:
                import coverage
            except ImportError:
                logging.warning(
                    f"{identifier}: COVERAGE_PROCESS_START is defined in the environment but the coverage module is not installed"
                )
            else:
                logging.debug(f"Starting coverage for {identifier}")
                coverage.process_startup()
                globals()["_coverage_started"] = True
        else:
            logging.debug(
                f"Requested to start coverage for {identifier} but _coverage_started is already set"
            )
