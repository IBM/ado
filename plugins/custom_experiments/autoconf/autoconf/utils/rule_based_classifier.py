# Copyright (c) IBM Corporation

# SPDX-License-Identifier: MIT

import logging

import pandas as pd

# Configure logging

logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],  # Output to console
)

logger = logging.getLogger(__name__)


def to_series(x: pd.Series | pd.DataFrame | dict) -> pd.Series:
    if isinstance(x, pd.Series):
        return x

    if isinstance(x, pd.DataFrame):
        if len(x) != 1:
            raise ValueError(f"DataFrame must have exactly 1 row, got {len(x)}")
        return x.iloc[0]

    if isinstance(x, dict):
        s = pd.Series(x)
        if s.empty:
            raise ValueError("Config from dict cannot be empty")
        return s

    raise TypeError(f"Expected Series, DataFrame, or dict, got {type(x).__name__}")


def is_row_valid(
    config: pd.Series | pd.DataFrame | dict,
    err_prefix: str = "Rule-based classifier error: ",
) -> tuple[bool, list[str]]:
    """
    Applies to rows a rule-based classification
    """
    errors = []
    config = to_series(config)

    # Rule 1
    try:
        if config["batch_size"] % config["number_gpus"] != 0:
            errors.append(
                err_prefix + "batch_size must be evenly divisible by number_gpus."
            )
    except Exception as e:
        except_string = """Rule based on divisibility of 'batch_size' by 'number_gpus'
                    cannot be applied, probably because number_gpus has not been specified on the config"""
        logger.info(f"{e}:{except_string}")
        return False, errors

    return len(errors) == 0, errors
