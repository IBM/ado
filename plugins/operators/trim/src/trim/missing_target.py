# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

from trim.trim_pydantic import MissingTargetMeasurementMode, MissingTargetMeasurements
from trim.utils.exceptions import InsufficientDataError


def record_unmeasured_entity(
    entity_identifier: str,
    missing_target_measurements: MissingTargetMeasurements,
    total_unmeasured: int,
    target_output: str,
    additional_info: str,
    logger: logging.Logger,
) -> None:
    """Log a warning or raise an error when an entity lacks a targetOutput measurement.

    Applies the ``missingTargetMeasurements`` policy:

    - If ``mode`` is ``Error``, or if the number of missing measurements has
      reached or exceeded the configured ``budget``, logs at ERROR level and
      raises ``InsufficientDataError``.
    - Otherwise logs at WARNING level with the current missing-measurement count
      relative to the budget (displayed as ``"unlimited"`` when ``budget`` is
      ``None``).

    Args:
        entity_identifier: The entity whose measurement lacked the target output.
        additional_info: Extra context to append to the log message.
        missing_target_measurements: Policy governing how missing target
            measurements are handled (mode, budget, defaultValue).
        total_unmeasured: Running count of entities seen so far that lacked the
            target output measurement (including this one).
        target_output: Bare identifier of the expected target output property.
        logger: Logger to write the message to.

    Raises:
        InsufficientDataError: When ``mode`` is ``Error``, or when
            ``total_unmeasured`` has reached or exceeded the configured
            ``budget``.
    """
    msg = f"The measurements obtained for {entity_identifier} did not contain the target output property '{target_output}'."

    if additional_info:
        msg += f" Additional info: {additional_info}."

    if missing_target_measurements.mode == MissingTargetMeasurementMode.Error or (
        missing_target_measurements.budget is not None
        and missing_target_measurements.budget <= total_unmeasured
    ):
        logger.error(msg)
        raise InsufficientDataError(msg)

    budget: int | str
    if missing_target_measurements.budget is None:
        budget = "unlimited"
    else:
        budget = missing_target_measurements.budget

    msg += f"Missing Target Measurements budget: {total_unmeasured}/{budget}"

    logger.warning(msg)
