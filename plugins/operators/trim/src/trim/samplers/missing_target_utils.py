# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

from orchestrator.core.discoveryspace.space import DiscoverySpace
from trim.samplers.no_priors_parameters import (
    BaseTrimSamplerParameters,
    MissingTargetMode,
)
from trim.utils.exceptions import InsufficientDataError
from trim.utils.logging_utils import (
    log_unable_to_proceed_with_iterative_modeling_and_raise_error,
)

logger = logging.getLogger(__name__)


def record_missing_and_check_budget(
    params: BaseTrimSamplerParameters,
    entity_id: str,
    missing_count: int,
    discoverySpace: DiscoverySpace,
    additional_info: str = "",
) -> int:
    """Record a missing-target entity and enforce the budget / mode policy.

    Called by both ``NoPriorsSampleSelector`` and ``TrimSampleSelector`` whenever
    a yielded entity produced no measurement for the target variable.

    Behaviour by mode:

    - ``MissingTargetMode.RaiseError``: delegates to
      :func:`~trim.utils.logging_utils.log_unable_to_proceed_with_iterative_modeling_and_raise_error`
      (never returns).
    - ``MissingTargetMode.InjectDefaultValue`` / ``MissingTargetMode.Skip``:
      appends ``entity_id`` to
      ``params.missing_target_variables.no_target_variable_entities``, increments
      ``missing_count``, and raises
      :class:`~trim.utils.exceptions.InsufficientDataError` when ``missing_count``
      exceeds ``budget`` (if set).

    Args:
        params: The sampler parameters containing the ``missing_target_variables``
            policy.  Must expose a ``targetOutput`` attribute (present on both
            :class:`~trim.samplers.no_priors_parameters.NoPriorsParameters` and
            :class:`~trim.trim_pydantic.TrimParameters`).
        entity_id: Identifier of the entity that did not produce a target
            measurement.
        missing_count: Current count of missing-target entities seen so far.
        discoverySpace: The active discovery space (used for the error message).
        additional_info: Extra context string appended to the error message.

    Returns:
        The updated ``missing_count`` (incremented by one).

    Raises:
        InsufficientDataError: When mode is ``RaiseError`` or the budget is
            exceeded.
    """
    mtv = params.missing_target_variables

    if mtv.mode == MissingTargetMode.RaiseError:
        log_unable_to_proceed_with_iterative_modeling_and_raise_error(
            discoverySpace=discoverySpace,
            target_output=params.targetOutput,  # type: ignore[attr-defined]
            additional_info=additional_info,
        )

    # InjectDefaultValue or Skip — record and check budget
    mtv.no_target_variable_entities.append(entity_id)
    missing_count += 1

    budget = mtv.budget
    if budget is not None and missing_count > budget:
        msg = (
            f"The number of entities that did not produce a target measurement "
            f"({missing_count}) exceeds the configured budget ({budget}). "
            f"Entity '{entity_id}' was the last to trigger this limit. "
            f"{additional_info}"
        )
        logger.error(msg)
        raise InsufficientDataError("Missing-target budget exceeded.\n\n" + msg)

    return missing_count
