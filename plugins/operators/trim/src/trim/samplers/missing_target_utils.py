# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

import pandas as pd

from orchestrator.core.discoveryspace.space import DiscoverySpace, Entity
from trim.samplers.no_priors_parameters import (
    BaseTrimSamplerParameters,
    MissingTargetMode,
)
from trim.utils.exceptions import InsufficientDataError
from trim.utils.logging_utils import (
    log_unable_to_proceed_with_iterative_modeling_and_raise_error,
)

logger = logging.getLogger(__name__)


def entity_measured_target(
    entity: Entity,
    discoverySpace: DiscoverySpace,
    target_output: str,
) -> tuple[bool, "pd.Series"]:
    """Check whether *entity* has a non-null measurement for *target_output*.

    Looks up the entity directly from the sample store via
    :meth:`~orchestrator.core.discoveryspace.space.DiscoverySpace.entity_for_point`
    and inspects its
    :meth:`~orchestrator.schema.entity.Entity.seriesRepresentation`.  This avoids
    rebuilding the full source DataFrame on every post-yield check.

    Args:
        entity: The entity that was just yielded and measured.
        discoverySpace: The active discovery space (re-fetches the stored entity
            with up-to-date measurement results).
        target_output: Identifier of the target property column.

    Returns:
        A ``(hit, series)`` tuple where *hit* is ``True`` when the stored entity
        has a non-null value for *target_output*, and *series* is the full
        :class:`pandas.Series` from ``seriesRepresentation`` (useful for
        constructing a DataFrame row without a second store call).
    """
    point = {
        cpv.property.identifier: cpv.value
        for cpv in entity.constitutive_property_values
    }
    stored = discoverySpace.entity_for_point(point)  # type: ignore[arg-type]
    series = stored.seriesRepresentation(experimentReferences=None)
    target_val = series.get(target_output)
    hit = target_val is not None and not (
        # pd.isna returns a scalar bool for scalar input; for array-like
        # values (rare) treat the entity as measured if any value is non-null.
        pd.isna(target_val)
        if not hasattr(target_val, "__len__")
        else all(pd.isna(v) for v in target_val)
    )
    return hit, series


def _entity_mask(entity: Entity, source_df: pd.DataFrame) -> "pd.Series | bool":
    """Return a boolean mask selecting rows in *source_df* that match *entity*.

    Matches on constitutive property values to avoid int/float identifier
    formatting mismatches after a DB round-trip.
    """
    mask: bool | pd.Series = True
    for cpv in entity.constitutive_property_values:
        col = cpv.property.identifier
        if col in source_df.columns:
            mask = mask & (source_df[col] == cpv.value)
    return mask


def entity_hit_in_source(entity: Entity, source_df: pd.DataFrame) -> bool:
    """Return True if *entity* has a row in *source_df*.

    Matches on constitutive property values rather than the identifier string
    to avoid false misses caused by int/float formatting differences.  After a
    DB round-trip pydantic coerces ``numpy.int64`` values to ``float``, turning
    an identifier like ``"foo.60"`` into ``"foo.60.0"``, which would never
    match the freshly-constructed pool identifier.

    Args:
        entity: The entity to look up.
        source_df: DataFrame returned by ``get_source_and_target`` (only rows
            with a non-null target output, columns include all constitutive
            property identifiers).

    Returns:
        ``True`` if a row whose constitutive property columns all match the
        entity's values is found in *source_df*.
    """
    if source_df.empty:
        return False
    mask = _entity_mask(entity, source_df)
    return bool(mask.any()) if not isinstance(mask, bool) else False


def entity_row_in_source(entity: Entity, source_df: pd.DataFrame) -> pd.DataFrame:
    """Return the row(s) in *source_df* that correspond to *entity*.

    Args:
        entity: The entity to look up.
        source_df: DataFrame returned by ``get_source_and_target``.

    Returns:
        A (possibly empty) sub-DataFrame of the matching rows.
    """
    if source_df.empty:
        return source_df
    mask = _entity_mask(entity, source_df)
    if isinstance(mask, bool):
        return source_df.iloc[0:0]  # empty with same columns
    return source_df[mask]


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
