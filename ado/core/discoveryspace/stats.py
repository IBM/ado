# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import math
from typing import TYPE_CHECKING, Annotated

import pydantic

if TYPE_CHECKING:
    from rich.status import Status

    from ado.core.discoveryspace.space import DiscoverySpace
    from ado.core.samplestore.base import ActiveSampleStore
    from ado.metastore.base import ResourceStore


class DiscoverySpaceStatistics(pydantic.BaseModel):
    """Aggregated statistics for a single discovery space.

    Lightweight fields are always populated. Heavy fields (those that require
    Python-side computation) are ``None`` when ``lightweight_only=True`` was
    requested or when the field is not applicable (e.g. the entity space is
    continuous or not defined).

    Attributes:
        number_of_experiments: Number of experiments configured in the space's
            measurement space.
        number_of_operations: Total number of operations linked to this space.
        number_of_explore_operations: Number of operations whose ``operationType``
            is ``explore``.
        number_measured_entities: DISTINCT entity IDs that appear in at least one
            measurement result across all operations on this space.
        size_of_entity_space: Total number of points in the entity space when the
            space is discrete. ``None`` if the space is continuous, not defined,
            or ``lightweight_only=True``.
        number_unmeasured_entities: ``size_of_entity_space`` minus
            ``number_measured_entities``. ``None`` when ``lightweight_only=True``.
            May be ``math.inf`` for continuous spaces or ``math.nan`` when
            ``size_of_entity_space`` could not be determined.
        number_matching_entities: Count of entities in the sample store that
            satisfy ``isEntityInSpace`` for this space. ``None`` when
            ``lightweight_only=True``.
        number_matching_entities_with_measurements: Subset of
            ``number_matching_entities`` that have at least one measurement whose
            experiment reference is in the space's measurement space. ``None``
            when ``lightweight_only=True``.
        entities_with_all_measurements: Entities in the space that have a result
            for every experiment in the measurement space. ``None`` when
            ``lightweight_only=True``.
        entities_with_partial_measurements: Entities with at least one result but
            not for every experiment in the measurement space. ``None`` when
            ``lightweight_only=True``.
        matching_entities_with_all_measurements: Matching entities in the sample
            store that have a result for every experiment in the measurement
            space. ``None`` when ``lightweight_only=True``.
    """

    # --- Lightweight fields (always populated) ---
    number_of_experiments: Annotated[
        int,
        pydantic.Field(
            description=(
                "Number of experiments configured in the space's measurement space."
            )
        ),
    ]
    number_of_operations: Annotated[
        int,
        pydantic.Field(description="Total number of operations linked to this space."),
    ]
    number_of_explore_operations: Annotated[
        int,
        pydantic.Field(
            description=("Number of operations whose operationType is 'explore'.")
        ),
    ]
    number_measured_entities: Annotated[
        int,
        pydantic.Field(
            description=(
                "DISTINCT entity IDs that appear in at least one measurement result "
                "across all operations on this space."
            )
        ),
    ]

    # --- Heavy fields (None when lightweight_only=True or not applicable) ---
    size_of_entity_space: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description=(
                "Total number of points in the entity space when discrete. "
                "None if the space is continuous, not defined, or lightweight_only=True."
            ),
        ),
    ]
    number_unmeasured_entities: Annotated[
        int | float | None,
        pydantic.Field(
            default=None,
            description=(
                "size_of_entity_space minus number_measured_entities. "
                "None when lightweight_only=True. "
                "math.inf for continuous spaces; math.nan when size_of_entity_space "
                "could not be determined."
            ),
        ),
    ]
    number_matching_entities: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description=(
                "Count of entities in the sample store satisfying isEntityInSpace. "
                "None when lightweight_only=True."
            ),
        ),
    ]
    number_matching_entities_with_measurements: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description=(
                "Subset of number_matching_entities that have at least one measurement "
                "whose experiment reference is in the space's measurement space. "
                "None when lightweight_only=True."
            ),
        ),
    ]
    entities_with_all_measurements: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description=(
                "Entities in the space that have a result for every experiment in "
                "the measurement space. None when lightweight_only=True."
            ),
        ),
    ]
    entities_with_partial_measurements: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description=(
                "Entities with at least one result but not for every experiment in "
                "the measurement space. None when lightweight_only=True."
            ),
        ),
    ]
    matching_entities_with_all_measurements: Annotated[
        int | None,
        pydantic.Field(
            default=None,
            description=(
                "Matching entities in the sample store that have a result for every "
                "experiment in the measurement space. None when lightweight_only=True."
            ),
        ),
    ]


def lightweight_space_statistics(
    space_ids: set[str],
    space_ids_to_operation_ids: dict[str, set[str]],
    metastore: "ResourceStore",
    sample_store: "ActiveSampleStore",
) -> "dict[str, DiscoverySpaceStatistics]":
    """Compute lightweight statistics for spaces given only their IDs and stores.

    Unlike :func:`space_statistics_for_spaces`, this function does not require
    :class:`~ado.core.discoveryspace.space.DiscoverySpace` instances —
    it only needs the space IDs, the operation-ID mapping, and the two stores.
    Heavy fields (``size_of_entity_space``, ``number_unmeasured_entities``,
    ``number_matching_entities``, ``number_matching_entities_with_measurements``)
    are always ``None``.

    Args:
        space_ids: Set of space URIs to compute statistics for.
        space_ids_to_operation_ids: Mapping from each space URI to the set of
            operation IDs that belong to that space.
        metastore: The :class:`~ado.metastore.base.ResourceStore` to
            query for operation/experiment counts.
        sample_store: The :class:`~ado.core.samplestore.base.ActiveSampleStore`
            to query for measured-entity counts.

    Returns:
        ``dict[space_id, DiscoverySpaceStatistics]`` keyed by space URI.
    """
    if not space_ids:
        return {}

    metastore_stats_by_space_id: dict[str, DiscoverySpaceStatistics] = (
        metastore.get_space_metastore_stats(space_ids)
    )
    sample_store_stats_by_space_id: dict[str, DiscoverySpaceStatistics] = (
        sample_store.space_entity_statistics(
            space_ids_to_operation_ids=space_ids_to_operation_ids,
        )
    )

    missing = space_ids - metastore_stats_by_space_id.keys()
    if missing:
        raise KeyError(f"Metastore returned no statistics for space(s): {missing}")
    missing = space_ids - sample_store_stats_by_space_id.keys()
    if missing:
        raise KeyError(f"Sample store returned no statistics for space(s): {missing}")

    return {
        space_id: DiscoverySpaceStatistics(
            number_of_experiments=metastore_stats_by_space_id[
                space_id
            ].number_of_experiments,
            number_of_operations=metastore_stats_by_space_id[
                space_id
            ].number_of_operations,
            number_of_explore_operations=metastore_stats_by_space_id[
                space_id
            ].number_of_explore_operations,
            number_measured_entities=sample_store_stats_by_space_id[
                space_id
            ].number_measured_entities,
            size_of_entity_space=None,
            number_unmeasured_entities=None,
            number_matching_entities=None,
            number_matching_entities_with_measurements=None,
            entities_with_all_measurements=None,
            entities_with_partial_measurements=None,
            matching_entities_with_all_measurements=None,
        )
        for space_id in space_ids
    }


def space_statistics_for_spaces(
    spaces: "list[DiscoverySpace]",
    lightweight_only: bool = False,
    spinner: "Status | None" = None,
) -> "dict[str, DiscoverySpaceStatistics]":
    """Compute statistics for multiple discovery spaces with minimal DB round-trips.

    Issues a single batched metastore query for all spaces, then one batched
    sample-store query per distinct sample store (spaces from different sample
    stores are handled correctly).

    Args:
        spaces: List of :class:`~ado.core.discoveryspace.space.DiscoverySpace`
            instances to summarise.  Spaces may belong to different sample stores.
        lightweight_only: When ``True`` skip all Python-side computation and
            return ``None`` for the heavy fields in every space's statistics.
        spinner: Optional rich status spinner to update with per-space progress
            messages during the heavy computation pass.

    Returns:
        ``dict[space_id, DiscoverySpaceStatistics]`` keyed by
        :attr:`~ado.core.discoveryspace.space.DiscoverySpace.uri`.
    """
    from ado.schema.entityspace import EntitySpaceRepresentation

    if not spaces:
        return {}

    # Group spaces by sample store so we can issue one batched query per store.
    spaces_by_sample_store: dict[str, list[DiscoverySpace]] = {}
    for space in spaces:
        spaces_by_sample_store.setdefault(space.sample_store.identifier, []).append(
            space
        )

    metastore = spaces[0].metadataStore

    lightweight_stats: dict[str, DiscoverySpaceStatistics] = {}
    for store_spaces in spaces_by_sample_store.values():
        group_space_ids = {s.uri for s in store_spaces}
        group_operation_ids = {s.uri: set(s.operations) for s in store_spaces}
        lightweight_stats.update(
            lightweight_space_statistics(
                space_ids=group_space_ids,
                space_ids_to_operation_ids=group_operation_ids,
                metastore=metastore,
                sample_store=store_spaces[0].sample_store,
            )
        )

    if lightweight_only:
        return lightweight_stats

    # ------------------------------------------------------------------
    # Heavy path — requires DiscoverySpace instances for Python-side work.
    # ------------------------------------------------------------------

    result: dict[str, DiscoverySpaceStatistics] = {}
    total = len(spaces)

    for idx, space in enumerate(spaces, start=1):
        if spinner is not None:
            spinner.update(
                f"Computing heavy statistics for {space.uri} ({idx}/{total})"
            )
        base = lightweight_stats[space.uri]
        number_measured = base.number_measured_entities

        # Heavy path
        size_of_entity_space: int | None = None
        if (
            space.entitySpace is not None
            and isinstance(space.entitySpace, EntitySpaceRepresentation)
            and space.entitySpace.isDiscreteSpace
        ):
            size_of_entity_space = space.entitySpace.size

        if space.entitySpace is None or not isinstance(
            space.entitySpace, EntitySpaceRepresentation
        ):
            number_unmeasured: int | float | None = math.nan
        elif not space.entitySpace.isDiscreteSpace:
            number_unmeasured = math.inf
        else:
            number_unmeasured = size_of_entity_space - number_measured

        matching_entities = space.matchingEntities()
        number_matching = len(matching_entities)

        measurement_exp_refs = set(space.measurementSpace.experimentReferences)
        experiments_in_measurement_space = len(space.measurementSpace.experiments)
        number_matching_with_measurements = sum(
            1
            for e in matching_entities
            if len(e.observedPropertyValues) > 0
            and not measurement_exp_refs.isdisjoint(set(e.experimentReferences))
        )
        matching_entities_with_all_measurements = sum(
            1
            for e in matching_entities
            if len(measurement_exp_refs.intersection(e.experimentReferences))
            == experiments_in_measurement_space
        )

        sampled_entities = space.sampledEntities()
        sampled_entities_with_all_measurements = sum(
            1
            for e in sampled_entities
            if len(e.observedPropertyValues) > 0
            and len(measurement_exp_refs.intersection(e.experimentReferences))
            == experiments_in_measurement_space
        )
        entities_with_partial_measurements = (
            number_measured - sampled_entities_with_all_measurements
        )

        result[space.uri] = DiscoverySpaceStatistics(
            number_of_experiments=base.number_of_experiments,
            number_of_operations=base.number_of_operations,
            number_of_explore_operations=base.number_of_explore_operations,
            number_measured_entities=number_measured,
            size_of_entity_space=size_of_entity_space,
            number_unmeasured_entities=number_unmeasured,
            number_matching_entities=number_matching,
            number_matching_entities_with_measurements=number_matching_with_measurements,
            entities_with_all_measurements=sampled_entities_with_all_measurements,
            entities_with_partial_measurements=entities_with_partial_measurements,
            matching_entities_with_all_measurements=matching_entities_with_all_measurements,
        )

    return result
