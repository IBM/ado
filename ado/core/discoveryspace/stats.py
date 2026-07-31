# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import math
from typing import TYPE_CHECKING, Annotated, NamedTuple

import pydantic

from ado.cli.utils.output.prints import ADO_SPINNER_GETTING_OUTPUT_READY
from ado.core.resources import CoreResourceKinds

if TYPE_CHECKING:
    from rich.status import Status

    from ado.core.discoveryspace.space import DiscoverySpace
    from ado.core.samplestore.base import ActiveSampleStore
    from ado.metastore.base import ResourceStore


class _SpaceSamplingState(NamedTuple):
    """Intermediate per-space values stashed at the end of Pass 1."""

    sampled_ids: set[str]
    matching_ids: set[str]
    size_of_entity_space: int | None
    number_unmeasured: int | float | None
    number_matching: int
    number_measured: int


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
    entity_ids_by_space_id: "dict[str, set[str]]",
    metastore: "ResourceStore",
) -> "dict[str, DiscoverySpaceStatistics]":
    """Compute lightweight statistics for spaces given only their IDs and stores.

    Heavy fields (``size_of_entity_space``, ``number_unmeasured_entities``,
    ``number_matching_entities``, ``number_matching_entities_with_measurements``)
    are always ``None``.

    Args:
        space_ids: Set of space URIs to compute statistics for.
        entity_ids_by_space_id: Mapping from each space URI to the set of
            entity IDs sampled in that space (empty set for spaces with no
            operations).
        metastore: The :class:`~ado.metastore.base.ResourceStore` to
            query for operation/experiment counts.

    Returns:
        ``dict[space_id, DiscoverySpaceStatistics]`` keyed by space URI.
    """
    if not space_ids:
        return {}

    metastore_stats_by_space_id: dict[str, DiscoverySpaceStatistics] = (
        metastore.get_space_metastore_stats(space_ids)
    )

    missing = space_ids - metastore_stats_by_space_id.keys()
    if missing:
        raise KeyError(f"Metastore returned no statistics for space(s): {missing}")

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
            number_measured_entities=len(entity_ids_by_space_id.get(space_id, set())),
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

    # Flat map: operation_id → space_id (each operation belongs to one space).
    entity_ids_by_space_id: dict[str, set[str]] = {}
    lightweight_stats: dict[str, DiscoverySpaceStatistics] = {}
    total = len(spaces)
    lightweight_idx = 0
    for store_spaces in spaces_by_sample_store.values():
        sample_store: ActiveSampleStore = store_spaces[0].sample_store
        group_space_ids: set[str] = {s.uri for s in store_spaces}
        for space in store_spaces:
            lightweight_idx += 1
            if spinner is not None:
                spinner.update(
                    f"Computing lightweight statistics for {space.uri} ({lightweight_idx}/{total})"
                )

        ops_by_space: dict[str, dict[CoreResourceKinds, set[str]]] = (
            metastore.get_resources_by_relationship(  # type: ignore[assignment]
                kind=CoreResourceKinds.DISCOVERYSPACE,
                identifier=group_space_ids,
                relationship="child",
                result_kinds={CoreResourceKinds.OPERATION},
                max_hops=1,
                identifiers_only=True,
            )
        )
        operation_id_to_space_id: dict[str, str] = {
            op_id: space_id
            for space_id, kinds in ops_by_space.items()
            for op_id in kinds.get(CoreResourceKinds.OPERATION, set())
        }
        group_entity_ids: dict[str, set[str]] = {s.uri: set() for s in store_spaces}
        if operation_id_to_space_id:
            entity_identifiers_by_operation = (
                sample_store.entity_identifiers_in_operations(
                    set(operation_id_to_space_id.keys()), group_by_operation=True
                )
            )
            for operation_id, ids in entity_identifiers_by_operation.items():
                group_entity_ids[operation_id_to_space_id[operation_id]].update(ids)
        entity_ids_by_space_id.update(group_entity_ids)
        group_stats: dict[str, DiscoverySpaceStatistics] = lightweight_space_statistics(
            space_ids=group_space_ids,
            entity_ids_by_space_id=group_entity_ids,
            metastore=metastore,
        )
        lightweight_stats.update(group_stats)

    if lightweight_only:
        return lightweight_stats

    # ------------------------------------------------------------------
    # Heavy path — requires DiscoverySpace instances for Python-side work.
    # ------------------------------------------------------------------

    # Intermediate results collected during Pass 1 (keyed by space URI).
    # Each entry stores everything needed to compute the final statistics once
    # the batched entity_experiment_references results are available.
    intermediate: dict[str, _SpaceSamplingState] = {}

    # Per-store accumulator: union of all entity IDs (matching + sampled) that
    # need experiment-reference lookups, keyed by sample-store identifier.
    entity_identifiers_by_sample_store: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Pass 1: entity fetching and matching — no entity_experiment_references
    # ------------------------------------------------------------------
    for idx, space in enumerate(spaces, start=1):
        if spinner is not None:
            spinner.update(
                f"Computing heavy statistics for {space.uri} ({idx}/{total})"
            )
        base = lightweight_stats[space.uri]
        number_measured = base.number_measured_entities

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
            # size_of_entity_space is always set when isDiscreteSpace is True
            number_unmeasured = (size_of_entity_space or 0) - number_measured

        # We already know which entities have been sampled in operations on this
        # space so there is no need to call sampledEntities().
        sampled_entity_ids = entity_ids_by_space_id.get(space.uri, set())
        sampled_entities = space.sample_store.get_entities(
            identifiers=sampled_entity_ids, require_measurements=False
        )

        # Filter to entities that actually belong to this space. This also
        # confirms the entity was found in the store — IDs not present in the
        # store are silently dropped by get_entities, so any entity that
        # survived to this point was genuinely retrieved.
        sampled_entities = [
            e for e in sampled_entities if space.entitySpace.isEntityInSpace(e)
        ]

        # If the space has been completely sampled, sampled entities == matching
        # entities, so we can skip the more expensive matchingEntities() call.
        if (
            size_of_entity_space is not None
            and len(sampled_entities) == size_of_entity_space
        ):
            matching_entities = sampled_entities
        else:
            matching_entities = space.matchingEntities(require_measurements=False)

        # Re-derive IDs from the filtered entity objects rather than reusing
        # sampled_entity_ids, because the filter above may have dropped some.
        sampled_ids = {
            e.identifier for e in sampled_entities if e.identifier is not None
        }
        matching_ids = {
            e.identifier for e in matching_entities if e.identifier is not None
        }

        # Accumulate all entity IDs that will need experiment-reference lookups
        # for this store group, to be queried in a single batch after Pass 1.
        store_id = space.sample_store.identifier
        if store_id not in entity_identifiers_by_sample_store:
            entity_identifiers_by_sample_store[store_id] = set()

        entity_identifiers_by_sample_store[store_id].update(matching_ids)
        entity_identifiers_by_sample_store[store_id].update(sampled_ids)

        intermediate[space.uri] = _SpaceSamplingState(
            sampled_ids=sampled_ids,
            matching_ids=matching_ids,
            size_of_entity_space=size_of_entity_space,
            number_unmeasured=number_unmeasured,
            number_matching=len(matching_entities),
            number_measured=number_measured,
        )

    if spinner:
        spinner.update(ADO_SPINNER_GETTING_OUTPUT_READY)

    # ------------------------------------------------------------------
    # Batch: one entity_experiment_references call per sample-store group
    # ------------------------------------------------------------------
    exp_refs_by_store: dict[str, dict] = {}
    for store_id, entity_identifiers in entity_identifiers_by_sample_store.items():
        store = spaces_by_sample_store[store_id][0].sample_store
        exp_refs_by_store[store_id] = store.entity_experiment_references(
            entity_identifiers
        )

    # ------------------------------------------------------------------
    # Pass 2: Python-only — distribute batched results to each space
    # ------------------------------------------------------------------
    result: dict[str, DiscoverySpaceStatistics] = {}

    for space in spaces:
        base = lightweight_stats[space.uri]
        r = intermediate[space.uri]
        sampled_ids = r.sampled_ids
        matching_ids = r.matching_ids
        size_of_entity_space = r.size_of_entity_space
        number_unmeasured = r.number_unmeasured
        number_matching = r.number_matching
        number_measured = r.number_measured

        store_id = space.sample_store.identifier
        store_exp_refs = exp_refs_by_store.get(store_id, {})

        # Slice the store-level batch result down to only the entities this space needs.
        exp_refs_by_matching_entity = {
            k: store_exp_refs[k] for k in matching_ids if k in store_exp_refs
        }
        exp_refs_by_sampled_entity = {
            k: store_exp_refs[k] for k in sampled_ids if k in store_exp_refs
        }

        exp_refs_in_measurement_space = set(space.measurementSpace.experimentReferences)
        experiments_in_measurement_space = len(space.measurementSpace.experiments)

        number_matching_with_measurements = 0
        matching_entities_with_all_measurements = 0
        for entity_id in matching_ids:
            entity_exp_refs = exp_refs_by_matching_entity.get(entity_id, set())
            n_measured = len(
                exp_refs_in_measurement_space.intersection(entity_exp_refs)
            )
            if n_measured > 0:
                number_matching_with_measurements += 1
            if n_measured == experiments_in_measurement_space:
                matching_entities_with_all_measurements += 1

        sampled_entities_with_all_measurements = sum(
            1
            for entity_id in sampled_ids
            if len(
                exp_refs_in_measurement_space.intersection(
                    exp_refs_by_sampled_entity.get(entity_id, set())
                )
            )
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
