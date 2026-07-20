# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import datetime
import json
import math
import typing

import pydantic
import typer
import yaml

from ado.cli.models.types import AdoGetSupportedOutputFormats
from ado.cli.utils.generic.constants import (
    SECONDS_IN_A_DAY,
    SECONDS_IN_A_MINUTE,
    SECONDS_IN_AN_HOUR,
)
from ado.cli.utils.jsonpath.filters import remove_fields_from_dictionary
from ado.cli.utils.output.prints import (
    ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE,
    ERROR,
    WARN,
    console_print,
)
from ado.cli.utils.pydantic.constants import (
    event_importance_order,
    minimize_output_context,
)
from ado.core import (
    ADOResource,
    CoreResourceKinds,
    DiscoverySpaceResource,
    OperationResource,
)
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.metadata import ConfigurationMetadata
from ado.core.operation.resource import (
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from ado.core.resources import ADOResourceEventEnum, ADOResourceStatus
from ado.schema.domain import VariableTypeEnum
from ado.utilities.output import (
    printable_pydantic_model,
)
from ado.utilities.pandas import reorder_dataframe_columns

if typing.TYPE_CHECKING:
    import pandas as pd
    from rich.status import Status

    from ado.cli.models.parameters import AdoGetCommandParameters
    from ado.metastore.project import ProjectContext
    from ado.metastore.sqlstore import SQLStore


def format_default_ado_get_single_resource(
    resource: ADOResource, show_details: bool
) -> "pd.DataFrame":
    import json

    import pandas as pd

    columns = (
        ["IDENTIFIER", "NAME", "AGE"]
        if not show_details
        else ["IDENTIFIER", "NAME", "DESCRIPTION", "LABELS", "AGE"]
    )

    if isinstance(resource, OperationResource):
        # Insert before AGE to produce SPACE, STATUS, EXIT_STATE, AGE:
        columns.insert(-1, "SPACE")
        columns.insert(-1, "STATUS")
        columns.insert(-1, "EXIT_STATE")

    if not resource:
        return pd.DataFrame(columns=columns)

    metadata = resource.config.metadata or ConfigurationMetadata()
    output = {
        "IDENTIFIER": resource.identifier,
        "NAME": metadata.name or "",
        "AGE": timedelta_to_string(
            time_since_timestamp(resource.created).total_seconds()
        ),
    }

    if show_details:
        output["DESCRIPTION"] = metadata.description or ""
        output["LABELS"] = json.dumps(metadata.labels) if metadata.labels else None

    if isinstance(resource, OperationResource):
        status_update = most_important_status_update(resource.status)
        output["STATUS"] = status_update.event.value
        output["EXIT_STATE"] = (
            status_update.exit_state.value
            if isinstance(status_update.event, OperationResourceEventEnum)
            and status_update.exit_state is not None
            else "N/A"
        )
        output["SPACE"] = resource.config.spaces[0] if resource.config.spaces else ""

    # AP: if we don't set the index manually, pandas will complain with
    #   ValueError: If using all scalar values, you must pass an index
    # We also use the columns array to reorder the columns
    return pd.DataFrame(output, index=[0])[columns]


def format_default_ado_get_multiple_resources(
    resources: "pd.DataFrame", resource_kind: CoreResourceKinds
) -> "pd.DataFrame":
    if resources.empty:
        return resources

    # AP 13-12-2024:
    # Currently only Operations support status updates.
    # We try to keep it flexible.
    status_model = pydantic.RootModel[list[ADOResourceStatus]]
    if resource_kind == CoreResourceKinds.OPERATION:
        status_model = pydantic.RootModel[list[OperationResourceStatus]]

    columns = list(resources.columns)
    if resource_kind == CoreResourceKinds.OPERATION:
        resources["STATUS"] = resources["STATUS"].apply(
            lambda x: most_important_status_update(
                status_model.model_validate(json.loads(x)).root if x else None
            )
        )
        resources["EXIT_STATE"] = resources["STATUS"].apply(
            lambda x: (
                x.exit_state.value
                if isinstance(x.event, OperationResourceEventEnum)
                and x.exit_state is not None
                else "N/A"
            )
        )
        resources["STATUS"] = resources["STATUS"].apply(lambda x: x.event.value)
        resources = reorder_dataframe_columns(
            df=resources,
            move_to_start=[],
            move_to_end=["SPACE", "STATUS", "EXIT_STATE", "AGE"],
        )
        columns = list(resources.columns)

    # Avoid printing null or None in the NAME column
    resources["NAME"] = resources["NAME"].fillna("")

    if "DESCRIPTION" in resources.columns:
        resources["DESCRIPTION"] = resources["DESCRIPTION"].fillna("")

    # AP: the default formatting of timedelta objects is too verbose
    # we convert it to
    if "AGE" in resources.columns:
        resources["AGE"] = resources["AGE"].apply(
            lambda x: timedelta_to_string(x.total_seconds())
        )

    return resources[columns]


def build_resource_listing_dataframe(
    resources: dict[str, "ADOResource"],
    resource_kind: "CoreResourceKinds",
    sort_by_age_descending: bool = False,
    show_details: bool = False,
) -> "pd.DataFrame":
    """Build the same DataFrame shape as ``getResourceIdentifiersOfKind`` from a resources dict.

    Used when specific identifiers are requested so that only those resources are
    fetched (via ``getResources``) instead of loading every resource of that kind
    and then filtering in-memory.

    Args:
        resources: Mapping of identifier → ``ADOResource`` as returned by
            ``sql_store.getResources``.
        resource_kind: The kind of the resources (determines extra columns).
        sort_by_age_descending: When ``True``, sort the resulting DataFrame by
            ``AGE`` in descending order and reset the index.
        show_details: When ``True``, include ``DESCRIPTION`` and ``LABELS``
            columns in the returned DataFrame.

    Returns:
        A ``pd.DataFrame`` with columns ``IDENTIFIER``, ``NAME``, ``AGE``
        (as ``datetime.timedelta``) and, for operations, additionally
        ``STATUS`` (JSON string) and ``SPACE``. When ``show_details`` is
        ``True`` the columns also include ``DESCRIPTION`` and ``LABELS``.
        The DataFrame is compatible with
        :func:`format_default_ado_get_multiple_resources`.
    """
    import pandas as pd

    now = datetime.datetime.now(datetime.timezone.utc)

    def resource_to_row(
        identifier: str, resource: "ADOResource"
    ) -> dict[str, typing.Any]:
        metadata = resource.config.metadata or ConfigurationMetadata()
        row = {
            "IDENTIFIER": identifier,
            "NAME": metadata.name,
            "AGE": now - resource.created,
        }

        if show_details:
            row["DESCRIPTION"] = metadata.description
            row["LABELS"] = json.dumps(metadata.labels) if metadata.labels else None

        if resource_kind == CoreResourceKinds.OPERATION:
            row["STATUS"] = pydantic.RootModel[list[ADOResourceStatus]](
                resource.status
            ).model_dump_json()
            row["SPACE"] = resource.config.spaces[0] if resource.config.spaces else None

        return row

    columns = ["IDENTIFIER", "NAME", "AGE"]
    if show_details:
        columns = ["IDENTIFIER", "NAME", "DESCRIPTION", "LABELS", "AGE"]
    if resource_kind == CoreResourceKinds.OPERATION:
        columns.extend(["STATUS", "SPACE"])

    df = pd.DataFrame(
        [
            resource_to_row(identifier, resource)
            for identifier, resource in resources.items()
        ],
        columns=columns,
    )

    if sort_by_age_descending:
        df = df.sort_values(by="AGE", ascending=False).reset_index(drop=True)

    return df


def format_ado_get_stats_for_operations(
    df: "pd.DataFrame",
    sql_store: "SQLStore",
    spinner: "Status | None" = None,
    include_request_columns: bool = False,
) -> "pd.DataFrame":
    """Append measurement-statistics columns to an operations DataFrame.

    Issues two queries regardless of the number of operations:
    one recursive-CTE query to resolve operation→samplestore relationships,
    then one aggregation query per distinct samplestore (grouped by
    operation_id) to fetch all stats in bulk.

    Args:
        df: DataFrame with at least an ``IDENTIFIER`` column (one row per
            operation).
        sql_store: The ``SQLStore`` to use for relationship and samplestore
            queries.
        spinner: Optional rich status spinner to update with progress messages.
        include_request_columns: When ``True``, also append
            ``TOTAL_REQUESTS``, ``FAILED_REQUESTS``, and
            ``SUCCESSFUL_REQUESTS`` columns in addition to the default
            result-level columns. Defaults to ``False``.

    Returns:
        The same DataFrame with extra columns appended.
        Always appended: ``TOTAL_RESULTS``, ``SUCCESSFUL_RESULTS``,
        ``FAILED_RESULTS``, ``MEASURED_ENTITIES``.
        Appended when *include_request_columns* is ``True``:
        ``TOTAL_REQUESTS``, ``FAILED_REQUESTS``, ``SUCCESSFUL_REQUESTS``.
        Operations with no recorded measurements show ``0`` in all stats
        columns.
    """

    from ado.core.resources import CoreResourceKinds
    from ado.core.samplestore.base import SampleStore

    _RESULT_COLUMNS = [
        "TOTAL_RESULTS",
        "SUCCESSFUL_RESULTS",
        "FAILED_RESULTS",
        "MEASURED_ENTITIES",
    ]
    _REQUEST_COLUMNS = [
        "TOTAL_REQUESTS",
        "FAILED_REQUESTS",
        "SUCCESSFUL_REQUESTS",
    ]

    operation_ids: set[str] = set(df["IDENTIFIER"])

    # Round-trip 2: one recursive-CTE query for all operations at once
    # Returns {operation_id: {CoreResourceKinds.SAMPLESTORE: {samplestore_id, ...}, ...}}
    relationships: dict[str, dict[CoreResourceKinds, set[str]]] = (
        sql_store.get_resources_by_relationship(
            kind=CoreResourceKinds.OPERATION,
            identifier=operation_ids,
            hierarchy_direction="up",
            max_hops=2,
            identifiers_only=True,
        )
    )

    # Invert to {samplestore_id: set_of_operation_ids}.
    samplestore_to_operation_ids: dict[str, set[str]] = {}
    for operation_id, kind_map in relationships.items():
        for samplestore_id in kind_map.get(CoreResourceKinds.SAMPLESTORE, set()):
            samplestore_to_operation_ids.setdefault(samplestore_id, set()).add(
                operation_id
            )

    # Round-trip 3: fetch all distinct samplestore resources in one batch query.
    samplestore_id_to_resource: dict[str, ADOResource] = sql_store.getResources(
        list(samplestore_to_operation_ids.keys())
    )

    # Round-trips 4…K+3: one batched stats query per unique samplestore.
    # SampleStore is instantiated from the already-fetched resource — no extra DB round-trip.
    stats_lookup: dict[str, dict[str, int]] = {}
    total_samplestores = len(samplestore_to_operation_ids)
    for index, (
        samplestore_id,
        operation_ids_in_store,
    ) in enumerate(samplestore_to_operation_ids.items(), start=1):
        if spinner is not None:
            spinner.update(
                f"Calculating stats for operations in samplestore {index}/{total_samplestores}: {samplestore_id}"
            )
        sample_store = SampleStore.from_resource(
            samplestore_id_to_resource[samplestore_id]
        )
        for stat in sample_store.operation_measurement_statistics(
            operation_ids=operation_ids_in_store
        ):
            stats_lookup[stat.operation_id] = {
                "TOTAL_RESULTS": stat.total_results,
                "SUCCESSFUL_RESULTS": stat.successful_results,
                "FAILED_RESULTS": stat.failed_results,
                "MEASURED_ENTITIES": stat.measured_entities,
                "TOTAL_REQUESTS": stat.total_requests,
                "FAILED_REQUESTS": stat.failed_requests,
                "SUCCESSFUL_REQUESTS": stat.successful_requests,
            }

    # Attach result-level stats columns; operations with no samplestore mapping show 0.
    all_columns = _RESULT_COLUMNS + (
        _REQUEST_COLUMNS if include_request_columns else []
    )
    zeros = dict.fromkeys(_RESULT_COLUMNS + _REQUEST_COLUMNS, 0)
    for col in all_columns:
        df[col] = df["IDENTIFIER"].apply(
            lambda operation_id, c=col: stats_lookup.get(operation_id, zeros)[c]
        )

    return df


def format_ado_get_stats_for_spaces(
    df: "pd.DataFrame",
    sql_store: "SQLStore",
    spinner: "Status | None" = None,
    include_heavy: bool = False,
    space_resources: "dict[str, ADOResource] | None" = None,
    project_context: "ProjectContext | None" = None,
) -> "pd.DataFrame":
    """Append statistics columns to a discovery spaces DataFrame.

    Issues a single batched query for all spaces' child operations, then a
    single batched query for their parent sample stores.

    When ``include_heavy=False`` (default), also issues a metastore stats
    query and a ``space_entity_statistics`` query per distinct sample store,
    then appends four lightweight columns.

    When ``include_heavy=True``, builds full
    :class:`~ado.core.discoveryspace.space.DiscoverySpace` instances
    per samplestore group and delegates to
    :func:`~ado.core.discoveryspace.stats.space_statistics_for_spaces`
    (which issues both the metastore and sample-store queries internally),
    then appends all lightweight **and** heavy columns in one pass.

    Args:
        df: DataFrame with at least an ``IDENTIFIER`` column (one row per
            discovery space).
        sql_store: The ``SQLStore`` to use for relationship and stats queries.
        spinner: Optional rich status spinner to update with progress messages.
        include_heavy: When ``True`` also compute and append the heavy stats
            columns (``SIZE_OF_ENTITY_SPACE``, ``UNMEASURED_ENTITIES``,
            ``MATCHING_ENTITIES``, ``MATCHING_WITH_MEASUREMENTS``,
            ``ENTITIES_WITH_ALL_MEASUREMENTS``,
            ``ENTITIES_WITH_PARTIAL_MEASUREMENTS``,
            ``MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS``).
            Requires ``space_resources`` and ``project_context``.
        space_resources: Mapping of space identifier → hydrated
            :class:`~ado.core.resources.ADOResource`.  Required when
            ``include_heavy=True``; ignored otherwise.
        project_context: Project context used to instantiate
            :class:`~ado.core.discoveryspace.space.DiscoverySpace`.
            Required when ``include_heavy=True``; ignored otherwise.

    Returns:
        The same DataFrame with extra columns appended.
        Lightweight columns: ``EXPERIMENTS``, ``OPERATIONS``,
        ``EXPLORE_OPERATIONS``, ``MEASURED_ENTITIES``.
        Heavy columns (only when ``include_heavy=True``):
        ``SIZE_OF_ENTITY_SPACE``, ``UNMEASURED_ENTITIES``,
        ``MATCHING_ENTITIES``, ``MATCHING_WITH_MEASUREMENTS``,
        ``ENTITIES_WITH_ALL_MEASUREMENTS``,
        ``ENTITIES_WITH_PARTIAL_MEASUREMENTS``,
        ``MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS``.
        Spaces with no recorded operations or sample store show ``0`` in all
        lightweight stats columns; heavy columns show ``None`` for spaces
        where the value cannot be determined.
    """
    from ado.core.resources import CoreResourceKinds
    from ado.core.samplestore.base import SampleStore

    space_ids: set[str] = set(df["IDENTIFIER"])

    # Query 1: for each space, get its child operations.
    # Returns {space_id: {CoreResourceKinds.OPERATION: set[operation_id]}}
    space_child_relationships: dict[str, dict[CoreResourceKinds, set[str]]] = (
        sql_store.get_resources_by_relationship(
            kind=CoreResourceKinds.DISCOVERYSPACE,
            identifier=space_ids,
            hierarchy_direction="down",
            max_hops=1,
            identifiers_only=True,
        )
    )

    # Build space_id_to_operation_ids: {space_id: set[operation_id]}
    space_id_to_operation_ids: dict[str, set[str]] = {}
    for space_id in space_ids:
        space_children_by_kind = space_child_relationships.get(space_id, {})
        space_id_to_operation_ids[space_id] = space_children_by_kind.get(
            CoreResourceKinds.OPERATION, set()
        )

    # Query 2: for each space, get its parent sample store(s),
    # returning hydrated SampleStoreResource objects.
    # Returns {space_id: {CoreResourceKinds.SAMPLESTORE: {samplestore_id: SampleStoreResource}}}
    space_parent_relationships: dict[
        str, dict[CoreResourceKinds, dict[str, ADOResource]]
    ] = sql_store.get_resources_by_relationship(
        kind=CoreResourceKinds.DISCOVERYSPACE,
        identifier=space_ids,
        hierarchy_direction="up",
        max_hops=1,
        identifiers_only=False,
    )

    # Invert to {samplestore_id: set[space_id]}, keeping the hydrated resource.
    samplestore_id_to_resource: dict[str, ADOResource] = {}
    samplestore_id_to_space_ids: dict[str, set[str]] = {}
    for space_id, space_parents_by_kind in space_parent_relationships.items():
        samplestore_resources = space_parents_by_kind.get(
            CoreResourceKinds.SAMPLESTORE, {}
        )
        for samplestore_id, samplestore_resource in samplestore_resources.items():
            samplestore_id_to_resource[samplestore_id] = samplestore_resource
            samplestore_id_to_space_ids.setdefault(samplestore_id, set()).add(space_id)

    from ado.core.discoveryspace.stats import (
        DiscoverySpaceStatistics,
        space_statistics_for_spaces,
    )

    all_stats: dict[str, DiscoverySpaceStatistics] = {}
    total_samplestores = len(samplestore_id_to_space_ids)

    if include_heavy:
        # Heavy path: build all DiscoverySpace instances first (phase 1), then
        # compute statistics in a single pass (phase 2).  Keeping the two phases
        # separate means the spinner never alternates between "Initialising" and
        # "Computing statistics" messages.
        from ado.core.discoveryspace.space import DiscoverySpace

        # Phase 1: initialise all spaces.
        total_spaces = len(space_ids)
        all_spaces: list[DiscoverySpace] = []
        for (
            samplestore_id,
            samplestore_space_ids,
        ) in samplestore_id_to_space_ids.items():
            sample_store = SampleStore.from_resource(
                samplestore_id_to_resource[samplestore_id]
            )
            for space_id in samplestore_space_ids:
                if spinner is not None:
                    spinner.update(
                        f"Initialising space {space_id} ({len(all_spaces) + 1}/{total_spaces})"
                    )
                all_spaces.append(
                    DiscoverySpace.from_configuration(
                        conf=space_resources[space_id].config,
                        project_context=project_context,  # type: ignore[arg-type]
                        identifier=space_id,
                        metadata_store=sql_store,
                        sample_store=sample_store,
                        load_experiment_catalog=False,
                    )
                )

        # Phase 2: compute statistics for all spaces at once.
        all_stats.update(
            space_statistics_for_spaces(
                all_spaces, lightweight_only=False, spinner=spinner
            )
        )
    else:
        # Lightweight path: issue a metastore stats query and one
        # space_entity_statistics query per distinct sample store.
        metastore_stats = sql_store.get_space_metastore_stats(space_ids)

        for index, (samplestore_id, samplestore_space_ids) in enumerate(
            samplestore_id_to_space_ids.items(), start=1
        ):
            if spinner is not None:
                spinner.update(
                    f"Calculating stats for spaces in samplestore {index}/{total_samplestores}: {samplestore_id}"
                )
            sample_store = SampleStore.from_resource(
                samplestore_id_to_resource[samplestore_id]
            )
            operation_ids_per_space = {
                space_id: space_id_to_operation_ids[space_id]
                for space_id in samplestore_space_ids
            }
            entity_stats_per_space = sample_store.space_entity_statistics(
                space_ids_to_operation_ids=operation_ids_per_space
            )
            for space_id, entity_stats in entity_stats_per_space.items():
                all_stats[space_id] = DiscoverySpaceStatistics(
                    number_of_experiments=metastore_stats[
                        space_id
                    ].number_of_experiments,
                    number_of_operations=metastore_stats[space_id].number_of_operations,
                    number_of_explore_operations=metastore_stats[
                        space_id
                    ].number_of_explore_operations,
                    number_measured_entities=entity_stats.number_measured_entities or 0,
                )

    # Attach lightweight columns; spaces with no data show 0.
    df["EXPERIMENTS"] = df["IDENTIFIER"].apply(
        lambda space_id: (
            all_stats[space_id].number_of_experiments if space_id in all_stats else 0
        )
    )
    df["OPERATIONS"] = df["IDENTIFIER"].apply(
        lambda space_id: (
            all_stats[space_id].number_of_operations if space_id in all_stats else 0
        )
    )
    df["EXPLORE_OPERATIONS"] = df["IDENTIFIER"].apply(
        lambda space_id: (
            all_stats[space_id].number_of_explore_operations
            if space_id in all_stats
            else 0
        )
    )
    df["MEASURED_ENTITIES"] = df["IDENTIFIER"].apply(
        lambda space_id: (
            all_stats[space_id].number_measured_entities if space_id in all_stats else 0
        )
    )

    # Attach heavy columns only when requested.
    if include_heavy:
        _heavy_field_map = {
            "SIZE_OF_ENTITY_SPACE": "size_of_entity_space",
            "UNMEASURED_ENTITIES": "number_unmeasured_entities",
            "MATCHING_ENTITIES": "number_matching_entities",
            "MATCHING_WITH_MEASUREMENTS": "number_matching_entities_with_measurements",
            "ENTITIES_WITH_ALL_MEASUREMENTS": "entities_with_all_measurements",
            "ENTITIES_WITH_PARTIAL_MEASUREMENTS": "entities_with_partial_measurements",
            "MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS": "matching_entities_with_all_measurements",
        }
        for col, field in _heavy_field_map.items():
            df[col] = df["IDENTIFIER"].apply(
                lambda space_id, field_name=field: (
                    getattr(all_stats[space_id], field_name, None)
                    if space_id in all_stats
                    else None
                )
            )

        # Coerce SIZE_OF_ENTITY_SPACE and UNMEASURED_ENTITIES to int where the
        # value is finite (i.e. not inf/nan/None).  Pandas stores mixed
        # int/float/None columns as float64, which renders integers as "45.0".
        def _coerce_to_int_if_finite(v: object) -> object:
            if isinstance(v, float) and math.isfinite(v):
                return int(v)
            return v

        for col in ("SIZE_OF_ENTITY_SPACE", "UNMEASURED_ENTITIES"):
            if col in df.columns:
                df[col] = df[col].apply(_coerce_to_int_if_finite)

    return df


def format_ado_get_stats_for_samplestores(
    df: "pd.DataFrame",
    sql_store: "SQLStore",
    spinner: "Status | None" = None,
) -> "pd.DataFrame":
    """Append 3 statistics columns to a sample stores DataFrame.

    Iterates over each store, updating the spinner with progress, and collects
    entity, result, and experiment counts.

    Args:
        df: DataFrame with at least an ``IDENTIFIER`` column (one row per
            sample store).
        sql_store: The ``SQLStore`` to use for loading each sample store.
        spinner: Optional rich status spinner to update with progress messages.

    Returns:
        The same DataFrame with three extra columns appended:
        ``ENTITIES``, ``RESULTS``, ``EXPERIMENTS``.
        Stores with no recorded data show ``0`` in all stats columns.
    """
    from ado.core.samplestore.base import SampleStore

    samplestore_ids: list[str] = list(df["IDENTIFIER"])
    total_samplestores = len(samplestore_ids)

    # Fetch all samplestore resources in a single batch query, then instantiate
    # SampleStore objects from the hydrated resources — no per-store round-trip.
    resources: dict[str, ADOResource] = sql_store.getResources(samplestore_ids)

    stats_by_id = {}
    for index, samplestore_id in enumerate(samplestore_ids, start=1):
        if samplestore_id not in resources:
            continue
        if spinner is not None:
            spinner.update(
                f"Calculating stats for samplestore {index}/{total_samplestores}: {samplestore_id}"
            )
        store = SampleStore.from_resource(resources[samplestore_id])
        stats_by_id[samplestore_id] = store.samplestore_statistics()

    # Attach stats columns; stores with no data show 0.
    df["ENTITIES"] = df["IDENTIFIER"].apply(
        lambda sid: stats_by_id[sid].number_of_entities if sid in stats_by_id else 0
    )
    df["RESULTS"] = df["IDENTIFIER"].apply(
        lambda sid: stats_by_id[sid].number_of_results if sid in stats_by_id else 0
    )
    df["EXPERIMENTS"] = df["IDENTIFIER"].apply(
        lambda sid: stats_by_id[sid].number_of_experiments if sid in stats_by_id else 0
    )

    return df


def _format_bytes(n: int) -> str:
    """Return a compact human-readable representation of a byte count.

    Uses 1024-based binary units (B, KiB, MiB, GiB, TiB).  Values below
    1 KiB are shown as ``"<n> B"``.  Larger values are shown with one
    decimal place and the appropriate unit suffix, e.g. ``"4.9 KiB"`` or
    ``"1.2 MiB"``.

    Args:
        n: Number of bytes (non-negative integer).

    Returns:
        A short human-readable string such as ``"512 B"`` or ``"3.7 MiB"``.
    """
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TiB"


def format_ado_get_stats_for_datacontainers(
    df: "pd.DataFrame",
    sql_store: "SQLStore",
    spinner: "Status | None" = None,
) -> "pd.DataFrame":
    """Append 4 statistics columns to a data containers DataFrame.

    Issues one batched stats query for all containers at once.

    Args:
        df: DataFrame with at least an ``IDENTIFIER`` column (one row per
            data container).
        sql_store: The ``SQLStore`` to use for stats queries.
        spinner: Optional rich status spinner to update with progress messages.

    Returns:
        The same DataFrame with four extra columns appended:
        ``TABLES``, ``LOCATIONS``, ``KEY_VALUES``, ``DATA_BYTES``.
        ``DATA_BYTES`` is formatted as a human-readable size (e.g. ``"4.9 KB"``).
        Containers with no recorded data show ``0`` / ``"0 B"`` in all stats columns.
    """
    datacontainer_ids: set[str] = set(df["IDENTIFIER"])

    if spinner is not None:
        spinner.update("Calculating stats for data containers...")

    stats_by_id = sql_store.get_datacontainer_stats(datacontainer_ids)

    _int_columns = [
        ("TABLES", "number_of_tables"),
        ("LOCATIONS", "number_of_locations"),
        ("KEY_VALUES", "number_of_key_values"),
    ]
    for col, attr in _int_columns:
        df[col] = df["IDENTIFIER"].apply(
            lambda cid, a=attr: (
                getattr(stats_by_id[cid], a, 0) if cid in stats_by_id else 0
            )
        )

    df["DATA_BYTES"] = df["IDENTIFIER"].apply(
        lambda cid: _format_bytes(
            stats_by_id[cid].total_data_bytes if cid in stats_by_id else 0
        )
    )

    return df


def format_resource_for_ado_get_custom_format(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:
    match parameters.output_format:
        case AdoGetSupportedOutputFormats.CONFIG:
            return _config_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case AdoGetSupportedOutputFormats.YAML:
            return _yaml_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case AdoGetSupportedOutputFormats.JSON:
            return _json_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case AdoGetSupportedOutputFormats.RAW:
            return _raw_formatter_for_ado_resource(
                to_print=to_print, parameters=parameters
            )
        case _:
            raise ValueError(
                f"Output format {parameters.output_format.value} is not supported."
            )


def _config_formatter_for_ado_resource(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:

    if isinstance(to_print, list):
        console_print(f"{ERROR}{ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE}", stderr=True)
        raise typer.Exit(1)

    if not hasattr(to_print, "config"):
        console_print(
            f"{ERROR}The resource requested does not have a config field.", stderr=True
        )
        raise typer.Exit(1)

    if parameters.minimize_output:
        serialization_context = minimize_output_context
        serialization_target = _minimize_ado_resource_representation(to_print).config
    else:
        serialization_context = None
        serialization_target = to_print.config

    # To handle lists correctly, we need to select all items of the list
    if parameters.exclude_fields and not parameters.resource_id:
        parameters.exclude_fields = [
            f"[*].{field_exclusion}" for field_exclusion in parameters.exclude_fields
        ]

    # AP: 28/07/2025:
    # We can't simply use model_dump because we would end up with errors like:
    #    RepresenterError: ('cannot represent an object', <ADOResourceEventEnum.CREATED: 'created'>)
    # when calling yaml.safe_dump
    dict_representation = yaml.safe_load(
        printable_pydantic_model(serialization_target).model_dump_json(
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        )
    )

    if parameters.exclude_fields:
        dict_representation = remove_fields_from_dictionary(
            dict_representation, parameters.exclude_fields
        )

    return yaml.safe_dump(dict_representation)


def _yaml_formatter_for_ado_resource(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:

    if parameters.minimize_output:
        serialization_context = minimize_output_context
        serialization_target = _minimize_ado_resource_representation(to_print)
    else:
        serialization_context = None
        serialization_target = to_print

    # To handle lists correctly, we need to select all items of the list
    if parameters.exclude_fields and not parameters.resource_id:
        parameters.exclude_fields = [
            f"[*].{field_exclusion}" for field_exclusion in parameters.exclude_fields
        ]

    # AP: 28/07/2025:
    # We can't simply use model_dump because we would end up with errors like:
    #    RepresenterError: ('cannot represent an object', <ADOResourceEventEnum.CREATED: 'created'>)
    # when calling yaml.safe_dump
    dict_representation = yaml.safe_load(
        printable_pydantic_model(serialization_target).model_dump_json(
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        )
    )

    if parameters.exclude_fields:
        dict_representation = remove_fields_from_dictionary(
            dict_representation, parameters.exclude_fields
        )

    return yaml.safe_dump(dict_representation)


def _json_formatter_for_ado_resource(
    to_print: (
        ADOResource | list[ADOResource] | pydantic.BaseModel | list[pydantic.BaseModel]
    ),
    parameters: "AdoGetCommandParameters",
) -> str:

    if parameters.minimize_output:
        serialization_context = minimize_output_context
        serialization_target = _minimize_ado_resource_representation(to_print)
    else:
        serialization_context = None
        serialization_target = to_print

    # When exclude_fields is False, we know our data is valid.
    # We don't need to do any processing other than
    # using printable_pydantic_model to handle lists.
    if not parameters.exclude_fields:
        return printable_pydantic_model(serialization_target).model_dump_json(
            indent=2,
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        )

    # Here we need to remove some fields and this might
    # mean creating a model that's invalid.

    # To handle lists correctly, we need to select all items of the list
    if not parameters.resource_id:
        parameters.exclude_fields = [
            f"[*].{field_exclusion}" for field_exclusion in parameters.exclude_fields
        ]

    printable_model = printable_pydantic_model(serialization_target)
    filtered_representation = remove_fields_from_dictionary(
        input_dictionary=printable_model.model_dump(
            exclude_none=parameters.exclude_none,
            exclude_unset=parameters.exclude_unset,
            exclude_defaults=parameters.exclude_default,
            context=serialization_context,
        ),
        fields_to_remove=parameters.exclude_fields,
    )

    # AP: 28/07/2025:
    # pydantic's json serializer can handle more data types
    # so we need to construct a model on-the-fly.
    # We can't use `model_validate` as we might have removed
    # a required field.
    if isinstance(filtered_representation, list):
        model = printable_model.model_construct(filtered_representation)
    else:
        model = printable_model.model_construct(**filtered_representation)

    # AP: 30/09/2025
    # We set warnings="none" as otherwise we'd print a ton of:
    # PydanticSerializationUnexpectedValue(Expected `SOME_MODEL` -
    # serialized value may not be as expected [input_value={...}, input_type=dict])
    return model.model_dump_json(
        indent=2,
        warnings="none",
        exclude_none=parameters.exclude_none,
        exclude_unset=parameters.exclude_unset,
        exclude_defaults=parameters.exclude_default,
        context=serialization_context,
    )


def _raw_formatter_for_ado_resource(
    to_print: (
        ADOResource
        | list[ADOResource]
        | pydantic.BaseModel
        | list[pydantic.BaseModel]
        | dict
    ),
    parameters: "AdoGetCommandParameters",
) -> str:
    import pprint

    if parameters.minimize_output:
        console_print(
            f"{WARN}Minimizing output is not supported for the raw output type.",
            stderr=True,
        )

    return pprint.pformat(to_print)


def _minimize_ado_resource_representation(
    to_print: (
        ADOResource | list[ADOResource] | pydantic.BaseModel | list[pydantic.BaseModel]
    ),
) -> ADOResource | pydantic.BaseModel:
    if isinstance(to_print, list):
        console_print(
            f"{ERROR}The minimal output format can only be used "
            f"when a resource identifier is provided.",
            stderr=True,
        )
        raise typer.Exit(1)

    if isinstance(to_print, DiscoverySpaceResource):
        to_print.config = to_print.config.convert_experiments_to_reference_list()
    if isinstance(to_print, DiscoverySpaceConfiguration):
        for property in to_print.entitySpace:
            if (
                property.propertyDomain.variableType
                == VariableTypeEnum.BINARY_VARIABLE_TYPE
            ):
                property.propertyDomain.probabilityFunction = None
                property.propertyDomain.domainRange = None
                property.propertyDomain.interval = None
                property.propertyDomain.values = None

    return to_print


def time_since_timestamp(ts: datetime.datetime) -> datetime.timedelta:
    # AP: there are still some datetimes that are not timezone aware
    return (
        datetime.datetime.now() - ts
        if not ts.tzinfo
        else datetime.datetime.now(tz=datetime.timezone.utc) - ts
    )


def most_important_status_update(
    statuses: list[OperationResourceStatus],
) -> OperationResourceStatus:

    if not statuses:
        return OperationResourceStatus(event=ADOResourceEventEnum.ADDED)

    status_updates = [s.event for s in statuses]
    for important_event in event_importance_order:
        if important_event in status_updates:
            idx = status_updates.index(important_event)
            return statuses[idx]

    return OperationResourceStatus(event=ADOResourceEventEnum.ADDED)


def timedelta_to_string(total_seconds: float) -> str:
    if math.isnan(total_seconds):
        return "NaT"
    if total_seconds < SECONDS_IN_A_MINUTE:
        return f"{round(total_seconds)}s"
    if total_seconds < SECONDS_IN_AN_HOUR:
        minutes, seconds = divmod(total_seconds, SECONDS_IN_A_MINUTE)
        return f"{int(minutes)}m{round(seconds)}s"
    if total_seconds < SECONDS_IN_A_DAY:
        hours, remainder = divmod(total_seconds, SECONDS_IN_AN_HOUR)
        minutes = remainder / SECONDS_IN_A_MINUTE
        return f"{int(hours)}h{round(minutes)}m"
    days, remainder = divmod(total_seconds, SECONDS_IN_A_DAY)
    hours = remainder / SECONDS_IN_AN_HOUR
    return f"{int(days)}d{round(hours)}h"
