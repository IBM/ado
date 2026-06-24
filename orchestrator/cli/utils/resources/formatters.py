# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import datetime
import json
import math
import typing

import pydantic
import typer
import yaml

from orchestrator.cli.models.types import AdoGetSupportedOutputFormats
from orchestrator.cli.utils.generic.constants import (
    SECONDS_IN_A_DAY,
    SECONDS_IN_A_MINUTE,
    SECONDS_IN_AN_HOUR,
)
from orchestrator.cli.utils.jsonpath.filters import remove_fields_from_dictionary
from orchestrator.cli.utils.output.prints import (
    ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE,
    ERROR,
    WARN,
    console_print,
)
from orchestrator.cli.utils.pydantic.constants import (
    event_importance_order,
    minimize_output_context,
)
from orchestrator.core import (
    ADOResource,
    CoreResourceKinds,
    DiscoverySpaceResource,
    OperationResource,
)
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.operation.resource import (
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.core.resources import ADOResourceEventEnum, ADOResourceStatus
from orchestrator.schema.domain import VariableTypeEnum
from orchestrator.utilities.output import (
    printable_pydantic_model,
)
from orchestrator.utilities.pandas import reorder_dataframe_columns

if typing.TYPE_CHECKING:
    import pandas as pd
    from rich.status import Status

    from orchestrator.cli.models.parameters import AdoGetCommandParameters
    from orchestrator.metastore.sqlstore import SQLStore


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


def format_ado_get_stats_for_operations(
    df: "pd.DataFrame",
    sql_store: "SQLStore",
    spinner: "Status | None" = None,
) -> "pd.DataFrame":
    """Append 4 measurement-statistics columns to an operations DataFrame.

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

    Returns:
        The same DataFrame with four extra columns appended:
        ``TOTAL_RESULTS``, ``SUCCESSFUL_RESULTS``, ``FAILED_RESULTS``,
        ``MEASURED_ENTITIES``. Operations with no recorded measurements show
        ``0`` in all stats columns.
    """

    from orchestrator.core.resources import CoreResourceKinds
    from orchestrator.core.samplestore.base import SampleStore

    _STATS_COLUMNS = [
        "TOTAL_RESULTS",
        "SUCCESSFUL_RESULTS",
        "FAILED_RESULTS",
        "MEASURED_ENTITIES",
    ]

    operation_ids: set[str] = set(df["IDENTIFIER"])

    # Round-trip 2: one recursive-CTE query for all operations at once.
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

    # Invert to {samplestore_id: set_of_operation_ids}
    samplestore_to_operation_ids: dict[str, set[str]] = {}
    for operation_id, kind_map in relationships.items():
        samplestore_ids = kind_map.get(CoreResourceKinds.SAMPLESTORE, set())
        for samplestore_id in samplestore_ids:
            samplestore_to_operation_ids.setdefault(samplestore_id, set()).add(
                operation_id
            )

    # Round-trips 3…K+2: one load + one batched stats query per unique samplestore.
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
        sample_store = SampleStore.from_identifier(samplestore_id, sql_store)
        for stat in sample_store.operation_measurement_statistics(
            operation_ids=operation_ids_in_store
        ):
            stats_lookup[stat.operation_id] = {
                "TOTAL_RESULTS": stat.total_results,
                "SUCCESSFUL_RESULTS": stat.successful_results,
                "FAILED_RESULTS": stat.failed_results,
                "MEASURED_ENTITIES": stat.measured_entities,
            }

    # Attach stats columns; operations with no samplestore mapping show 0.
    zeros = dict.fromkeys(_STATS_COLUMNS, 0)
    for col in _STATS_COLUMNS:
        df[col] = df["IDENTIFIER"].apply(
            lambda operation_id, c=col: stats_lookup.get(operation_id, zeros)[c]
        )

    return df


def format_ado_get_stats_for_spaces(
    df: "pd.DataFrame",
    sql_store: "SQLStore",
    spinner: "Status | None" = None,
) -> "pd.DataFrame":
    """Append 4 space-statistics columns to a discovery spaces DataFrame.

    Issues a metastore stats query for all spaces at once, then resolves each
    space's linked operations and sample stores.  For each distinct sample
    store a single batched :meth:`space_entity_statistics` query is issued.

    Args:
        df: DataFrame with at least an ``IDENTIFIER`` column (one row per
            discovery space).
        sql_store: The ``SQLStore`` to use for relationship and stats queries.
        spinner: Optional rich status spinner to update with progress messages.

    Returns:
        The same DataFrame with four extra columns appended:
        ``EXPERIMENTS``, ``OPERATIONS``, ``EXPLORE_OPERATIONS``,
        ``MEASURED_ENTITIES``.
        Spaces with no recorded operations or sample store show ``0`` in all
        stats columns.
    """
    from orchestrator.core.resources import CoreResourceKinds
    from orchestrator.core.samplestore.base import SampleStore

    _STATS_COLUMNS = [
        "EXPERIMENTS",
        "OPERATIONS",
        "EXPLORE_OPERATIONS",
        "MEASURED_ENTITIES",
    ]

    space_ids: set[str] = set(df["IDENTIFIER"])

    # Query 1: metastore stats (number_of_experiments, number_of_operations,
    # number_of_explore_operations) for all spaces in one SQL query.
    metastore_stats = sql_store.get_space_metastore_stats(space_ids)

    # Query 2: for each space, get its child operations.
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

    # Query 3: for each space, get its parent sample store(s),
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

    # For each distinct sample store, issue one batched space_entity_statistics query.
    space_id_to_measured_entity_count: dict[str, int] = {}
    total_samplestores = len(samplestore_id_to_space_ids)
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
            space_id_to_measured_entity_count[space_id] = (
                entity_stats.number_measured_entities or 0
            )

    # Attach stats columns; spaces with no data show 0.
    # get_space_metastore_stats always returns an entry for every requested
    # space_id, so direct attribute access is safe.
    df["EXPERIMENTS"] = df["IDENTIFIER"].apply(
        lambda space_id: metastore_stats[space_id].number_of_experiments
    )
    df["OPERATIONS"] = df["IDENTIFIER"].apply(
        lambda space_id: metastore_stats[space_id].number_of_operations
    )
    df["EXPLORE_OPERATIONS"] = df["IDENTIFIER"].apply(
        lambda space_id: metastore_stats[space_id].number_of_explore_operations
    )
    df["MEASURED_ENTITIES"] = df["IDENTIFIER"].apply(
        lambda space_id: space_id_to_measured_entity_count.get(space_id, 0)
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
