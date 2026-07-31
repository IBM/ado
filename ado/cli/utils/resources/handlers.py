# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import logging
import pathlib
import typing

import pydantic
import rich.rule
import typer
import yaml
from rich.console import RenderableType
from rich.status import Status

from ado.cli.models.types import (
    AdoEditSupportedEditors,
    AdoGetSupportedOutputFormats,
    AdoShowTraceSupportedOutputFormats,
)
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.output.dataframes import df_to_output
from ado.cli.utils.output.prints import (
    ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE,
    ADO_INFO_EMPTY_DATAFRAME,
    ADO_SPINNER_GETTING_OUTPUT_READY,
    ADO_SPINNER_QUERYING_DB,
    ADO_SPINNER_SAVING_TO_DB,
    ERROR,
    SUCCESS,
    console_print,
    cyan,
)
from ado.cli.utils.resources.formatters import (
    format_ado_get_stats_for_datacontainers,
    format_ado_get_stats_for_operations,
    format_ado_get_stats_for_samplestores,
    format_ado_get_stats_for_spaces,
    format_default_ado_get_multiple_resources,
    format_default_ado_get_single_resource,
    format_resource_for_ado_get_custom_format,
)
from ado.core.metadata import ConfigurationMetadata
from ado.metastore.base import ResourceDoesNotExistError
from ado.utilities.output import pydantic_model_as_yaml
from ado.utilities.rich import dataframe_to_rich_table

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import pandas as pd

    from ado.cli.models.parameters import (
        AdoGetCommandParameters,
        AdoShowTraceCommandParameters,
        AdoUpgradeCommandParameters,
    )
    from ado.core import ADOResource, CoreResourceKinds
    from ado.metastore.project import ProjectContext
    from ado.metastore.sqlstore import SQLStore


def resolve_related_to_identifiers(
    source_kind: "CoreResourceKinds",
    source_id: str,
    requested_kind: "CoreResourceKinds",
    sql_store: "SQLStore",
) -> set[str]:
    """Return the set of *requested_kind* resource identifiers related to *source_id*.

    Traverses the full resource relationship graph in both directions, so multi-hop
    relationships (e.g. operations related to a store via a space) are included.

    Args:
        source_kind: The kind of the source resource.
        source_id: The identifier of the source resource.
        requested_kind: The kind of resources to return.
        sql_store: The SQL store to query.

    Returns:
        A (possibly empty) set of identifiers of *requested_kind* resources
        related to *source_id*.

    Raises:
        ResourceDoesNotExistError: When the source resource does not exist.
    """
    if not sql_store.containsResourceWithIdentifier(
        identifier=source_id, kind=source_kind
    ):
        raise ResourceDoesNotExistError(resource_id=source_id, kind=source_kind)

    related = sql_store.get_resources_by_relationship(
        kind=source_kind,
        identifier=source_id,
        relationship="both",
        identifiers_only=True,
    )
    return related.get(requested_kind, set())


def _render_dataframe_table_output(
    df: "pd.DataFrame", parameters: "AdoGetCommandParameters"
) -> None:
    """Render a DataFrame as a rich table or print the empty-data message."""
    import rich.box

    if df.empty:
        console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
        return

    if parameters.output_file:
        do_not_truncate = True
    else:
        do_not_truncate = (
            ["IDENTIFIER"] if not parameters.no_trunc else parameters.no_trunc
        )

    table = dataframe_to_rich_table(
        df,
        box=rich.box.SQUARE,
        show_index=True,
        show_edge=True,
        do_not_truncate_columns=do_not_truncate,
    )
    _write_or_print_output(table, parameters.output_file)


def _build_table_output_dataframe(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds | None",
    dataframe: "pd.DataFrame | None",
    resources: "list[ADOResource] | ADOResource | None",
) -> "pd.DataFrame":
    """Build the DataFrame used by table-like get output formats."""
    import pandas as pd

    if dataframe is not None:
        return dataframe

    if resources is not None:
        if isinstance(resources, list):
            if not resources:
                return pd.DataFrame()
            return pd.concat(
                [
                    format_default_ado_get_single_resource(
                        resource=resource, show_details=parameters.show_details
                    )
                    for resource in resources
                ],
                ignore_index=True,
            )

        return format_default_ado_get_single_resource(
            resource=resources, show_details=parameters.show_details
        )

    if resource_type is None:
        console_print(
            f"{ERROR}resource_type must be provided when dataframe and resources are None",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not parameters.resource_id:
            resources_df = sql_store.getResourceIdentifiersOfKind(
                kind=resource_type.value,
                field_selectors=parameters.field_selectors,
                details=parameters.show_details,
            )

            if parameters.related_to is not None:
                source_kind, source_id = parameters.related_to
                related_ids = resolve_related_to_identifiers(
                    source_kind=source_kind,
                    source_id=source_id,
                    requested_kind=resource_type,
                    sql_store=sql_store,
                )
                resources_df = resources_df[
                    resources_df["IDENTIFIER"].isin(related_ids)
                ].reset_index(drop=True)

            status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
            return format_default_ado_get_multiple_resources(
                resources=resources_df,
                resource_kind=resource_type,
            )

        resource = sql_store.getResource(
            identifier=parameters.resource_id, kind=resource_type
        )

        if not resource:
            status.stop()
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=resource_type
            )

        return format_default_ado_get_single_resource(
            resource=resource, show_details=parameters.show_details
        )


def handle_ado_get(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds | None" = None,
    dataframe: "pd.DataFrame | None" = None,
    resources: "list[ADOResource] | ADOResource | None" = None,
) -> None:
    """
    Unified handler for all ado get commands.

    Delegates to format-specific handlers that fetch data efficiently.

    Args:
        parameters: Command parameters including output format and filters
        resource_type: Type of resource to fetch (for DB queries)
        dataframe: Pre-built DataFrame (for custom data sources)
        resources: Pre-fetched resources (for custom filtering)

    Raises:
        ValueError: If an identifier column (either "IDENTIFIER" or the value of
            parameters.no_trunc if it's a list with a single element) is not found
            in the provided dataframe when using NAME output format.
    """
    match parameters.output_format:
        case AdoGetSupportedOutputFormats.NAME:
            _handle_name_format(parameters, resource_type, dataframe, resources)
        case AdoGetSupportedOutputFormats.TABLE:
            _handle_table_format(parameters, resource_type, dataframe, resources)
        case AdoGetSupportedOutputFormats.STATS:
            _handle_stats_format(parameters, resource_type, dataframe, resources)
        case AdoGetSupportedOutputFormats.RAW:
            _handle_raw_format(parameters, resource_type)
        case (
            AdoGetSupportedOutputFormats.YAML
            | AdoGetSupportedOutputFormats.JSON
            | AdoGetSupportedOutputFormats.CONFIG
        ):
            _handle_structured_formats(parameters, resource_type, resources)
        case _:
            raise NotImplementedError(
                f"Output format {parameters.output_format} is not implemented"
            )


def _handle_name_format(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds | None",
    dataframe: "pd.DataFrame | None",
    resources: "list[ADOResource] | ADOResource | None",
) -> None:
    """
    Handle NAME output format - output identifiers only (most efficient).

    Raises:
        ValueError: If an identifier column (either "IDENTIFIER" or the value of
            parameters.no_trunc if it's a list with a single element) is not found
            in the provided dataframe.
    """

    # If dataframe provided, extract identifier column
    if dataframe is not None:
        if dataframe.empty:
            console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
            return

        identifier_column = (
            parameters.no_trunc[0]
            if isinstance(parameters.no_trunc, list) and len(parameters.no_trunc) == 1
            else "IDENTIFIER"
        )
        if identifier_column not in dataframe.columns:
            raise ValueError(
                f"Identifier column '{identifier_column}' not found in dataframe. "
                f"Available columns: {', '.join(dataframe.columns)}"
            )
        output = "\n".join(
            str(identifier) for identifier in dataframe[identifier_column]
        )
        _write_or_print_output(output, parameters.output_file)
        return

    # If resources provided, extract identifiers
    if resources is not None:
        if isinstance(resources, list):
            output = "\n".join(resource.identifier for resource in resources)
        else:
            output = resources.identifier
        _write_or_print_output(output, parameters.output_file)
        return

    # Otherwise use efficient DB query
    if resource_type is None:
        console_print(
            f"{ERROR}resource_type must be provided when dataframe and resources are None",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if parameters.resource_id:
            # Single resource: verify it exists and output its identifier
            if not sql_store.containsResourceWithIdentifier(
                identifier=parameters.resource_id, kind=resource_type
            ):
                status.stop()
                raise ResourceDoesNotExistError(
                    resource_id=parameters.resource_id, kind=resource_type
                )
            status.stop()
            _write_or_print_output(parameters.resource_id, parameters.output_file)
        else:
            # Multiple resources: use efficient getResourceIdentifiersOfKind
            identifiers_df = sql_store.getResourceIdentifiersOfKind(
                kind=resource_type.value,
                field_selectors=parameters.field_selectors,
                details=False,
            )

            if parameters.related_to is not None:
                source_kind, source_id = parameters.related_to
                related_ids = resolve_related_to_identifiers(
                    source_kind=source_kind,
                    source_id=source_id,
                    requested_kind=resource_type,
                    sql_store=sql_store,
                )
                identifiers_df = identifiers_df[
                    identifiers_df["IDENTIFIER"].isin(related_ids)
                ].reset_index(drop=True)

            status.stop()
            if identifiers_df.empty:
                console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
                return
            # Output one identifier per line
            output = "\n".join(
                str(identifier) for identifier in identifiers_df["IDENTIFIER"]
            )
            _write_or_print_output(output, parameters.output_file)


def _handle_table_format(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds | None",
    dataframe: "pd.DataFrame | None",
    resources: "list[ADOResource] | ADOResource | None",
) -> None:
    """Handle TABLE output format - render DataFrame as table."""
    output_df = _build_table_output_dataframe(
        parameters=parameters,
        resource_type=resource_type,
        dataframe=dataframe,
        resources=resources,
    )
    return _render_dataframe_table_output(output_df, parameters)


def _handle_stats_format(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds | None",
    dataframe: "pd.DataFrame | None",
    resources: "list[ADOResource] | ADOResource | None",
) -> None:
    """Handle STATS output format - TABLE columns plus measurement/space stats columns.

    Supported for:
    - operations (columns: TOTAL_RESULTS, SUCCESSFUL_RESULTS, FAILED_RESULTS,
      MEASURED_ENTITIES)
    - discovery spaces (columns: EXPERIMENTS, OPERATIONS, EXPLORE_OPERATIONS,
      MEASURED_ENTITIES)
    - sample stores (columns: ENTITIES, RESULTS, EXPERIMENTS)
    - data containers (columns: TABLES, LOCATIONS, KEY_VALUES, DATA_BYTES)

    For any other resource type the handler prints an error message and exits
    with code 1.
    """
    from ado.core import CoreResourceKinds

    _SUPPORTED = {
        CoreResourceKinds.OPERATION,
        CoreResourceKinds.DISCOVERYSPACE,
        CoreResourceKinds.SAMPLESTORE,
        CoreResourceKinds.DATACONTAINER,
    }
    if resource_type is not None and resource_type not in _SUPPORTED:
        console_print(
            f"{ERROR}The 'stats' output format is only supported for operations, "
            f"discovery spaces, sample stores, and data containers.",
            stderr=True,
        )
        raise typer.Exit(1)

    base_df = _build_table_output_dataframe(
        parameters=parameters,
        resource_type=resource_type,
        dataframe=dataframe,
        resources=resources,
    )

    if base_df.empty:
        _render_dataframe_table_output(base_df, parameters)
        return

    # Append stats columns.
    sql_store_for_stats = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_GETTING_OUTPUT_READY) as status:
        if resource_type == CoreResourceKinds.DISCOVERYSPACE:
            enriched_df = format_ado_get_stats_for_spaces(
                base_df,
                sql_store_for_stats,
                spinner=status,
            )
        elif resource_type == CoreResourceKinds.SAMPLESTORE:
            enriched_df = format_ado_get_stats_for_samplestores(
                base_df,
                sql_store_for_stats,
                spinner=status,
            )
        elif resource_type == CoreResourceKinds.DATACONTAINER:
            enriched_df = format_ado_get_stats_for_datacontainers(
                base_df,
                sql_store_for_stats,
                spinner=status,
            )
        else:
            enriched_df = format_ado_get_stats_for_operations(
                base_df,
                sql_store_for_stats,
                spinner=status,
            )

    _render_dataframe_table_output(enriched_df, parameters)


def _handle_raw_format(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds | None",
) -> None:
    """Handle RAW output format - output raw dict representation."""
    if not parameters.resource_id:
        console_print(
            f"{ERROR}Raw output mode is available only when specifying a resource_id",
            stderr=True,
        )
        raise typer.Exit(1)

    if resource_type is None:
        console_print(
            f"{ERROR}resource_type must be provided for RAW format",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB):
        resources = sql_store.getResourceRaw(parameters.resource_id)

    output_content = format_resource_for_ado_get_custom_format(
        to_print=resources, parameters=parameters
    )

    _write_or_print_output(output_content, parameters.output_file)


def _handle_structured_formats(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds | None",
    resources: "list[ADOResource] | ADOResource | None",
) -> None:
    """Handle YAML, JSON, CONFIG formats."""
    # Validate CONFIG format requirements
    if (
        parameters.output_format == AdoGetSupportedOutputFormats.CONFIG
        and not parameters.resource_id
        and resources is None
    ):
        console_print(f"{ERROR}{ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE}", stderr=True)
        raise typer.Exit(1)

    # If resources provided, use them
    if resources is not None:
        output_content = format_resource_for_ado_get_custom_format(
            to_print=resources, parameters=parameters
        )
        _write_or_print_output(output_content, parameters.output_file)
        return

    # Otherwise fetch from DB
    if resource_type is None:
        console_print(
            f"{ERROR}resource_type must be provided when resources are None",
            stderr=True,
        )
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if parameters.resource_id:
            fetched_resources = sql_store.getResource(
                identifier=parameters.resource_id, kind=resource_type
            )
            if not fetched_resources:
                status.stop()
                raise ResourceDoesNotExistError(
                    resource_id=parameters.resource_id, kind=resource_type
                )
        else:
            all_resources = sql_store.getResourcesOfKind(
                kind=resource_type.value,
                field_selectors=parameters.field_selectors,
            )
            if parameters.related_to is not None:
                source_kind, source_id = parameters.related_to
                related_ids = resolve_related_to_identifiers(
                    source_kind=source_kind,
                    source_id=source_id,
                    requested_kind=resource_type,
                    sql_store=sql_store,
                )
                fetched_resources = [
                    r for rid, r in all_resources.items() if rid in related_ids
                ]
            else:
                fetched_resources = list(all_resources.values())

    output_content = format_resource_for_ado_get_custom_format(
        to_print=fetched_resources, parameters=parameters
    )

    _write_or_print_output(output_content, parameters.output_file)


def _write_or_print_output(
    content: str | RenderableType, output_file: pathlib.Path | None
) -> None:
    """Helper to write to file or print to console.

    Args:
        content: String content or rich renderable to output
        output_file: Optional file path. If provided, content is written to file.
    """
    if output_file:
        # Convert to string if it's a rich renderable
        if not isinstance(content, str):
            from ado.utilities.rich import render_to_string

            content = render_to_string(content, auto_width=True)
        output_file.write_text(content)
        console_print(f"{SUCCESS}Output written to {output_file}", stderr=True)
    else:
        console_print(content)


def print_related_resources(
    resource_id: str,
    resource_type: "CoreResourceKinds",
    sql: "SQLStore",
    hide_banner: bool = False,
    max_hops: int | None = None,
) -> None:
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not sql.containsResourceWithIdentifier(identifier=resource_id):
            status.stop()
            raise ResourceDoesNotExistError(resource_id=resource_id, kind=resource_type)

        status.update("Finding related resources")
        related_resources = sql.get_resources_by_relationship(
            kind=resource_type,
            identifier=resource_id,
            max_hops=max_hops,
            identifiers_only=True,
        )

    if not related_resources:
        console_print("There are no related resources", stderr=True)
        return

    if not hide_banner:
        console_print(rich.rule.Rule(title="RELATED RESOURCES"))

    for kind, identifiers in sorted(
        related_resources.items(), key=lambda kv: kv[0].value
    ):
        console_print(cyan(kind.value))
        for identifier in identifiers:
            console_print(f"  - {identifier}")


def strategic_merge_configuration_metadata(
    base: dict[str, typing.Any], patch: dict[str, typing.Any]
) -> dict[str, typing.Any]:
    """
    Strategic merge for metadata dicts: ``labels`` is merged; other top-level
    keys are replaced via ``dict.update`` (oc/kubectl style).
    """
    merged = dict(base)
    overrides = dict(patch)
    if "labels" in overrides:
        new_labels = overrides.pop("labels")
        old_labels = merged.get("labels", {})
        if new_labels is None:
            merged["labels"] = None
        elif old_labels is None:
            merged["labels"] = new_labels
        else:
            merged["labels"] = old_labels | new_labels

    return merged | overrides


def handle_edit_resource_metadata(
    resource_id: str,
    resource_type: "CoreResourceKinds",
    project_context: "ProjectContext",
    editor: AdoEditSupportedEditors,
    metadata_path: pathlib.Path | None = None,
    metadata_patch: str | None = None,
) -> None:
    import subprocess  # noqa: S404
    import tempfile

    import ado.cli.utils.pydantic.serializers

    sql = get_sql_store(project_context=project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        resource = sql.getResource(identifier=resource_id, kind=resource_type)
        if not resource:
            status.stop()
            raise ResourceDoesNotExistError(resource_id=resource_id, kind=resource_type)

    # Non-interactive mode: use patch or patch_file (editor is ignored)
    if metadata_path is not None or metadata_patch is not None:
        try:
            raw = yaml.safe_load(
                metadata_patch
                if metadata_patch is not None
                else metadata_path.read_text()
            )
            if raw is not None and not isinstance(raw, dict):
                console_print(
                    f"{ERROR}The provided metadata must be a YAML/JSON object "
                    f"(mapping), not {type(raw).__name__}.",
                    stderr=True,
                )
                raise typer.Exit(1)
            _ = ConfigurationMetadata.model_validate(raw)
        except (OSError, yaml.YAMLError, ValueError) as e:
            console_print(
                f"{ERROR}The provided metadata was invalid: {e}",
                stderr=True,
            )
            raise typer.Exit(1) from e
        if raw is None:
            new_metadata = {}
        else:
            base_dict = resource.config.metadata.model_dump()
            new_metadata = strategic_merge_configuration_metadata(
                base=base_dict, patch=raw
            )
    else:
        # Interactive mode: use editor
        with tempfile.TemporaryDirectory() as d:
            file = pathlib.Path(d) / pathlib.Path("tmp_metadata.yaml")
            ado.cli.utils.pydantic.serializers.serialise_pydantic_model(
                model=resource.config.metadata,
                output_path=file,
                suppress_success_message=True,
            )

            try:
                subprocess.run([editor.value, file], check=True)  # noqa: S603
            except subprocess.CalledProcessError as e:
                console_print(
                    f"{ERROR}The editor exited with an error: {e}", stderr=True
                )
                raise typer.Exit(1) from e

            new_metadata = yaml.safe_load(file.read_text())

    try:
        resource.config.metadata = ConfigurationMetadata.model_validate(new_metadata)
    except pydantic.ValidationError as e:
        console_print(f"{ERROR}The updated metadata was invalid: {e}", stderr=True)
        raise typer.Exit(1) from e
    with Status(ADO_SPINNER_SAVING_TO_DB):
        sql.updateResource(resource)

    console_print(SUCCESS, stderr=True)


def handle_ado_upgrade(
    parameters: "AdoUpgradeCommandParameters",
    resource_type: "CoreResourceKinds",
) -> None:
    """Upgrade resources of the given type.

    Args:
        parameters: Command parameters
        resource_type: The type of resource to upgrade
    """
    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        resources = sql_store.getResourcesOfKind(
            kind=resource_type.value, ignore_validation_errors=False
        )

        for idx, resource in enumerate(resources.values()):
            status.update(ADO_SPINNER_SAVING_TO_DB + f" ({idx + 1}/{len(resources)})")
            sql_store.updateResource(resource=resource)

    console_print(SUCCESS)


def render_trace_output(
    measurement_requests: list,
    parameters: "AdoShowTraceCommandParameters",
    *,
    include_operation_id: bool = False,
    operation_space_map: dict[str, str] | None = None,
) -> None:
    """Render measurement trace data to the configured output format.

    This is the shared rendering step used by all ``show trace`` handlers.
    It handles YAML serialisation, DataFrame construction, column reordering,
    column hiding, and final output.

    Args:
        measurement_requests: The fetched measurement requests (already filtered).
        parameters: The ``AdoShowTraceCommandParameters`` carrying output format,
            hide_fields, no_trunc, output_file, and unroll_entities settings.
        include_operation_id: When True, each row is stamped with the request's
            own operation ID (read from ``request.operation_id``).
        operation_space_map: When not None, a mapping from operation ID to space ID.
            Each row is stamped with the space ID for its operation.
    """
    import pandas as pd

    from ado.cli.resources.trace_common import (
        REQUEST_COLUMN,
        REQUEST_COLUMNS_MOVE_TO_END,
        RESULT_COLUMN,
        RESULT_COLUMNS_MOVE_TO_END,
        build_request_level_rows,
        build_result_level_rows,
    )
    from ado.utilities.pandas import reorder_dataframe_columns

    # YAML path: serialise the raw pydantic objects and return early
    if parameters.output_format == AdoShowTraceSupportedOutputFormats.YAML:
        yaml_output = pydantic_model_as_yaml(measurement_requests)  # type: ignore[arg-type]
        if parameters.output_file:
            parameters.output_file.write_text(yaml_output)
        else:
            console_print(yaml_output)
        return

    # Build rows for the chosen view mode
    if parameters.unroll_entities:
        rows = build_result_level_rows(
            measurement_requests,
            include_operation_id=include_operation_id,
            operation_space_map=operation_space_map,
        )
        move_to_end = RESULT_COLUMNS_MOVE_TO_END
        id_col = RESULT_COLUMN.REQUEST_ID.value
        space_id_col = RESULT_COLUMN.SPACE_ID.value
        op_id_col = RESULT_COLUMN.OPERATION_ID.value
    else:
        rows = build_request_level_rows(
            measurement_requests,
            include_operation_id=include_operation_id,
            operation_space_map=operation_space_map,
        )
        move_to_end = REQUEST_COLUMNS_MOVE_TO_END
        id_col = REQUEST_COLUMN.REQUEST_ID.value
        space_id_col = REQUEST_COLUMN.SPACE_ID.value
        op_id_col = REQUEST_COLUMN.OPERATION_ID.value

    # Build move_to_start: Request ID always first, then Space ID and
    # Operation ID when present (space before operation).
    move_to_start = [id_col]
    if operation_space_map is not None:
        move_to_start.append(space_id_col)
    if include_operation_id:
        move_to_start.append(op_id_col)

    df = pd.DataFrame(rows)

    df = reorder_dataframe_columns(
        df=df,
        move_to_start=move_to_start,
        move_to_end=move_to_end,
    )

    if parameters.hide_fields:
        df = df.drop(parameters.hide_fields, axis="columns", errors="ignore")

    df_to_output(
        df=df,
        output_format=parameters.output_format.value,
        output_file=parameters.output_file,
        no_trunc=parameters.no_trunc,
    )
