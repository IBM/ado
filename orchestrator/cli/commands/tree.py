# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Display resource relationship trees for the active project context."""

from __future__ import annotations

import pathlib  # noqa: TC003
import typing
from typing import Annotated

import typer

from orchestrator.cli.exceptions.handlers import handle_resource_does_not_exist
from orchestrator.cli.models.choice import HiddenPluralChoice
from orchestrator.cli.models.parameters import AdoTreeCommandParameters
from orchestrator.cli.models.types import (
    AdoTreeSupportedOutputFormats,
    AdoTreeSupportedResourceTypes,
)
from orchestrator.cli.utils.generic.common import get_effective_resource_id
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.input.parsers import parse_key_value_pairs
from orchestrator.cli.utils.output.prints import ERROR, SUCCESS, console_print
from orchestrator.cli.utils.output.tree_renderer import (
    render_tree_forest_to_flat,
    render_tree_forest_to_json,
    render_tree_forest_to_text,
)
from orchestrator.cli.utils.queries.parser import prepare_query_filters_for_db
from orchestrator.core.resource_tree import (
    ResourceTreeBuilder,
    ResourceTreeOptions,
    collect_tree_identifiers,
    enrich_tree_nodes,
)
from orchestrator.metastore.base import ResourceDoesNotExistError

if typing.TYPE_CHECKING:
    from orchestrator.cli.core.config import AdoConfiguration

TREE_OPTIONS = "Tree options"
OUTPUT_CONFIGURATION_OPTIONS = "Output configuration options"
FILTER_OPTIONS = "Filter options"


def _resolve_scoped_root_identifier(
    parameters: AdoTreeCommandParameters,
) -> str | None:
    """Resolve the scoped root identifier when the command is resource-scoped."""
    if parameters.from_resource_id is not None:
        return parameters.from_resource_id

    if parameters.resource_type is None:
        return None

    if parameters.resource_id is None and not parameters.use_latest:
        console_print(
            f"{ERROR}You must specify a resource id, --use-latest, or --from when "
            "scoping the tree to a resource type",
            stderr=True,
        )
        raise typer.Exit(1)

    return get_effective_resource_id(
        explicit_resource_id=parameters.resource_id,
        resource_type=parameters.resource_type.value,
        project_context=parameters.ado_configuration.project_context,
    )


def _write_or_print_output(content: str, output_file: pathlib.Path | None) -> None:
    if output_file is not None:
        output_file.write_text(content)
        console_print(f"{SUCCESS}Output written to {output_file}", stderr=True)
    else:
        console_print(content)


def render_resource_tree(parameters: AdoTreeCommandParameters) -> None:
    """Build and render a resource tree for the active project context."""
    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    scoped_root_identifier = _resolve_scoped_root_identifier(parameters)

    if (
        scoped_root_identifier is not None
        and not sql_store.containsResourceWithIdentifier(
            identifier=scoped_root_identifier
        )
    ):
        raise ResourceDoesNotExistError(resource_id=scoped_root_identifier)

    builder = ResourceTreeBuilder(sql_store)
    matching_identifiers = builder.matching_identifiers_for_selectors(
        parameters.field_selectors
    )
    kind_filter = (
        frozenset(parameters.kind_filter)
        if parameters.kind_filter is not None
        else None
    )
    options = ResourceTreeOptions(
        all_relationships=parameters.all_relationships,
        invert=parameters.invert,
        depth=parameters.depth,
        dedupe=parameters.dedupe,
        include_orphans=parameters.include_orphans,
        kind_filter=kind_filter,
        matching_identifiers=matching_identifiers,
        scoped_root_identifier=scoped_root_identifier,
        sort=parameters.sort,
    )
    forest = builder.build(options)

    needs_fetch = parameters.sort or parameters.names or parameters.metadata
    if forest and needs_fetch:
        resources = sql_store.getResources(list(collect_tree_identifiers(forest)))
        forest = enrich_tree_nodes(
            forest,
            resources,
            show_names=parameters.names,
            show_age=parameters.sort,
            show_metadata=parameters.metadata,
        )

    match parameters.output_format:
        case AdoTreeSupportedOutputFormats.JSON:
            content = render_tree_forest_to_json(forest)
        case AdoTreeSupportedOutputFormats.FLAT:
            content = render_tree_forest_to_flat(forest)
        case AdoTreeSupportedOutputFormats.TREE:
            content = render_tree_forest_to_text(
                forest,
                show_names=parameters.names,
                show_age=parameters.sort,
                show_metadata=parameters.metadata,
            )

    _write_or_print_output(content=content, output_file=parameters.output_file)


def tree_resources(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoTreeSupportedResourceTypes | None,
        typer.Argument(
            help="Optional resource type to scope the tree.",
            show_default=False,
            click_type=HiddenPluralChoice(AdoTreeSupportedResourceTypes),
        ),
    ] = None,
    resource_id: Annotated[
        str | None,
        typer.Argument(
            help="Optional resource identifier to scope the tree.",
            show_default=False,
        ),
    ] = None,
    from_resource_id: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Scope the tree to the resource with this identifier.",
            show_default=False,
        ),
    ] = None,
    use_latest: Annotated[
        bool,
        typer.Option(
            "--use-latest",
            help="Use the latest identifier of the selected resource type when "
            "scoping the tree.",
            show_default=False,
        ),
    ] = False,
    invert: Annotated[
        bool,
        typer.Option(
            "--invert",
            "--reverse",
            help="Walk ancestors (providers) instead of descendants (outputs).",
            show_default=False,
        ),
    ] = False,
    depth: Annotated[
        int | None,
        typer.Option(
            "--depth",
            "-d",
            min=0,
            help="Maximum number of hops from each root.",
            show_default=False,
        ),
    ] = None,
    all_relationships: Annotated[
        bool,
        typer.Option(
            "--all-relationships",
            help="Include input-reference edges such as actuatorconfiguration→operation.",
            show_default=False,
        ),
    ] = False,
    dedupe: Annotated[
        bool,
        typer.Option(
            "--dedupe",
            help="Collapse repeated subtrees when the same node appears multiple times.",
            show_default=False,
        ),
    ] = False,
    include_orphans: Annotated[
        bool,
        typer.Option(
            "--include-orphans",
            help="Append resources with no relationships after the main forest.",
            show_default=False,
        ),
    ] = False,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help="Comma-separated resource kinds to include, retaining ancestor paths.",
            show_default=False,
        ),
    ] = None,
    query: Annotated[
        list[str] | None,
        typer.Option(
            "--query",
            "-q",
            help="Filter nodes by resource field values (JSON). Can be repeated.",
            show_default=False,
        ),
    ] = None,
    labels: Annotated[
        list[str] | None,
        typer.Option(
            "--label",
            help="Filter nodes by metadata labels in key=value format. Can be repeated.",
            show_default=False,
        ),
    ] = None,
    sort: Annotated[
        bool,
        typer.Option(
            "--sort",
            help="Order siblings by created timestamp and show age in node labels.",
            show_default=False,
        ),
    ] = False,
    names: Annotated[
        bool,
        typer.Option(
            "--names",
            help="Show config.metadata.name in brackets when set.",
            show_default=False,
        ),
    ] = False,
    metadata: Annotated[
        bool,
        typer.Option(
            "--metadata",
            help="Include description and labels in node labels.",
            show_default=False,
        ),
    ] = False,
    output_format: Annotated[
        AdoTreeSupportedOutputFormats,
        typer.Option(
            "--output",
            "-o",
            help="Output format.",
            case_sensitive=False,
        ),
    ] = AdoTreeSupportedOutputFormats.TREE,
    output_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output-file",
            help="Write formatted output to this file instead of stdout.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """
    Display resource relationship trees for the active project context.

    By default shows workflow lineage from sample stores downward. Use
    --all-relationships to include actuator configurations and other
    input-reference edges as separate roots.

    See https://ibm.github.io/ado/getting-started/ado/#ado-tree for examples.
    """
    ado_configuration: AdoConfiguration = ctx.obj

    if from_resource_id is not None and resource_id is not None:
        console_print(
            f"{ERROR}Specify either a resource id argument or --from, not both.",
            stderr=True,
        )
        raise typer.Exit(1)

    if from_resource_id is not None and resource_type is not None:
        console_print(
            f"{ERROR}--from cannot be combined with a resource type argument.",
            stderr=True,
        )
        raise typer.Exit(1)

    try:
        field_selectors = prepare_query_filters_for_db(parse_key_value_pairs(query))
        if labels:
            for parsed_label in parse_key_value_pairs(labels):
                for key, value in parsed_label.items():
                    field_selectors.extend(
                        prepare_query_filters_for_db(
                            {"config.metadata.labels": {key: value}}
                        )
                    )
    except ValueError as error:
        console_print(f"{ERROR}{error}", stderr=True)
        raise typer.Exit(1) from error

    kind_filter = [part.strip() for part in kind.split(",")] if kind else None

    parameters = AdoTreeCommandParameters(
        ado_configuration=ado_configuration,
        all_relationships=all_relationships,
        dedupe=dedupe,
        depth=depth,
        field_selectors=field_selectors,
        from_resource_id=from_resource_id,
        include_orphans=include_orphans,
        invert=invert,
        kind_filter=kind_filter,
        metadata=metadata,
        names=names,
        output_file=output_file,
        output_format=output_format,
        resource_id=resource_id if from_resource_id is None else from_resource_id,
        resource_type=resource_type,
        sort=sort,
        use_latest=use_latest,
    )

    try:
        render_resource_tree(parameters=parameters)
    except ResourceDoesNotExistError as error:
        handle_resource_does_not_exist(
            error=error, project_context=ado_configuration.project_context
        )


def register_tree_command(app: typer.Typer) -> None:
    """Register the ado tree command."""
    app.command(
        name="tree",
        no_args_is_help=False,
        options_metavar="",
    )(tree_resources)
