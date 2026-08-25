# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
import typing
from typing import Annotated

import typer

import ado.core.operation.config
import ado.metastore.project
from ado.cli.models.parameters import AdoTemplateCommandParameters
from ado.cli.models.types import AdoTemplateSupportedResourceTypes
from ado.cli.resources.actuator_configuration.template import (
    template_actuator_configuration,
)
from ado.cli.resources.context.template import template_context
from ado.cli.resources.discovery_space.template import template_discovery_space
from ado.cli.resources.document.template import template_document
from ado.cli.resources.operation.template import template_operation
from ado.cli.resources.sample_store.template import template_sample_store
from ado.cli.utils.input.parsers import (
    enum_choice_with_plural_parser,
)

if typing.TYPE_CHECKING:
    from ado.cli.core.config import AdoConfiguration

TEMPLATE_OPERATION_PANEL_NAME = "Operation-specific options"
TEMPLATE_ACTUATORCONFIGURATION_PANEL_NAME = "ActuatorConfiguration-specific options"
TEMPLATE_SPACE_PANEL_NAME = "Space-specific options"
TEMPLATE_CONTEXT_PANEL_NAME = "Context-specific options"


def template_resource(
    ctx: typer.Context,
    resource_type: Annotated[
        AdoTemplateSupportedResourceTypes,
        typer.Argument(
            ...,
            help="The kind of the resource to template.",
            show_default=False,
            parser=enum_choice_with_plural_parser(AdoTemplateSupportedResourceTypes),
            metavar=f"[{'|'.join(m.value for m in AdoTemplateSupportedResourceTypes)}]",
        ),
    ],
    operator_name: Annotated[
        str | None,
        typer.Option(
            help="""
            Name of the operator to use in the operation. If unset, a generic operation will be output.

            Disregarded when resource type is not operation.
            """,
            rich_help_panel=TEMPLATE_OPERATION_PANEL_NAME,
            show_default=False,
        ),
    ] = None,
    actuator_identifier: Annotated[
        str | None,
        typer.Option(
            help="""
            Identifier of the actuator to template a configuration for.
            If unset, a generic actuatorconfiguration will be output.

            Disregarded when resource type is not actuatorconfiguration.
            """,
            rich_help_panel=TEMPLATE_ACTUATORCONFIGURATION_PANEL_NAME,
            show_default=False,
        ),
    ] = None,
    from_experiments: Annotated[
        list[str] | None,
        typer.Option(
            "--from-experiment",
            "-e",
            help="""
            Identifier of the experiments to template a space for.
            Can be specified multiple times.
            If unset, a generic actuatorconfiguration will be output.

            If an actuator id is required to uniquely identify an experiment,
            include it in the resource id as actuator_id.experiment_id.

            Disregarded when resource type is not space.
            """,
            rich_help_panel=TEMPLATE_SPACE_PANEL_NAME,
            show_default=False,
        ),
    ] = None,
    operator_type: Annotated[
        ado.core.operation.config.DiscoveryOperationEnum | None,
        typer.Option(
            help="""
            Type of the operator to use in the operation.
            If unset, an attempt will be made to find the operator by name.

            Disregarded when resource type is not operation.
            """,
            rich_help_panel=TEMPLATE_OPERATION_PANEL_NAME,
            show_default=False,
        ),
    ] = None,
    template_local_context: Annotated[
        bool,
        typer.Option(
            "--local-context",
            help="Reference a local in the context.",
            rich_help_panel=TEMPLATE_CONTEXT_PANEL_NAME,
        ),
    ] = False,
    output_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--output-file",
            help="Write output to the specified file instead of stdout.",
            show_default=False,
            writable=True,
        ),
    ] = None,
    include_schema: Annotated[
        bool,
        typer.Option(
            "--include-schema",
            help="Output the JSON schema of the requested model in addition to the template.",
        ),
    ] = False,
    parameters_only_schema: Annotated[
        bool,
        typer.Option(
            help="""
            When set (default) and using --include-schema, the schema output will be
            that of the operator parameters, not of the operation.

            Disregarded when resource type is not operation.
             """,
            rich_help_panel=TEMPLATE_OPERATION_PANEL_NAME,
        ),
    ] = True,
) -> None:
    """
    Output templates for creating resources and contexts.

    See https://ibm.github.io/ado/latest/cli-reference/#ado-template
    for detailed documentation and examples.

    Examples:

    # Create a template for a local context
    ado template context --local-context

    # Create a template for an operation that uses the ray_tune operator
    ado template operation --operator-name ray_tune
    """
    ado_configuration: AdoConfiguration = ctx.obj

    parameters = AdoTemplateCommandParameters(
        actuator_identifier=actuator_identifier,
        ado_configuration=ado_configuration,
        from_experiments=from_experiments,
        include_schema=include_schema,
        operator_name=operator_name,
        operator_type=operator_type,
        output_file=output_file,
        parameters_only_schema=parameters_only_schema,
        template_local_context=template_local_context,
    )

    method_mapping = {
        AdoTemplateSupportedResourceTypes.ACTUATOR_CONFIGURATION: template_actuator_configuration,
        AdoTemplateSupportedResourceTypes.CONTEXT: template_context,
        AdoTemplateSupportedResourceTypes.DISCOVERY_SPACE: template_discovery_space,
        AdoTemplateSupportedResourceTypes.DOCUMENT: template_document,
        AdoTemplateSupportedResourceTypes.SAMPLE_STORE: template_sample_store,
        AdoTemplateSupportedResourceTypes.OPERATION: template_operation,
    }

    method_mapping[resource_type](parameters=parameters)


def register_template_command(app: typer.Typer) -> None:
    app.command(
        name="template",
        no_args_is_help=True,
    )(template_resource)
