# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


import pathlib
import typing

import pydantic
import rich.rule
import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.types import (
    AdoEditSupportedEditors,
    AdoGetSupportedOutputFormats,
)
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.output.prints import (
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
from orchestrator.cli.utils.resources.formatters import (
    format_default_ado_get_multiple_resources,
    format_default_ado_get_single_resource,
    format_resource_for_ado_get_custom_format,
)
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.metastore.base import ResourceDoesNotExistError
from orchestrator.utilities.rich import dataframe_to_rich_table

if typing.TYPE_CHECKING:
    from orchestrator.cli.models.parameters import (
        AdoGetCommandParameters,
        AdoUpgradeCommandParameters,
    )
    from orchestrator.core import CoreResourceKinds
    from orchestrator.metastore.project import ProjectContext
    from orchestrator.metastore.sqlstore import SQLStore


def handle_ado_get_special_formats(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds",
) -> None:

    if (
        parameters.output_format == AdoGetSupportedOutputFormats.CONFIG
        and not parameters.resource_id
    ):
        console_print(f"{ERROR}{ADO_GET_CONFIG_ONLY_WHEN_SINGLE_RESOURCE}", stderr=True)
        raise typer.Exit(1)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:

        if parameters.output_format == AdoGetSupportedOutputFormats.RAW:

            if not parameters.resource_id:
                status.stop()
                console_print(
                    f"{ERROR}Raw output mode is available only when specifying a resource_id",
                    stderr=True,
                )
                raise typer.Exit(1)

            resources = sql_store.getResourceRaw(parameters.resource_id)

        else:
            if parameters.resource_id:
                resources = sql_store.getResource(
                    identifier=parameters.resource_id, kind=resource_type
                )
                if not resources:
                    status.stop()
                    raise ResourceDoesNotExistError(
                        resource_id=parameters.resource_id, kind=resource_type
                    )
            else:
                resources = list(
                    sql_store.getResourcesOfKind(
                        kind=resource_type.value,
                        field_selectors=parameters.field_selectors,
                    ).values()
                )

        status.stop()
        console_print(
            format_resource_for_ado_get_custom_format(
                to_print=resources, parameters=parameters
            )
        )


def handle_ado_get_default_format(
    parameters: "AdoGetCommandParameters",
    resource_type: "CoreResourceKinds",
) -> None:

    import rich.box

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not parameters.resource_id:
            resources = sql_store.getResourceIdentifiersOfKind(
                kind=resource_type.value,
                field_selectors=parameters.field_selectors,
                details=parameters.show_details,
            )

            status.update(ADO_SPINNER_GETTING_OUTPUT_READY)
            output_df = format_default_ado_get_multiple_resources(
                resources=resources,
                resource_kind=resource_type,
            )

            status.stop()
            if output_df.empty:
                console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
                return

            console_print(
                dataframe_to_rich_table(
                    output_df, box=rich.box.SQUARE, show_index=True, show_edge=True
                )
            )
            return

        resource = sql_store.getResource(
            identifier=parameters.resource_id, kind=resource_type
        )
        status.stop()

        if not resource:
            raise ResourceDoesNotExistError(
                resource_id=parameters.resource_id, kind=resource_type
            )

        output_df = format_default_ado_get_single_resource(
            resource=resource, show_details=parameters.show_details
        )

        console_print(
            dataframe_to_rich_table(output_df, box=rich.box.SQUARE, show_edge=True)
        )


def print_related_resources(
    resource_id: str,
    resource_type: "CoreResourceKinds",
    sql: "SQLStore",
    hide_banner: bool = False,
) -> None:
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        if not sql.containsResourceWithIdentifier(identifier=resource_id):
            status.stop()
            raise ResourceDoesNotExistError(resource_id=resource_id, kind=resource_type)

        status.update("Finding related resources")
        related_resources = sql.getRelatedResourceIdentifiers(resource_id)

    if related_resources.empty:
        console_print("There are no related resources", stderr=True)
        return

    if not hide_banner:
        console_print(rich.rule.Rule(title="RELATED RESOURCES"))
    previous_resource_kind = ""
    for _, row in related_resources.iterrows():
        if row["TYPE"] != previous_resource_kind:
            console_print(cyan(row["TYPE"]))
            previous_resource_kind = row["TYPE"]
        console_print(f"  - {row['IDENTIFIER']}")


def handle_edit_resource_metadata(
    resource_id: str,
    resource_type: "CoreResourceKinds",
    project_context: "ProjectContext",
    editor: AdoEditSupportedEditors,
) -> None:
    import subprocess  # noqa: S404
    import tempfile

    import orchestrator.cli.utils.pydantic.serializers

    sql = get_sql_store(project_context=project_context)
    with Status(ADO_SPINNER_QUERYING_DB) as status:
        resource = sql.getResource(identifier=resource_id, kind=resource_type)
        if not resource:
            status.stop()
            raise ResourceDoesNotExistError(resource_id=resource_id, kind=resource_type)

    with tempfile.TemporaryDirectory() as d:
        file = pathlib.Path(d) / pathlib.Path("tmp_metadata.yaml")
        orchestrator.cli.utils.pydantic.serializers.serialise_pydantic_model(
            model=resource.config.metadata,
            output_path=file,
            suppress_success_message=True,
        )

        try:
            subprocess.run([editor.value, file], check=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            console_print(f"{ERROR}The editor exited with an error: {e}", stderr=True)
            raise typer.Exit(1) from e

        try:
            new_metadata = ConfigurationMetadata.model_validate(
                yaml.safe_load(file.read_text())
            )
        except pydantic.ValidationError as e:
            console_print(f"{ERROR}The updated metadata was invalid: {e}", stderr=True)
            raise typer.Exit(1) from e

    resource.config.metadata = new_metadata
    with Status(ADO_SPINNER_SAVING_TO_DB):
        sql.updateResource(resource)

    console_print(SUCCESS, stderr=True)


def handle_ado_upgrade(
    parameters: "AdoUpgradeCommandParameters",
    resource_type: "CoreResourceKinds",
) -> None:
    """Upgrade resources, optionally applying legacy validators

    Args:
        parameters: Command parameters including legacy validator options
        resource_type: The type of resource to upgrade
    """
    # Import all validator modules to ensure they're registered
    _import_legacy_validators()

    # Handle --list-legacy flag
    if parameters.list_legacy:
        from orchestrator.cli.utils.legacy.list import list_legacy_validators

        list_legacy_validators(resource_type)
        return

    # Get legacy validators if specified
    legacy_validators = None
    if parameters.apply_legacy_validator:
        from orchestrator.core.legacy.registry import LegacyValidatorRegistry

        legacy_validators = []
        for validator_id in parameters.apply_legacy_validator:
            validator = LegacyValidatorRegistry.get_validator(validator_id)
            if validator is None:
                console_print(
                    f"{ERROR}Unknown legacy validator: {validator_id}", stderr=True
                )
                raise typer.Exit(1)
            legacy_validators.append(validator)

    sql_store = get_sql_store(
        project_context=parameters.ado_configuration.project_context
    )

    # Import resource class mapping for validation
    from orchestrator.core import kindmap

    with Status(ADO_SPINNER_QUERYING_DB) as status:
        # When legacy validators are specified, work with raw data
        if legacy_validators:

            identifiers = sql_store.getResourceIdentifiersOfKind(
                kind=resource_type.value
            )

            for idx, identifier in enumerate(identifiers):
                status.update(
                    ADO_SPINNER_SAVING_TO_DB + f" ({idx + 1}/{len(identifiers)})"
                )

                # Get raw data
                resource_dict = sql_store.getResourceRaw(identifier)
                if resource_dict is None:
                    continue

                # Apply legacy validators
                for validator in legacy_validators:
                    resource_dict = validator.validator_function(resource_dict)

                # Validate and save the migrated resource
                resource_class = kindmap[resource_type.value]
                resource = resource_class.model_validate(resource_dict)
                sql_store.updateResource(resource=resource)
        else:
            # Normal upgrade path without legacy validators
            try:
                resources = sql_store.getResourcesOfKind(
                    kind=resource_type.value, ignore_validation_errors=False
                )
            except ValueError as err:
                # Validation error occurred - check if legacy validators can help
                _handle_upgrade_validation_error(err, resource_type, parameters)
                raise typer.Exit(1) from err

            for idx, resource in enumerate(resources.values()):
                status.update(
                    ADO_SPINNER_SAVING_TO_DB + f" ({idx + 1}/{len(resources)})"
                )
                sql_store.updateResource(resource=resource)

    console_print(SUCCESS)


def _import_legacy_validators() -> None:
    """Import all legacy validator modules to ensure they're registered"""
    # Import validator modules to trigger decorator registration
    try:
        # Discovery Space validators
        import orchestrator.core.legacy.validators.discoveryspace.entitysource_to_samplestore  # noqa: F401
        import orchestrator.core.legacy.validators.discoveryspace.properties_field_removal  # noqa: F401

        # Operation validators
        import orchestrator.core.legacy.validators.operation.actuators_field_removal  # noqa: F401
        import orchestrator.core.legacy.validators.operation.randomwalk_mode_to_sampler_config  # noqa: F401

        # Sample Store validators
        import orchestrator.core.legacy.validators.resource.entitysource_to_samplestore  # noqa: F401
        import orchestrator.core.legacy.validators.samplestore.entitysource_migrations  # noqa: F401
        import orchestrator.core.legacy.validators.samplestore.v1_to_v2_csv_migration  # noqa: F401
    except ImportError:
        pass  # Validators may not be available in all installations


def _handle_upgrade_validation_error(
    error: ValueError,
    resource_type: "CoreResourceKinds",
    parameters: "AdoUpgradeCommandParameters",
) -> None:
    """Handle validation errors during upgrade by suggesting legacy validators

    Analyzes the validation error to extract deprecated field names, finds
    applicable legacy validators, and displays helpful suggestions to the user.

    Args:
        error: The ValueError containing validation error details
        resource_type: The type of resource being upgraded
        parameters: The upgrade command parameters
    """
    from rich.console import Console

    from orchestrator.core.legacy.registry import LegacyValidatorRegistry
    from orchestrator.core.resources import CoreResourceKinds

    console = Console()

    # Import all validator modules to ensure they're registered
    _import_legacy_validators()

    # Extract error message
    error_msg = str(error)

    # Try to extract deprecated field names from the error message
    # The error message contains validation errors with field names
    deprecated_fields = []

    # Look for common patterns in pydantic validation errors
    import re

    # Pattern: field_name followed by validation error
    field_patterns = [
        r"kind\s*\n\s*Input should be",  # kind field
        r"moduleType\s*\n\s*Input should be",  # moduleType field
        r"moduleClass\s*\n\s*",  # moduleClass field
        r"moduleName\s*\n\s*",  # moduleName field
        r"constitutivePropertyColumns",  # constitutivePropertyColumns field
        r"propertyMap",  # propertyMap field
        r"entitySourceIdentifier",  # entitySourceIdentifier field
        r"properties\s*\n",  # properties field
        r"actuators\s*\n",  # actuators field
        r"mode\s*\n",  # mode field (for randomwalk)
    ]

    for pattern in field_patterns:
        if re.search(pattern, error_msg, re.IGNORECASE):
            # Extract the field name from the pattern
            field_name = pattern.split(r"\s")[0].split(r"\\")[0]
            if field_name not in deprecated_fields:
                deprecated_fields.append(field_name)

    # Find applicable legacy validators
    validators = []
    if deprecated_fields:
        validators = LegacyValidatorRegistry.find_validators_for_fields(
            resource_type=resource_type, field_names=deprecated_fields
        )

    # If no validators found by field matching, get all validators for this resource type
    if not validators:
        validators = LegacyValidatorRegistry.get_validators_for_resource(resource_type)

    # Display error message
    console.print(
        f"\n[bold red]Validation Error[/bold red] while upgrading {resource_type.value} resources"
    )
    console.print(
        "\n[yellow]Some resources could not be loaded due to deprecated fields or values.[/yellow]"
    )

    if deprecated_fields:
        console.print(
            f"\nDeprecated fields detected: [yellow]{', '.join(deprecated_fields)}[/yellow]"
        )

    if validators:
        console.print(
            "\n[bold cyan]Available legacy validators that may help:[/bold cyan]\n"
        )

        # Map resource types to their CLI names
        resource_name_mapping = {
            CoreResourceKinds.SAMPLESTORE: "samplestore",
            CoreResourceKinds.DISCOVERYSPACE: "discoveryspace",
            CoreResourceKinds.OPERATION: "operation",
            CoreResourceKinds.ACTUATORCONFIGURATION: "actuatorconfiguration",
            CoreResourceKinds.DATACONTAINER: "datacontainer",
        }
        resource_cli_name = resource_name_mapping.get(
            resource_type, resource_type.value
        )

        for validator in validators:
            console.print(f"  • [green]{validator.identifier}[/green]")
            console.print(f"    {validator.description}")
            console.print(f"    Handles: {', '.join(validator.deprecated_fields)}")
            console.print(f"    Deprecated: v{validator.deprecated_from_version}")
            console.print()

        console.print(
            "[bold magenta]To upgrade using legacy validators:[/bold magenta]"
        )
        validator_args = " ".join(
            f"--apply-legacy-validator {v.identifier}" for v in validators
        )
        console.print(f"  ado upgrade {resource_cli_name} {validator_args}")
        console.print()
        console.print("[bold magenta]To list all legacy validators:[/bold magenta]")
        console.print(f"  ado upgrade {resource_cli_name} --list-legacy")
    else:
        console.print(
            "\n[yellow]No legacy validators are available for this resource type.[/yellow]"
        )
        console.print("The resources may be too old or require manual intervention.")

    console.print()
