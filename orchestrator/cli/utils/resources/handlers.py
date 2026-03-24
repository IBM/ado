# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import logging
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

logger = logging.getLogger(__name__)

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
    # Import validators package to trigger registration via __init__.py
    import orchestrator.core.legacy.validators  # noqa: F401

    # Handle --list-legacy-validators flag
    if parameters.list_legacy_validators:
        from orchestrator.cli.utils.legacy.list import list_legacy_validators

        list_legacy_validators(resource_type)
        return

    # Get legacy validators if specified
    legacy_validators = None
    if parameters.apply_legacy_validator:
        from orchestrator.core.legacy.registry import LegacyValidatorRegistry

        # Validate all validator IDs exist and match resource type
        invalid_validators = []
        mismatched_validators = []
        for validator_id in parameters.apply_legacy_validator:
            validator = LegacyValidatorRegistry.get_validator(validator_id)
            if validator is None:
                invalid_validators.append(validator_id)
            elif validator.resource_type != resource_type:
                mismatched_validators.append(
                    (validator_id, validator.resource_type, resource_type)
                )

        if invalid_validators:
            console_print(
                f"{ERROR}Unknown legacy validator(s): {', '.join(invalid_validators)}",
                stderr=True,
            )
            raise typer.Exit(1)

        if mismatched_validators:
            for validator_id, validator_type, expected_type in mismatched_validators:
                console_print(
                    f"{ERROR}Validator '{validator_id}' is for {validator_type.value} resources, "
                    f"but you are upgrading {expected_type.value} resources",
                    stderr=True,
                )
            raise typer.Exit(1)

        # Resolve dependencies and order validators
        try:
            ordered_ids, missing_deps = LegacyValidatorRegistry.resolve_dependencies(
                parameters.apply_legacy_validator
            )

            if missing_deps:
                console_print(
                    f"{ERROR}Missing validator dependencies: {', '.join(missing_deps)}",
                    stderr=True,
                )
                raise typer.Exit(1)

            # Get validators in correct order
            legacy_validators = []
            for validator_id in ordered_ids:
                validator = LegacyValidatorRegistry.get_validator(validator_id)
                if validator is not None:
                    legacy_validators.append(validator)

            # Log the ordering
            if len(ordered_ids) > len(parameters.apply_legacy_validator):
                logger.info(
                    f"Auto-included dependencies: {[vid for vid in ordered_ids if vid not in parameters.apply_legacy_validator]}"
                )

            logger.debug(
                f"Validators in execution order: {[v.identifier for v in legacy_validators]}"
            )

        except ValueError as e:
            # Circular dependency detected
            console_print(f"{ERROR}{e}", stderr=True)
            raise typer.Exit(1) from e

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

            # Phase 1: Collect and validate all migrations (transaction safety)
            # Validate all resources before saving any to ensure atomicity
            migrations = []
            resource_class = kindmap[resource_type.value]

            for idx, identifier in enumerate(identifiers["IDENTIFIER"]):
                status.update(
                    ADO_SPINNER_QUERYING_DB
                    + f" - Validating ({idx + 1}/{len(identifiers)})"
                )

                # Get raw data
                resource_dict = sql_store.getResourceRaw(identifier)
                if resource_dict is None:
                    continue

                # Apply legacy validators
                try:
                    for validator in legacy_validators:
                        logger.debug(
                            f"Applying validator: {validator.identifier} to {identifier}"
                        )
                        resource_dict = validator.validator_function(resource_dict)
                        logger.debug(
                            f"Validator {validator.identifier} completed for {identifier}"
                        )

                    # Validate the migrated resource (don't save yet)
                    resource = resource_class.model_validate(resource_dict)
                    migrations.append((identifier, resource))

                except Exception as e:
                    logger.error(f"Migration failed for {identifier}: {e}")
                    console_print(
                        f"{ERROR}Migration validation failed for {identifier}: {e}",
                        stderr=True,
                    )
                    console_print(
                        f"{ERROR}No resources were modified (all-or-nothing transaction safety)",
                        stderr=True,
                    )
                    raise typer.Exit(1) from e

            # Phase 2: All validations passed, now save all resources
            logger.info(
                f"All {len(migrations)} resources validated successfully, applying changes..."
            )

            for idx, (identifier, migrated_resource) in enumerate(migrations):
                status.update(
                    ADO_SPINNER_SAVING_TO_DB + f" ({idx + 1}/{len(migrations)})"
                )

                try:
                    sql_store.updateResource(resource=migrated_resource)
                except Exception as e:
                    logger.error(f"Failed to save {identifier}: {e}")
                    console_print(
                        f"{ERROR}Failed to save {identifier}. Database may be in inconsistent state.",
                        stderr=True,
                    )
                    console_print(
                        f"{ERROR}Manual intervention may be required to restore consistency.",
                        stderr=True,
                    )
                    raise typer.Exit(1) from e
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

    from orchestrator.cli.utils.legacy.common import (
        extract_deprecated_fields_from_value_error,
        print_validator_suggestions,
    )
    from orchestrator.core.legacy.registry import LegacyValidatorRegistry

    console = Console()

    # Import validators package to trigger registration via __init__.py
    import orchestrator.core.legacy.validators  # noqa: F401

    # Extract field paths, error details, and leaf field names from the error
    full_field_paths, field_errors, leaf_field_names = (
        extract_deprecated_fields_from_value_error(error, resource_type)
    )

    # Find applicable legacy validators using leaf field names for better matching
    validators = []
    if leaf_field_names:
        validators = LegacyValidatorRegistry.find_validators_for_fields(
            resource_type=resource_type, field_names=leaf_field_names
        )

    # If no validators found by field matching, get all validators for this resource type
    if not validators:
        validators = LegacyValidatorRegistry.get_validators_for_resource(resource_type)

    # Display error message
    console.print(
        f"\n[bold red]Validation Error[/bold red] while upgrading {resource_type.value} resources"
    )
    console.print(
        "\n[yellow]Some resources could not be loaded due to validation errors.[/yellow]"
    )

    if full_field_paths:
        console.print(
            f"\n[bold]Fields with validation errors:[/bold] [yellow]{len(full_field_paths)} field(s)[/yellow]"
        )
        # Show detailed error messages for each field path
        console.print("\n[bold]Error details:[/bold]")
        for field_path in sorted(full_field_paths):
            console.print(f"  • [cyan]{field_path}[/cyan]:")
            for error_msg in field_errors.get(field_path, []):
                console.print(f"    - {error_msg}")

    if validators:
        print_validator_suggestions(
            validators=validators,
            resource_type=resource_type,
            console=console,
            show_all_validators=True,
        )
    else:
        console.print(
            "\n[yellow]No legacy validators are available for this resource type.[/yellow]"
        )
        console.print("The resources may be too old or require manual intervention.")

    console.print()
