# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.status import Status

from ado.cli.models.parameters import AdoCreateCommandParameters
from ado.cli.models.types import AdoCreateSupportedResourceTypes
from ado.cli.resources.actuator_configuration.create import (
    create_actuator_configuration,
)
from ado.cli.resources.discovery_space.create import create_discovery_space
from ado.cli.utils.output.prints import (
    ADO_CREATE_DRY_RUN_CONFIG_VALID,
    ERROR,
    INFO,
    SUCCESS,
    WARN,
    console_print,
    latest_identifier_for_resource_not_found,
    magenta,
    value_in_configuration_replaced_with_latest_identifier_for_resource,
)
from ado.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from ado.cli.utils.resources.formatters import most_important_status_update
from ado.core import CoreResourceKinds
from ado.core.operation.config import DiscoveryOperationResourceConfiguration
from ado.core.operation.operation import OperationException, OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
)
from ado.core.resources import ADOResourceReference
from ado.modules.operators.errors import OperatorVersionMismatchError


def create_operation(parameters: AdoCreateCommandParameters) -> str | None:

    import ado.modules.operators.orchestrate
    from ado.modules.operators.base import InterruptedOperationError

    try:
        op_resource_configuration: DiscoveryOperationResourceConfiguration = (
            DiscoveryOperationResourceConfiguration.model_validate(
                yaml.safe_load(parameters.resource_configuration_file.read_text())
            )
        )
    except (pydantic.ValidationError, ValueError) as e:
        console_print(
            f"{ERROR}The operation configuration provided was not valid:\n{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e
    except OperatorVersionMismatchError as error:
        console_print(
            f"{ERROR}Operator version mismatch when creating the operation: {error}",
            stderr=True,
        )
        raise typer.Exit(1) from error

    if parameters.override_values:
        op_resource_configuration = override_values_in_pydantic_model(
            model=op_resource_configuration, override_values=parameters.override_values
        )

    if parameters.with_resources:
        if CoreResourceKinds.ACTUATORCONFIGURATION in parameters.with_resources:
            if isinstance(
                parameters.with_resources[CoreResourceKinds.ACTUATORCONFIGURATION], str
            ):
                op_resource_configuration.actuatorConfigurationIdentifiers = [
                    parameters.with_resources[CoreResourceKinds.ACTUATORCONFIGURATION]
                ]
            else:
                op_resource_configuration.actuatorConfigurationIdentifiers = [
                    create_actuator_configuration(
                        AdoCreateCommandParameters(
                            ado_configuration=parameters.ado_configuration,
                            dry_run=False,
                            new_sample_store=False,
                            override_values=[],
                            resource_configuration_file=parameters.with_resources[
                                CoreResourceKinds.ACTUATORCONFIGURATION
                            ],
                            resource_type=AdoCreateSupportedResourceTypes.ACTUATOR_CONFIGURATION,
                            use_default_sample_store=False,
                            with_resources={},
                            use_latest=[],
                        )
                    )
                ]

        if CoreResourceKinds.DISCOVERYSPACE in parameters.with_resources:
            if isinstance(
                parameters.with_resources[CoreResourceKinds.DISCOVERYSPACE], str
            ):
                space_id = parameters.with_resources[CoreResourceKinds.DISCOVERYSPACE]
            else:
                space_id = create_discovery_space(
                    AdoCreateCommandParameters(
                        ado_configuration=parameters.ado_configuration,
                        dry_run=False,
                        new_sample_store=False,
                        override_values=[],
                        resource_configuration_file=parameters.with_resources[
                            CoreResourceKinds.DISCOVERYSPACE
                        ],
                        resource_type=AdoCreateSupportedResourceTypes.DISCOVERY_SPACE,
                        use_default_sample_store=False,
                        with_resources={},
                        use_latest=[],
                    )
                )
            op_resource_configuration.inputs["discoverySpace"] = ADOResourceReference(
                identifier=space_id,
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )

    elif parameters.use_latest:
        reuse_requested_latest_identifiers(
            resource_configuration=op_resource_configuration, parameters=parameters
        )

    if parameters.dry_run:
        console_print(f"{INFO}The operation YAML is syntactically valid.", stderr=True)

    if op_resource_configuration.actuatorConfigurationIdentifiers:
        with Status("Validating actuator configurations for operation") as status:
            try:
                op_resource_configuration.validate_actuatorconfigurations(
                    parameters.ado_configuration.project_context
                )
            except ValueError as e:
                status.stop()
                console_print(
                    f"{ERROR}The provided actuator configurations are "
                    f"not compatible with the discovery space: {e}",
                    stderr=True,
                )
                raise typer.Exit(1) from e

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return None

    try:
        operation_output = ado.modules.operators.orchestrate.orchestrate(
            operation_resource_configuration=op_resource_configuration,
            project_context=parameters.ado_configuration.project_context,
        )
    except ValueError as e:
        console_print(f"{ERROR}Failed to create operation:\n\t{e}", stderr=True)
        raise typer.Exit(1) from e
    except InterruptedOperationError as e:
        console_print(
            f"{ERROR}Created operation with identifier {magenta(e.operation_identifier)} "
            "but it was interrupted.",
            stderr=True,
        )
        raise typer.Exit(3) from None
    except KeyboardInterrupt as e:
        console_print(
            f"{INFO}Operation creation has been stopped due to a keyboard interrupt.",
            stderr=True,
        )
        raise typer.Exit(3) from e
    except OperationException as e:
        console_print(
            f"{ERROR}An unexpected error occurred. "
            f"Operation {magenta(e.operation.identifier)} did not run successfully:\n\n"
            f"{most_important_status_update(e.operation.status).message}",
            stderr=True,
        )
        raise typer.Exit(1) from e
    except BaseException as e:
        console_print(
            f"{ERROR}An unexpected error occurred. Failed to create operation:\n\n{e}",
            stderr=True,
        )
        raise

    return output_operation_result(result=operation_output)


def reuse_requested_latest_identifiers(
    resource_configuration: DiscoveryOperationResourceConfiguration,
    parameters: AdoCreateCommandParameters,
) -> None:
    """Fetch latest resource identifiers from database in a single batch query."""
    from ado.cli.utils.generic.wrappers import get_sql_store

    sql_store = get_sql_store(parameters.ado_configuration.project_context)

    # Batch query for all requested kinds at once
    latest_ids = sql_store.get_latest_resource_identifiers_of_kinds(
        kinds=parameters.use_latest
    )

    # Handle each requested resource kind
    for resource_kind in parameters.use_latest:
        latest_id = latest_ids.get(resource_kind)
        if not latest_id:
            console_print(
                latest_identifier_for_resource_not_found(resource_kind),
                stderr=True,
            )
            raise typer.Exit(1)

        if resource_kind == CoreResourceKinds.ACTUATORCONFIGURATION:
            resource_configuration.actuatorConfigurationIdentifiers = [latest_id]
        elif resource_kind == CoreResourceKinds.DISCOVERYSPACE:
            resource_configuration.inputs["discoverySpace"] = ADOResourceReference(
                identifier=latest_id,
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )
        else:
            # Unknown kind — skip (no field mapping defined)
            continue

        console_print(
            value_in_configuration_replaced_with_latest_identifier_for_resource(
                reused_resource_kind=resource_kind,
                target_resource_kind=CoreResourceKinds.OPERATION,
                replacement_identifier=latest_id,
            ),
            stderr=True,
        )


def output_operation_result(result: OperationOutput) -> str | None:
    # Output some padding
    console_print("", stderr=True)

    match result.exitStatus.exit_state:
        case OperationExitStateEnum.SUCCESS:
            console_print(
                f"{SUCCESS}Created operation with identifier {magenta(result.operation.identifier)} "
                "and it finished successfully."
            )
        case OperationExitStateEnum.ERROR:
            console_print(
                f"{WARN}Created operation with identifier {magenta(result.operation.identifier)} "
                "but it exited with an unexpected error.",
                stderr=True,
            )
            raise typer.Exit(2)
        case OperationExitStateEnum.FAIL:
            console_print(
                f"{ERROR}Created operation with identifier {magenta(result.operation.identifier)} "
                "but it reported that it failed.",
                stderr=True,
            )
            raise typer.Exit(2)
        case _:
            console_print(
                f"{ERROR}Operation exit state {result.exitStatus.exit_state} was unsupported.",
                stderr=True,
            )
            raise typer.Exit(1)

    return result.operation.identifier
