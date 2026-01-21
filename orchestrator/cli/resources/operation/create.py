# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic
import typer
import yaml
from rich.status import Status

from orchestrator.cli.models.parameters import AdoCreateCommandParameters
from orchestrator.cli.models.types import AdoCreateSupportedResourceTypes
from orchestrator.cli.resources.actuator_configuration.create import (
    create_actuator_configuration,
)
from orchestrator.cli.resources.discovery_space.create import create_discovery_space
from orchestrator.cli.utils.output.prints import (
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
from orchestrator.cli.utils.pydantic.updaters import override_values_in_pydantic_model
from orchestrator.cli.utils.resources.formatters import most_important_status_update
from orchestrator.core import CoreResourceKinds
from orchestrator.core.operation.config import (
    DiscoveryOperationResourceConfiguration,
    OperatorModuleConf,
)
from orchestrator.core.operation.operation import OperationException, OperationOutput
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
)


def create_operation(parameters: AdoCreateCommandParameters) -> str | None:

    import orchestrator.modules.operators.orchestrate
    from orchestrator.modules.actuators.base import MeasurementError
    from orchestrator.modules.operators.base import InterruptedOperationError

    try:
        op_resource_configuration = (
            DiscoveryOperationResourceConfiguration.model_validate(
                yaml.safe_load(parameters.resource_configuration_file.read_text())
            )
        )
        validate_operation(op_resource_configuration)
    except (pydantic.ValidationError, ValueError) as e:
        console_print(
            f"{ERROR}The operation configuration provided was not valid:\n{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    if parameters.override_values:
        op_resource_configuration = override_values_in_pydantic_model(
            model=op_resource_configuration, override_values=parameters.override_values
        )
        validate_operation(op_resource_configuration)

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
                op_resource_configuration.spaces = [
                    parameters.with_resources[CoreResourceKinds.DISCOVERYSPACE]
                ]
            else:
                op_resource_configuration.spaces = [
                    create_discovery_space(
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
                ]

    elif parameters.use_latest:
        reuse_requested_latest_identifiers(
            resource_configuration=op_resource_configuration, parameters=parameters
        )

    validate_operation(op_resource_configuration)

    if len(op_resource_configuration.spaces) > 1:
        console_print(
            f"{ERROR}the spaces field only supports one value for now.", stderr=True
        )
        raise typer.Exit(1)

    with Status("Validating actuator configurations for operation") as status:
        try:
            op_resource_configuration.validate_actuatorconfigurations(
                parameters.ado_configuration.project_context
            )
        except ValueError as e:
            status.stop()
            console_print(
                f"{ERROR}The actuator configuration validation failed:\n{e}",
                stderr=True,
            )
            raise typer.Exit(1) from e

    if parameters.dry_run:
        console_print(ADO_CREATE_DRY_RUN_CONFIG_VALID, stderr=True)
        return None

    try:
        operation_output = orchestrator.modules.operators.orchestrate.orchestrate(
            operation_resource_configuration=op_resource_configuration,
            project_context=parameters.ado_configuration.project_context,
            discovery_space_identifier=op_resource_configuration.spaces[0],
        )

    except MeasurementError as e:
        console_print(
            f"{ERROR}A measurement error was encountered while running the operation:\n\t{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e
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

    # Save the identifier of the resource we created
    # for reuse
    parameters.ado_configuration.latest_resource_ids[CoreResourceKinds.OPERATION] = (
        operation_output.operation.identifier
    )

    return output_operation_result(result=operation_output)


def validate_operation(
    resource_configuration: DiscoveryOperationResourceConfiguration,
) -> None:
    import orchestrator.modules.operators.base

    if isinstance(
        resource_configuration.operation.module,
        OperatorModuleConf,
    ):
        module_name = resource_configuration.operation.module.moduleName
        module_class = resource_configuration.operation.module.moduleClass
        import importlib

        try:
            operation: orchestrator.modules.operators.base.DiscoveryOperationBase = (
                getattr(importlib.import_module(module_name), module_class)
            )
        except ModuleNotFoundError as e:
            console_print(
                f"{ERROR}Cannot run operation. Operator {module_name}.{module_class} is not installed.",
                stderr=True,
            )
            raise typer.Exit(1) from e

        operation.validateOperationParameters(
            resource_configuration.operation.parameters
        )

    # AP: it is an OperatorFunctionConf
    else:

        import orchestrator.modules.operators.collections

        configuration_model = (
            orchestrator.modules.operators.collections.operationCollectionMap[
                resource_configuration.operation.module.operationType
            ].configuration_model_for_operation(
                resource_configuration.operation.module.operatorName
            )
        )

        if not configuration_model:
            console_print(
                f"{WARN}No configuration model was available for operation "
                f"{resource_configuration.operation.module.operatorName} of type "
                f"{resource_configuration.operation.module.operationType}",
                stderr=True,
            )
            return

        configuration_model.model_validate(resource_configuration.operation.parameters)


def reuse_requested_latest_identifiers(
    resource_configuration: DiscoveryOperationResourceConfiguration,
    parameters: AdoCreateCommandParameters,
) -> None:
    updated = False

    if CoreResourceKinds.ACTUATORCONFIGURATION in parameters.use_latest:
        latest_recorded_actuator_configuration = (
            parameters.ado_configuration.latest_resource_ids.get(
                CoreResourceKinds.ACTUATORCONFIGURATION
            )
        )

        if not latest_recorded_actuator_configuration:
            console_print(
                latest_identifier_for_resource_not_found(
                    CoreResourceKinds.ACTUATORCONFIGURATION
                ),
                stderr=True,
            )
            raise typer.Exit(1)

        updated = True
        resource_configuration.actuatorConfigurationIdentifiers = [
            latest_recorded_actuator_configuration
        ]

        console_print(
            value_in_configuration_replaced_with_latest_identifier_for_resource(
                reused_resource_kind=CoreResourceKinds.ACTUATORCONFIGURATION,
                target_resource_kind=CoreResourceKinds.OPERATION,
                replacement_identifier=latest_recorded_actuator_configuration,
            ),
            stderr=True,
        )

    if CoreResourceKinds.DISCOVERYSPACE in parameters.use_latest:
        latest_recorded_space = parameters.ado_configuration.latest_resource_ids.get(
            CoreResourceKinds.DISCOVERYSPACE
        )
        if not latest_recorded_space:
            console_print(
                latest_identifier_for_resource_not_found(
                    CoreResourceKinds.DISCOVERYSPACE
                ),
                stderr=True,
            )
            raise typer.Exit(1)

        updated = True
        resource_configuration.spaces = [latest_recorded_space]
        console_print(
            value_in_configuration_replaced_with_latest_identifier_for_resource(
                reused_resource_kind=CoreResourceKinds.DISCOVERYSPACE,
                target_resource_kind=CoreResourceKinds.OPERATION,
                replacement_identifier=latest_recorded_space,
            ),
            stderr=True,
        )

    if updated:
        validate_operation(resource_configuration)


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
