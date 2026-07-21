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
from ado.core.operation.config import (
    DiscoveryOperationResourceConfiguration,
    OperatorReference,
)
from ado.core.operation.operation import OperationException, OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
)
from ado.modules.operators.errors import OperatorVersionMismatchError


def _operator_input_name_for_kind(
    operation_data: dict,
    kind: CoreResourceKinds,
) -> str:
    """Resolve the operator input parameter name for a resource *kind*.

    Uses the operator's ``required_resource_inputs``. When several inputs share
    *kind*, prefers an input not yet present in ``operation_data["inputs"]``.

    Args:
        operation_data: Raw operation resource configuration dict (YAML).
        kind: Resource kind to bind (e.g. discoveryspace, datacontainer).

    Returns:
        The input parameter identifier to set.

    Raises:
        ValueError: If the operator has no matching input, or multiple matching
            inputs are unset / ambiguous.
    """
    from ado.modules.operators.collections import operator_metadata_for_reference

    module = (operation_data.get("operation") or {}).get("module") or {}
    if "operatorName" not in module or "operationType" not in module:
        raise ValueError(
            "Cannot resolve input name for "
            f"{kind.value!r}: operation.module must use operatorName/operationType."
        )

    metadata = operator_metadata_for_reference(
        OperatorReference(
            operatorName=module["operatorName"],
            operationType=module["operationType"],
        )
    )
    required = metadata.required_resource_inputs
    matching = [d for d in required if d.kind == kind]
    if not matching:
        raise ValueError(
            f"Operator {metadata.name!r} has no resource input of kind {kind.value!r}."
        )
    if len(matching) == 1:
        return matching[0].identifier

    existing_inputs = operation_data.get("inputs") or {}
    unset = [d.identifier for d in matching if d.identifier not in existing_inputs]
    if len(unset) == 1:
        return unset[0]
    names = [d.identifier for d in matching]
    raise ValueError(
        f"Operator {metadata.name!r} has multiple {kind.value!r} inputs "
        f"{names}; set them explicitly in the operation YAML."
    )


def _set_input_reference(
    operation_data: dict,
    kind: CoreResourceKinds,
    identifier: str,
) -> None:
    """Set a single named input reference on raw *operation_data*."""
    input_name = _operator_input_name_for_kind(operation_data, kind)
    inputs = operation_data.setdefault("inputs", {})
    inputs[input_name] = {
        "identifier": identifier,
        "kind": kind.value,
    }


def create_operation(parameters: AdoCreateCommandParameters) -> str | None:

    import ado.modules.operators.orchestrate
    from ado.modules.operators.base import InterruptedOperationError

    try:
        operation_data = yaml.safe_load(
            parameters.resource_configuration_file.read_text()
        )
        if not isinstance(operation_data, dict):
            raise ValueError("Operation configuration must be a YAML mapping.")
    except (yaml.YAMLError, ValueError, OSError) as e:
        console_print(
            f"{ERROR}The operation configuration provided was not valid:\n{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    # Apply --with / --use-latest before model validation so required inputs can
    # be supplied via CLI even when omitted from the YAML.
    try:
        if parameters.with_resources:
            _apply_with_resources(operation_data, parameters)
        elif parameters.use_latest:
            _apply_use_latest(operation_data, parameters)
    except (ValueError, typer.Exit):
        raise
    except Exception as e:
        console_print(
            f"{ERROR}Failed to apply --with/--use-latest to operation inputs:\n{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    try:
        op_resource_configuration: DiscoveryOperationResourceConfiguration = (
            DiscoveryOperationResourceConfiguration.model_validate(operation_data)
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


def _apply_with_resources(
    operation_data: dict,
    parameters: AdoCreateCommandParameters,
) -> None:
    """Apply ``--with`` resource overlays to raw *operation_data*."""
    if CoreResourceKinds.ACTUATORCONFIGURATION in parameters.with_resources:
        if isinstance(
            parameters.with_resources[CoreResourceKinds.ACTUATORCONFIGURATION], str
        ):
            operation_data["actuatorConfigurationIdentifiers"] = [
                parameters.with_resources[CoreResourceKinds.ACTUATORCONFIGURATION]
            ]
        else:
            operation_data["actuatorConfigurationIdentifiers"] = [
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
        if isinstance(parameters.with_resources[CoreResourceKinds.DISCOVERYSPACE], str):
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
        _set_input_reference(operation_data, CoreResourceKinds.DISCOVERYSPACE, space_id)

    if CoreResourceKinds.DATACONTAINER in parameters.with_resources:
        dc_value = parameters.with_resources[CoreResourceKinds.DATACONTAINER]
        if not isinstance(dc_value, str):
            console_print(
                f"{ERROR}--with datacontainer currently supports an existing "
                "identifier only (not a YAML file).",
                stderr=True,
            )
            raise typer.Exit(1)
        _set_input_reference(operation_data, CoreResourceKinds.DATACONTAINER, dc_value)


def _apply_use_latest(
    operation_data: dict,
    parameters: AdoCreateCommandParameters,
) -> None:
    """Apply ``--use-latest`` overlays to raw *operation_data*."""
    from ado.cli.utils.generic.wrappers import get_sql_store

    sql_store = get_sql_store(parameters.ado_configuration.project_context)
    latest_ids = sql_store.get_latest_resource_identifiers_of_kinds(
        kinds=parameters.use_latest
    )

    for resource_kind in parameters.use_latest:
        latest_id = latest_ids.get(resource_kind)
        if not latest_id:
            console_print(
                latest_identifier_for_resource_not_found(resource_kind),
                stderr=True,
            )
            raise typer.Exit(1)

        if resource_kind == CoreResourceKinds.ACTUATORCONFIGURATION:
            operation_data["actuatorConfigurationIdentifiers"] = [latest_id]
        elif resource_kind in (
            CoreResourceKinds.DISCOVERYSPACE,
            CoreResourceKinds.DATACONTAINER,
        ):
            _set_input_reference(operation_data, resource_kind, latest_id)
        else:
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
