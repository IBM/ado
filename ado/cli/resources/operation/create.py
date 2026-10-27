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
from ado.core.resources import ADOResourcePropertyDescriptor, ADOResourceReference
from ado.modules.operators.errors import OperatorVersionMismatchError


def _operator_input_descriptor_for_kind(
    raw_operation_configuration: dict,
    kind: CoreResourceKinds,
) -> ADOResourcePropertyDescriptor:
    """Resolve the operator input descriptor for a resource *kind*.

    Uses the operator's ``required_resource_inputs``. When several inputs share
    *kind*, prefers a unique list input, then an input that is not yet present.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict.
        kind: Resource kind to bind.

    Returns:
        The resource-input descriptor to set.

    Raises:
        ValueError: If the operator has no matching input, or multiple matching
            inputs are unset.
    """
    from ado.modules.operators.collections import operator_metadata_for_reference

    module = raw_operation_configuration.get("operation", {}).get("module", {})
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
    inputs_of_kind = [
        descriptor
        for descriptor in metadata.required_resource_inputs
        if descriptor.kind == kind
    ]
    if not inputs_of_kind:
        raise ValueError(
            f"Operator {metadata.name!r} has no resource input of kind {kind.value!r}."
        )
    if len(inputs_of_kind) == 1:
        return inputs_of_kind[0]

    list_inputs = [descriptor for descriptor in inputs_of_kind if descriptor.isList]
    if len(list_inputs) == 1:
        return list_inputs[0]

    present_inputs = raw_operation_configuration.get("inputs", {})
    missing_inputs = [
        descriptor
        for descriptor in inputs_of_kind
        if descriptor.identifier not in present_inputs
    ]
    if len(missing_inputs) == 1:
        return missing_inputs[0]
    input_identifiers_of_kind = [descriptor.identifier for descriptor in inputs_of_kind]
    raise ValueError(
        f"Operator {metadata.name!r} has multiple {kind.value!r} inputs "
        f"{input_identifiers_of_kind}; set them explicitly in the operation YAML."
    )


def _add_operation_input_for_reference(
    raw_operation_configuration: dict,
    reference: ADOResourceReference,
) -> None:
    """Set one named input reference on a raw operation configuration.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict.
        reference: Resource reference to store. ``kind`` must be set.

    Raises:
        ValueError: If the operator input name cannot be resolved, or if
            *reference* has no kind.
    """
    if reference.kind is None:
        raise ValueError("Operation input references must include a resource kind.")
    descriptor = _operator_input_descriptor_for_kind(
        raw_operation_configuration, reference.kind
    )
    payload = {
        "identifier": reference.identifier,
        "kind": reference.kind.value,
    }
    inputs = raw_operation_configuration.setdefault("inputs", {})
    if descriptor.isList:
        existing = inputs.get(descriptor.identifier)
        if isinstance(existing, list):
            existing.append(payload)
        elif existing is None:
            inputs[descriptor.identifier] = [payload]
        else:
            inputs[descriptor.identifier] = [existing, payload]
    else:
        inputs[descriptor.identifier] = payload
    if reference.kind == CoreResourceKinds.DISCOVERYSPACE:
        raw_operation_configuration.pop("spaces", None)


def create_operation(parameters: AdoCreateCommandParameters) -> str | None:

    import ado.modules.operators.orchestrate
    from ado.modules.operators.base import InterruptedOperationError

    try:
        raw_operation_configuration = yaml.load(
            parameters.resource_configuration_file.read_text(), Loader=SafeLoader
        )
        if not isinstance(raw_operation_configuration, dict):
            raise ValueError("Operation configuration must be a YAML mapping.")
    except (yaml.YAMLError, ValueError, OSError) as e:
        console_print(
            f"{ERROR}The operation configuration provided was not valid:\n{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    try:
        if parameters.with_resources:
            _apply_with_resources(raw_operation_configuration, parameters)
        elif parameters.use_latest:
            _apply_use_latest(raw_operation_configuration, parameters)
    except Exception as e:
        console_print(
            f"{ERROR}Failed to apply --with/--use-latest to operation inputs:\n{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    try:
        op_resource_configuration: DiscoveryOperationResourceConfiguration = (
            DiscoveryOperationResourceConfiguration.model_validate(
                raw_operation_configuration
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

    if len(op_resource_configuration.spaces) > 1:
        console_print(
            f"{ERROR}the spaces field only supports one value for now.", stderr=True
        )
        raise typer.Exit(1)

    if parameters.dry_run:
        console_print(f"{INFO}The operation YAML is syntactically valid.", stderr=True)

    if (
        not parameters.dry_run
        and op_resource_configuration.actuatorConfigurationIdentifiers
    ):
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
    """Apply ``--with`` resource overlays to raw *operation_data*.

    Args:
        operation_data: Raw operation resource configuration dict.
        parameters: CLI parameters carrying the ``--with`` values.
    """
    if CoreResourceKinds.ACTUATORCONFIGURATION in parameters.with_resources:
        if isinstance(
            parameters.with_resources[CoreResourceKinds.ACTUATORCONFIGURATION], str
        ):
            operation_data["actuatorConfigurationIdentifiers"] = [
                parameters.with_resources[CoreResourceKinds.ACTUATORCONFIGURATION]
            ]
        else:
            actuator_configuration_identifier = create_actuator_configuration(
                AdoCreateCommandParameters(
                    ado_configuration=parameters.ado_configuration,
                    dry_run=parameters.dry_run,
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
            # A dry-run of a new actuator configuration returns no identifier.
            if actuator_configuration_identifier is not None:
                operation_data["actuatorConfigurationIdentifiers"] = [
                    actuator_configuration_identifier
                ]

    if CoreResourceKinds.DISCOVERYSPACE in parameters.with_resources:
        if isinstance(parameters.with_resources[CoreResourceKinds.DISCOVERYSPACE], str):
            space_id = parameters.with_resources[CoreResourceKinds.DISCOVERYSPACE]
        else:
            space_id = create_discovery_space(
                AdoCreateCommandParameters(
                    ado_configuration=parameters.ado_configuration,
                    dry_run=parameters.dry_run,
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
        # A dry-run of a new discovery space returns no identifier.
        if space_id is not None:
            _add_operation_input_for_reference(
                operation_data,
                ADOResourceReference(
                    kind=CoreResourceKinds.DISCOVERYSPACE, identifier=space_id
                ),
            )

    if CoreResourceKinds.DATACONTAINER in parameters.with_resources:
        dcr_identifier = parameters.with_resources[CoreResourceKinds.DATACONTAINER]
        if not isinstance(dcr_identifier, str):
            console_print(
                f"{ERROR}--with datacontainer currently supports an existing "
                "identifier only (not a YAML file).",
                stderr=True,
            )
            raise typer.Exit(1)
        _add_operation_input_for_reference(
            operation_data,
            ADOResourceReference(
                kind=CoreResourceKinds.DATACONTAINER, identifier=dcr_identifier
            ),
        )


def _apply_use_latest(
    operation_data: dict,
    parameters: AdoCreateCommandParameters,
) -> None:
    """Apply ``--use-latest`` overlays to raw *operation_data*.

    Args:
        operation_data: Raw operation resource configuration dict.
        parameters: CLI parameters carrying the requested latest kinds.
    """
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
            _add_operation_input_for_reference(
                operation_data,
                ADOResourceReference(kind=resource_kind, identifier=latest_id),
            )
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
