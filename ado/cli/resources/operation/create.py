# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib

import pydantic
import typer
import yaml

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]

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
    cyan,
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
from ado.core.operation.inputs import KIND_FOR_OPERATOR_INPUT_TYPE
from ado.core.operation.operation import OperationException, OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
)
from ado.core.resources import ADOResourcePropertyDescriptor
from ado.modules.operators.errors import OperatorVersionMismatchError

_INPUT_KINDS = frozenset(KIND_FOR_OPERATOR_INPUT_TYPE.values())


def _operator_resource_inputs(
    raw_operation_configuration: dict,
) -> tuple[str, tuple[ADOResourcePropertyDescriptor, ...]]:
    """Return the operator name and resource-input descriptors for an operation.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict.

    Returns:
        The operator name and its resource-input descriptors. Operators that
        record no resource inputs have a single ``discoverySpace`` input.

    Raises:
        ValueError: If ``operation.module`` does not identify an operator.
    """
    import importlib

    from ado.core.operation.config import (
        _REQUIRED_INPUT_PROPERTY_FOR_ORIGINAL_FUNCTION_OPERATORS,
    )
    from ado.modules.operators.collections import operator_metadata_for_reference

    module = raw_operation_configuration.get("operation", {}).get("module", {})
    if "operatorName" in module and "operationType" in module:
        metadata = operator_metadata_for_reference(
            OperatorReference(
                operatorName=module["operatorName"],
                operationType=module["operationType"],
            )
        )
    elif "moduleClass" in module and "moduleName" in module:
        metadata = getattr(
            importlib.import_module(module["moduleName"]), module["moduleClass"]
        ).operator_metadata()
    else:
        raise ValueError(
            "Cannot resolve operation inputs: operation.module must set "
            "operatorName/operationType or moduleName/moduleClass."
        )

    return metadata.name, metadata.required_resource_inputs or (
        _REQUIRED_INPUT_PROPERTY_FOR_ORIGINAL_FUNCTION_OPERATORS,
    )


def _descriptor_for_kind(
    operator_name: str,
    descriptors: tuple[ADOResourcePropertyDescriptor, ...],
    kind: CoreResourceKinds,
    present_inputs: dict,
    assigned_inputs: set[str],
) -> ADOResourcePropertyDescriptor:
    """Deduce the input to bind for an unqualified resource *kind*.

    Inputs in *assigned_inputs* are excluded. Of the remaining inputs of
    *kind*, returns the only one, else the only list input, else the only one
    absent from *present_inputs*.

    Args:
        operator_name: Operator name used in error messages.
        descriptors: The operator's resource-input descriptors.
        kind: Resource kind to bind.
        present_inputs: Inputs already set in the operation configuration.
        assigned_inputs: Names of inputs already set from command-line flags.

    Returns:
        The descriptor of the input to bind.

    Raises:
        ValueError: If no input, or more than one input, can be chosen.
    """
    inputs_of_kind = [d for d in descriptors if d.kind == kind]
    if not inputs_of_kind:
        raise ValueError(
            f"Operator {operator_name!r} has no resource input of kind {kind.value!r}."
        )

    candidates = [d for d in inputs_of_kind if d.identifier not in assigned_inputs]
    for narrowed in (
        candidates,
        [d for d in candidates if d.isList],
        [d for d in candidates if d.identifier not in present_inputs],
    ):
        if len(narrowed) == 1:
            return narrowed[0]

    names = [d.identifier for d in inputs_of_kind]
    raise ValueError(
        f"Operator {operator_name!r} has multiple {kind.value!r} inputs {names}. "
        f"Use {kind.value}:<input> to choose one, e.g. {kind.value}:{names[0]}."
    )


def _descriptor_for_name(
    operator_name: str,
    descriptors: tuple[ADOResourcePropertyDescriptor, ...],
    input_name: str,
    kind: CoreResourceKinds,
) -> ADOResourcePropertyDescriptor:
    """Return the descriptor for the input *input_name* and check its kind.

    Args:
        operator_name: Operator name used in error messages.
        descriptors: The operator's resource-input descriptors.
        input_name: Name of the operator input.
        kind: Resource kind given for the input.

    Returns:
        The descriptor named *input_name*.

    Raises:
        ValueError: If the operator has no such input, or its kind differs.
    """
    by_name = {d.identifier: d for d in descriptors}
    descriptor = by_name.get(input_name)
    if descriptor is None:
        raise ValueError(
            f"Operator {operator_name!r} has no input named {input_name!r}. "
            f"Its resource inputs are {list(by_name)}."
        )
    if descriptor.kind != kind:
        raise ValueError(
            f"Input {input_name!r} of operator {operator_name!r} has kind "
            f"{descriptor.kind.value!r}, not {kind.value!r}."
        )
    return descriptor


def _set_input(
    raw_operation_configuration: dict,
    descriptor: ADOResourcePropertyDescriptor,
    identifier: str,
    assigned_inputs: set[str],
) -> None:
    """Write *identifier* to the input described by *descriptor*.

    The first assignment to an input replaces any value from the configuration
    file. Later assignments append for list inputs.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict.
        descriptor: Descriptor of the input to set.
        identifier: Resource identifier to store.
        assigned_inputs: Names of inputs already set from flags; updated in place.

    Raises:
        ValueError: If a scalar input is assigned more than once.
    """
    name = descriptor.identifier
    payload = {"identifier": identifier, "kind": descriptor.kind.value}
    inputs = raw_operation_configuration.setdefault("inputs", {})
    if descriptor.isList:
        if name in assigned_inputs:
            inputs[name].append(payload)
        else:
            inputs[name] = [payload]
    else:
        if name in assigned_inputs:
            raise ValueError(f"Input {name!r} was set more than once.")
        inputs[name] = payload
    assigned_inputs.add(name)
    if descriptor.kind == CoreResourceKinds.DISCOVERYSPACE:
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
        apply_resource_flags(
            raw_operation_configuration,
            parameters,
            latest_identifiers=_latest_identifiers(parameters),
        )
    except ValueError as e:
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


def _latest_identifiers(
    parameters: AdoCreateCommandParameters,
) -> dict[CoreResourceKinds, str]:
    """Look up the latest identifier of each kind requested by ``--use-latest``.

    Args:
        parameters: CLI parameters carrying the ``--use-latest`` values.

    Returns:
        Mapping of each requested kind to its latest identifier.

    Raises:
        typer.Exit: If a requested kind has no resources.
    """
    kinds = list(
        dict.fromkeys(
            [
                *(parameters.use_latest or []),
                *(kind for _, kind in parameters.useLatestInputs),
            ]
        )
    )
    if not kinds:
        return {}

    from ado.cli.utils.generic.wrappers import get_sql_store

    sql_store = get_sql_store(parameters.ado_configuration.project_context)
    latest_ids = sql_store.get_latest_resource_identifiers_of_kinds(kinds=kinds)
    for kind in kinds:
        if not latest_ids.get(kind):
            console_print(latest_identifier_for_resource_not_found(kind), stderr=True)
            raise typer.Exit(1)
    return latest_ids


def _identifier_for_with_value(
    kind: CoreResourceKinds,
    value: pathlib.Path | str,
    parameters: AdoCreateCommandParameters,
) -> str | None:
    """Return the identifier for a ``--with`` value, creating the resource if needed.

    Args:
        kind: Resource kind of the value.
        value: A resource identifier or a path to a resource configuration file.
        parameters: CLI parameters of the operation being created.

    Returns:
        The resource identifier, or ``None`` for a resource validated in a
        dry run.

    Raises:
        ValueError: If *value* is a file for a kind that cannot be created here.
    """
    if isinstance(value, str):
        return value

    creators = {
        CoreResourceKinds.DISCOVERYSPACE: (
            create_discovery_space,
            AdoCreateSupportedResourceTypes.DISCOVERY_SPACE,
        ),
        CoreResourceKinds.ACTUATORCONFIGURATION: (
            create_actuator_configuration,
            AdoCreateSupportedResourceTypes.ACTUATOR_CONFIGURATION,
        ),
    }
    if kind not in creators:
        raise ValueError(
            f"--with {kind.value} accepts an existing identifier only, "
            "not a configuration file."
        )
    creator, resource_type = creators[kind]
    return creator(
        AdoCreateCommandParameters(
            ado_configuration=parameters.ado_configuration,
            dry_run=parameters.dry_run,
            new_sample_store=False,
            override_values=[],
            resource_configuration_file=value,
            resource_type=resource_type,
            use_default_sample_store=False,
            with_resources={},
            use_latest=[],
        )
    )


def apply_resource_flags(
    raw_operation_configuration: dict,
    parameters: AdoCreateCommandParameters,
    latest_identifiers: dict[CoreResourceKinds, str],
) -> None:
    """Apply ``--with`` and ``--use-latest`` values to a raw operation configuration.

    Actuator configurations are applied first, then inputs named with
    ``KIND:INPUT``, then unqualified kinds. An unqualified kind binds to the
    single input of that kind not already set by a flag.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict;
            updated in place.
        parameters: CLI parameters carrying the ``--with`` and ``--use-latest``
            values.
        latest_identifiers: Latest identifier of each kind requested with
            ``--use-latest``.

    Raises:
        ValueError: If a kind is given by both flags, an input name or kind is
            invalid, a scalar input is set twice, or an unqualified kind cannot
            be bound to a single input.
    """
    with_resources = parameters.with_resources or {}
    use_latest = parameters.use_latest or []

    duplicated = set(with_resources).intersection(use_latest)
    if duplicated:
        names = ", ".join(sorted(kind.value for kind in duplicated))
        raise ValueError(f"{names} given with both --with and --use-latest.")

    actuator_configuration = CoreResourceKinds.ACTUATORCONFIGURATION
    actuator_configuration_identifier = None
    if actuator_configuration in with_resources:
        actuator_configuration_identifier = _identifier_for_with_value(
            actuator_configuration,
            with_resources[actuator_configuration],
            parameters,
        )
    elif actuator_configuration in use_latest:
        actuator_configuration_identifier = latest_identifiers[actuator_configuration]
        _print_latest_used(actuator_configuration, actuator_configuration_identifier)
    if actuator_configuration_identifier is not None:
        raw_operation_configuration["actuatorConfigurationIdentifiers"] = [
            actuator_configuration_identifier
        ]

    input_kinds = [k for k in (*with_resources, *use_latest) if k in _INPUT_KINDS]
    if not (parameters.withInputs or parameters.useLatestInputs or input_kinds):
        return

    operator_name, descriptors = _operator_resource_inputs(raw_operation_configuration)
    present_inputs = dict(raw_operation_configuration.get("inputs") or {})
    assigned_inputs: set[str] = set()

    for option in parameters.withInputs:
        descriptor = _descriptor_for_name(
            operator_name, descriptors, option.inputName, option.kind
        )
        identifier = _identifier_for_with_value(option.kind, option.value, parameters)
        if identifier is not None:
            _set_input(
                raw_operation_configuration, descriptor, identifier, assigned_inputs
            )

    for input_name, kind in parameters.useLatestInputs:
        descriptor = _descriptor_for_name(operator_name, descriptors, input_name, kind)
        _set_input(
            raw_operation_configuration,
            descriptor,
            latest_identifiers[kind],
            assigned_inputs,
        )
        _print_latest_used(kind, latest_identifiers[kind], input_name)

    for kind in input_kinds:
        descriptor = _descriptor_for_kind(
            operator_name, descriptors, kind, present_inputs, assigned_inputs
        )
        if kind in with_resources:
            identifier = _identifier_for_with_value(
                kind, with_resources[kind], parameters
            )
        else:
            identifier = latest_identifiers[kind]
            _print_latest_used(kind, identifier, descriptor.identifier)
        if identifier is not None:
            _set_input(
                raw_operation_configuration, descriptor, identifier, assigned_inputs
            )


def _print_latest_used(
    kind: CoreResourceKinds, identifier: str, input_name: str | None = None
) -> None:
    """Print that the latest identifier of *kind* is being used.

    Args:
        kind: Resource kind reused.
        identifier: Latest identifier of *kind*.
        input_name: Operation input set to *identifier*, if any.
    """
    if input_name is None:
        message = value_in_configuration_replaced_with_latest_identifier_for_resource(
            reused_resource_kind=kind,
            target_resource_kind=CoreResourceKinds.OPERATION,
            replacement_identifier=identifier,
        )
    else:
        message = (
            f"{INFO}Using the latest {kind.value} {cyan(identifier)} "
            f"for operation input {magenta(input_name)}."
        )
    console_print(message, stderr=True)


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
