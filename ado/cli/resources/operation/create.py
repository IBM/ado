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
    _REQUIRED_INPUT_PROPERTY_FOR_ORIGINAL_FUNCTION_OPERATORS,
    DiscoveryOperationResourceConfiguration,
    OperatorReference,
)
from ado.core.operation.inputs import OPERATOR_INPUT_KINDS
from ado.core.operation.operation import OperationException, OperationOutput
from ado.core.operation.resource import (
    OperationExitStateEnum,
)
from ado.core.resources import ADOResourcePropertyDescriptor, ADOResourceReference
from ado.modules.operators.errors import OperatorVersionMismatchError

# Kinds create operation accepts from --with and --use-latest.
_OPERATION_FLAG_KINDS = OPERATOR_INPUT_KINDS | {CoreResourceKinds.ACTUATORCONFIGURATION}

# Kinds whose --with value may be a configuration file, with how to create them.
_CREATORS_FOR_KIND = {
    CoreResourceKinds.DISCOVERYSPACE: (
        create_discovery_space,
        AdoCreateSupportedResourceTypes.DISCOVERY_SPACE,
    ),
    CoreResourceKinds.ACTUATORCONFIGURATION: (
        create_actuator_configuration,
        AdoCreateSupportedResourceTypes.ACTUATOR_CONFIGURATION,
    ),
}


def _resource_inputs_for_module(
    module: dict,
) -> tuple[str, tuple[ADOResourcePropertyDescriptor, ...]]:
    """Return an operator's name and the resource inputs it declares.

    Where the inputs come from depends on which keys *module* contains:

    - Operator reference (``operatorName`` and ``operationType``): the inputs
      registered for that operator.
    - Class module (``moduleName`` and ``moduleClass``): the inputs from the
      class's ``operator_metadata()``, or a single ``discoverySpace`` input if
      it declares none.
    - Script module (``name``): a single ``discoverySpace`` input.

    Args:
        module: The raw ``operation.module`` mapping.

    Returns:
        A tuple of the operator's name and its resource-input descriptors.

    Raises:
        ValueError: If *module* matches none of the cases above, has only one
            of ``moduleName`` and ``moduleClass``, or names a class that cannot
            be imported.
    """
    if "operatorName" in module and "operationType" in module:
        from ado.modules.operators.collections import operator_metadata_for_reference

        metadata = operator_metadata_for_reference(
            OperatorReference(
                operatorName=module["operatorName"],
                operationType=module["operationType"],
            )
        )
        return metadata.name, tuple(metadata.required_resource_inputs)

    if "moduleName" in module or "moduleClass" in module:
        import importlib

        module_name = module.get("moduleName")
        module_class = module.get("moduleClass")
        if not module_name or not module_class:
            raise ValueError(
                "operation.module with moduleName/moduleClass must set both fields."
            )
        try:
            operator_class = getattr(importlib.import_module(module_name), module_class)
        except (ImportError, AttributeError) as error:
            raise ValueError(
                f"Cannot load operator class {module_class!r} from module "
                f"{module_name!r}: {error}"
            ) from error
        metadata = operator_class.operator_metadata()
        declared_resource_inputs = tuple(metadata.required_resource_inputs)
        if not declared_resource_inputs:
            declared_resource_inputs = (
                _REQUIRED_INPUT_PROPERTY_FOR_ORIGINAL_FUNCTION_OPERATORS,
            )
        return metadata.name, declared_resource_inputs

    if "name" in module:
        return (
            str(module["name"]),
            (_REQUIRED_INPUT_PROPERTY_FOR_ORIGINAL_FUNCTION_OPERATORS,),
        )

    raise ValueError(
        "Cannot resolve operation input names: operation.module must be an "
        "operator reference, a moduleName/moduleClass configuration, or a script."
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
        assigned_inputs: Names of inputs already targeted by command-line flags.

    Returns:
        The descriptor of the input to bind.

    Raises:
        ValueError: If no input, or more than one input, can be chosen.
    """
    inputs_of_kind = [
        descriptor for descriptor in descriptors if descriptor.kind == kind
    ]
    if not inputs_of_kind:
        raise ValueError(
            f"Operator {operator_name!r} has no resource input of kind {kind.value!r}."
        )

    candidates = [
        descriptor
        for descriptor in inputs_of_kind
        if descriptor.identifier not in assigned_inputs
    ]
    if not candidates:
        names = [descriptor.identifier for descriptor in inputs_of_kind]
        if len(names) == 1:
            raise ValueError(
                f"Input {names[0]!r} of operator {operator_name!r} was already set."
            )
        raise ValueError(
            f"All {kind.value!r} inputs {names} of operator {operator_name!r} "
            "were already set."
        )

    if len(candidates) == 1:
        return candidates[0]

    list_typed_candidates = [
        descriptor for descriptor in candidates if descriptor.isList
    ]
    if len(list_typed_candidates) == 1:
        return list_typed_candidates[0]

    unconfigured_candidates = [
        descriptor
        for descriptor in candidates
        if descriptor.identifier not in present_inputs
    ]
    if len(unconfigured_candidates) == 1:
        return unconfigured_candidates[0]

    names = [descriptor.identifier for descriptor in inputs_of_kind]
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
    by_name = {descriptor.identifier: descriptor for descriptor in descriptors}
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
) -> None:
    """Write *identifier* to the input described by *descriptor*.

    A list input gets *identifier* appended after its existing entries; a single
    mapping already set for it is first wrapped in a list. Other inputs are
    replaced. Setting a discovery space input removes the legacy ``spaces``
    field.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict;
            updated in place.
        descriptor: Descriptor of the input to set.
        identifier: Resource identifier to store.
    """
    name = descriptor.identifier
    payload = ADOResourceReference(
        identifier=identifier, kind=descriptor.kind
    ).model_dump(mode="json")
    inputs = raw_operation_configuration.setdefault("inputs", {})
    if descriptor.isList:
        existing_entries = inputs.get(name)
        if existing_entries is None:
            existing_entries = []
        elif not isinstance(existing_entries, list):
            existing_entries = [existing_entries]
        inputs[name] = [*existing_entries, payload]
    else:
        inputs[name] = payload
    if descriptor.kind == CoreResourceKinds.DISCOVERYSPACE:
        raw_operation_configuration.pop("spaces", None)


def create_operation(parameters: AdoCreateCommandParameters) -> str | None:
    """Create and run an operation from a configuration file.

    Loads the operation YAML, applies ``--with``, ``--use-latest`` and
    ``--set`` values, and validates the result. In a dry run it stops after
    validation; otherwise it validates actuator configurations and runs the
    operation.

    Args:
        parameters: CLI parameters of the operation being created.

    Returns:
        The output of ``output_operation_result`` for the finished operation,
        or ``None`` for a dry run.

    Raises:
        typer.Exit: If the configuration or a flag is invalid (code 1), the
            operation fails (code 1), or it is interrupted (code 3).
    """
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
        apply_resource_flags(raw_operation_configuration, parameters)
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
    kinds = set(parameters.use_latest or [])
    kinds.update(reference.kind for reference in parameters.use_latest_inputs)
    if not kinds:
        return {}

    from ado.cli.utils.generic.wrappers import get_sql_store

    sql_store = get_sql_store(parameters.ado_configuration.project_context)
    latest_ids = sql_store.get_latest_resource_identifiers_of_kinds(kinds=list(kinds))
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
        kind: Resource kind of the value; a key of ``_CREATORS_FOR_KIND`` when
            *value* is a path.
        value: A resource identifier or a path to a resource configuration file.
        parameters: CLI parameters of the operation being created.

    Returns:
        The resource identifier, or ``None`` for a resource validated in a
        dry run.
    """
    if isinstance(value, str):
        return value

    creator, resource_type = _CREATORS_FOR_KIND[kind]
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


def _resolve_input_bindings(
    raw_operation_configuration: dict,
    parameters: AdoCreateCommandParameters,
) -> list[tuple[ADOResourcePropertyDescriptor, pathlib.Path | str | None]]:
    """Resolve the operation input targeted by each ``--with`` and ``--use-latest`` value.

    Inputs named with ``KIND:INPUT`` are resolved first, then unqualified
    kinds. An unqualified kind binds to the single input of that kind not
    already targeted by a flag.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict.
        parameters: CLI parameters carrying the ``--with`` and ``--use-latest``
            values.

    Returns:
        ``(descriptor, value)`` pairs in the order they are applied, where
        *value* is the ``--with`` value, or ``None`` to use the latest
        identifier of the descriptor's kind.

    Raises:
        ValueError: If ``operation.module`` cannot be resolved to an operator,
            an input name or kind is invalid, a scalar input is targeted twice,
            or an unqualified kind cannot be bound to a single input.
    """
    with_resources = parameters.with_resources or {}
    input_kinds = [
        kind
        for kind in (*with_resources, *(parameters.use_latest or []))
        if kind in OPERATOR_INPUT_KINDS
    ]
    named_inputs = [
        *((option, option.value) for option in parameters.with_inputs),
        *((reference, None) for reference in parameters.use_latest_inputs),
    ]
    if not (named_inputs or input_kinds):
        return []

    operation = raw_operation_configuration.get("operation")
    module = operation.get("module") if isinstance(operation, dict) else None
    if not isinstance(module, dict):
        raise ValueError(
            "operation.module must be a mapping to resolve operation inputs."
        )
    operator_name, descriptors = _resource_inputs_for_module(module)
    present_inputs = dict(raw_operation_configuration.get("inputs") or {})
    targeted_inputs: set[str] = set()
    bindings: list[tuple[ADOResourcePropertyDescriptor, pathlib.Path | str | None]] = []

    for reference, value in named_inputs:
        descriptor = _descriptor_for_name(
            operator_name, descriptors, reference.input_name, reference.kind
        )
        if descriptor.identifier in targeted_inputs and not descriptor.isList:
            raise ValueError(f"Input {descriptor.identifier!r} was set more than once.")
        targeted_inputs.add(descriptor.identifier)
        bindings.append((descriptor, value))

    for kind in input_kinds:
        descriptor = _descriptor_for_kind(
            operator_name, descriptors, kind, present_inputs, targeted_inputs
        )
        targeted_inputs.add(descriptor.identifier)
        bindings.append((descriptor, with_resources.get(kind)))

    return bindings


def apply_resource_flags(
    raw_operation_configuration: dict,
    parameters: AdoCreateCommandParameters,
) -> None:
    """Apply ``--with`` and ``--use-latest`` values to a raw operation configuration.

    All values are validated, and the input each one targets is resolved,
    before any resource is looked up or created from a ``--with`` file. The
    actuator configuration is then applied, followed by inputs named with
    ``KIND:INPUT`` and then unqualified kinds. Values for a list input are
    appended to the entries already in the configuration.

    Args:
        raw_operation_configuration: Raw operation resource configuration dict;
            updated in place.
        parameters: CLI parameters carrying the ``--with`` and ``--use-latest``
            values.

    Raises:
        ValueError: If a kind cannot be used by operations, an unqualified kind
            is given by both flags, or an input cannot be resolved (see
            ``_resolve_input_bindings``).
        typer.Exit: If a ``--with`` file is given for a kind that cannot be
            created from a file, or a requested latest identifier cannot be
            found.
    """
    with_resources = parameters.with_resources or {}
    use_latest = parameters.use_latest or []

    unsupported = {
        kind
        for kind in (*with_resources, *use_latest)
        if kind not in _OPERATION_FLAG_KINDS
    }
    if unsupported:
        raise ValueError(
            f"{', '.join(sorted(kind.value for kind in unsupported))} cannot be "
            "used when creating an operation. Supported kinds are "
            f"{', '.join(sorted(kind.value for kind in _OPERATION_FLAG_KINDS))}."
        )

    duplicated = set(with_resources).intersection(use_latest)
    if duplicated:
        names = ", ".join(sorted(kind.value for kind in duplicated))
        raise ValueError(f"{names} given with both --with and --use-latest.")

    with_values = [
        *with_resources.items(),
        *((option.kind, option.value) for option in parameters.with_inputs),
    ]
    for kind, value in with_values:
        if isinstance(value, pathlib.Path) and kind not in _CREATORS_FOR_KIND:
            console_print(
                f"{ERROR}--with {kind.value} accepts an existing identifier only, "
                "not a configuration file.",
                stderr=True,
            )
            raise typer.Exit(1)

    bindings = _resolve_input_bindings(raw_operation_configuration, parameters)
    latest_identifiers = _latest_identifiers(parameters)

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

    for descriptor, value in bindings:
        if value is None:
            identifier = latest_identifiers[descriptor.kind]
            _print_latest_used(descriptor.kind, identifier, descriptor.identifier)
        else:
            identifier = _identifier_for_with_value(descriptor.kind, value, parameters)
        if identifier is not None:
            _set_input(raw_operation_configuration, descriptor, identifier)


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
