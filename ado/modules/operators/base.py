# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Defines the interfaces to the operations that can be performed on DiscoverySpaces"""

import abc
import contextlib
import logging
import typing
from collections.abc import Callable
from typing import TypeAlias

import ray
import ray.exceptions

import ado.core.operation.resource
import ado.core.resources
import ado.metastore.project
import ado.modules
import ado.modules.actuators.replay
import ado.schema.reference
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    FunctionOperationInfo,
    OperatorModuleConf,
    OperatorReference,
)
from ado.core.operation.inputs import OPERATOR_INPUT_TYPE_FOR_KIND
from ado.core.operation.operation import OperationOutput
from ado.core.operation.resource import OperationResource
from ado.core.resources import ADOResourcePropertyDescriptor, ADOResourceReference
from ado.metastore.sqlstore import SQLStore
from ado.modules.actuators.measurement_queue import MeasurementQueue
from ado.modules.operators.discovery_space_manager import (
    DiscoverySpaceUpdateSubscriber,
)
from ado.schema.entity import Entity
from ado.schema.reference import ExperimentReference

if typing.TYPE_CHECKING:
    from ado.core.resources import ADOResource
    from ado.metastore.base import ResourceStore
    from ado.modules.actuators.base import ActuatorBase

    from .discovery_space_manager import DiscoverySpaceManagerActor

moduleLog = logging.getLogger("operation_base")


#: Callable that runs an operator after registration (nested calls / CLI).
#:
#: Structural convention::
#:
#:     (resource₁, …, resourceₙ, operationInfo=None, **parameters) -> OperationOutput
#:
#: Resource parameter names and types come from ``required_resource_inputs``;
#: they are not expressible as a single Protocol for arbitrary N.
OperatorCallable: TypeAlias = Callable[..., OperationOutput]


def validate_operator_call_shape(
    fn: typing.Callable,
    required_resource_inputs: list[ADOResourcePropertyDescriptor],
) -> None:
    """Validate that *fn* matches the operator call-shape convention.

    Leading positional-or-keyword parameters must be the resource input
    identifiers (in declaration order), followed by ``operationInfo``, then
    an optional ``**kwargs`` for operation parameters. The return type must
    be :class:`~ado.core.operation.operation.OperationOutput`.

    Resource *types* are checked separately by
    :func:`validate_resource_input_types`.

    Args:
        fn: The callable to validate (user implementation or stored wrapper).
        required_resource_inputs: Declared resource inputs for this operator.

    Raises:
        ValueError: If *fn* does not match the call-shape convention.
    """
    import inspect

    _HINTS_MISSING_HINT = (
        "Operator functions must declare the correct types for all parameters "
        "and the return value to pass validation."
    )

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Cannot validate operator function {fn!r}: signature could not "
            f"be inspected ({exc})."
        ) from exc

    params = list(sig.parameters.values())
    var_keyword = next(
        (p for p in params if p.kind == inspect.Parameter.VAR_KEYWORD),
        None,
    )
    var_positional = next(
        (p for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL),
        None,
    )
    if var_positional is not None:
        raise ValueError(
            "Operator function must not declare *args; resource inputs must be "
            "explicit named parameters."
        )

    positional = [
        p
        for p in params
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ]

    expected_names = [d.identifier for d in required_resource_inputs] + [
        "operationInfo"
    ]
    actual_names = [p.name for p in positional]

    if actual_names != expected_names:
        raise ValueError(
            "Operator function positional parameters must be the declared "
            f"resource inputs {[d.identifier for d in required_resource_inputs]!r} "
            f"followed by 'operationInfo'; got {actual_names!r}."
        )

    try:
        hints = typing.get_type_hints(fn)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Operator function has valid structure but type hints are missing "
            f"or unresolvable ({exc}). {_HINTS_MISSING_HINT}"
        ) from exc

    missing_hints: list[str] = [
        f"parameter {name!r}" for name in expected_names if name not in hints
    ]
    if "return" not in hints:
        missing_hints.append("return type")
    if missing_hints:
        raise ValueError(
            f"Operator function has valid structure but type hints are missing "
            f"for {missing_hints}. {_HINTS_MISSING_HINT}"
        )

    if hints["return"] != OperationOutput:
        raise ValueError(
            f"Operator function return type must be {OperationOutput!r}, "
            f"got {hints['return']!r}."
        )

    operation_info_type = hints["operationInfo"]
    expected_operation_info = FunctionOperationInfo | None
    if operation_info_type != expected_operation_info:
        raise ValueError(
            f"Operator function parameter 'operationInfo' must be "
            f"{expected_operation_info!r}, got {operation_info_type!r}."
        )

    if var_keyword is None:
        # **kwargs is optional
        pass


def validate_resource_input_types(
    fn: typing.Callable,
    required_resource_inputs: list[ADOResourcePropertyDescriptor] | None,
) -> None:
    """Validate that *fn*'s parameter type hints match its declared resource inputs.

    For each descriptor in *required_resource_inputs*, if *fn* declares an
    explicit, resolvable type hint for a parameter named
    ``descriptor.identifier``, that hint must equal the Python type ADO
    resolves for ``descriptor.kind`` (see
    :data:`ado.core.operation.inputs.OPERATOR_INPUT_TYPE_FOR_KIND`). A
    descriptor whose identifier is not an explicit parameter of *fn* (e.g.
    it is absorbed by a ``**kwargs`` catch-all) cannot be statically
    checked and is skipped.

    Args:
        fn: The operator callable to validate.
        required_resource_inputs: The resource inputs declared for *fn*.

    Raises:
        ValueError: If an explicitly-annotated parameter's type does not
            match the type expected for its declared resource kind.
    """
    if not required_resource_inputs:
        return

    try:
        hints = typing.get_type_hints(fn)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Cannot validate resource input types for operator function {fn!r}: "
            f"type hints are missing or unresolvable ({exc})."
        ) from exc

    for descriptor in required_resource_inputs:
        if descriptor.identifier not in hints:
            continue
        expected_type = OPERATOR_INPUT_TYPE_FOR_KIND[descriptor.kind]
        actual_type = hints[descriptor.identifier]
        if actual_type != expected_type:
            raise ValueError(
                f"Operator function parameter {descriptor.identifier!r} must be "
                f"{expected_type!r} to match declared resource kind "
                f"{descriptor.kind.value!r}, got {actual_type!r}."
            )


def validate_operator_registration(
    user_fn: typing.Callable,
    required_resource_inputs: list[ADOResourcePropertyDescriptor],
    operation_type: DiscoveryOperationEnum,
    stored_fn: typing.Callable | None = None,
) -> None:
    """Run all registration-time checks for an operator callable.

    Entry point used by operator decorators. Runs:

    1. :func:`~ado.modules.operators.inputs.validate_resource_inputs_for_operation_type`
       — declaration vs collection policy
    2. :func:`validate_resource_input_types` — resource name ↔ rich type
    3. :func:`validate_operator_call_shape` — resources then ``operationInfo``
       then parameters

    Args:
        user_fn: The operator implementation (before or under ``functools.wraps``).
        required_resource_inputs: Declared resource inputs.
        operation_type: Collection the operator is registered into.
        stored_fn: Optional stored callable (wrapper). When provided and distinct
            from *user_fn* under unwrap, its call shape is also validated
            (e.g. explore's generated function).

    Raises:
        ValueError: If any validation check fails.
    """
    import inspect

    from ado.modules.operators.inputs import (
        validate_resource_inputs_for_operation_type,
    )

    validate_resource_inputs_for_operation_type(
        required_resource_inputs, operation_type
    )
    validate_resource_input_types(user_fn, required_resource_inputs)
    validate_operator_call_shape(user_fn, required_resource_inputs)

    if stored_fn is not None and inspect.unwrap(stored_fn) is not user_fn:
        validate_operator_call_shape(stored_fn, required_resource_inputs)


# Some operations are RayActors: These operations use Actuators and StateUpdateQueue and require Ray
# Some operations are not RayActors: They don't have to use Actuators and StateUpdateQueue or Ray. They can use ray-workers


class DiscoveryOperationBase(metaclass=abc.ABCMeta):
    def operationIdentifier(self) -> str:
        """A unique id for the operation instance being run by the operator.

        Returns ``"<classname>-<8-hex-uuid>"``, stable for the lifetime of
        the instance.  Override when a deterministic or richer identifier is
        needed (e.g. embedding the operator version or a caller-supplied run
        id).
        """
        import uuid

        return f"{type(self).__name__.lower()}-{uuid.uuid4().hex[:8]}"

    @classmethod
    @abc.abstractmethod
    def operator_metadata(
        cls,
    ) -> "ado.core.operation.config.OperatorMetadata":
        """Return :class:`~ado.core.operation.config.OperatorMetadata` for this operator.

        Subclasses must override this method and return an
        :class:`~ado.core.operation.config.OperatorMetadata` instance
        that describes the operator's name, version, type, and configuration
        model.  The ``@explore_operation`` decorator fills in the ``function``
        and ``cls`` fields before registering::

            @classmethod
            def operator_metadata(cls) -> OperatorMetadata:
                return OperatorMetadata(
                    name="my_op",
                    version="0.1.0",
                    configuration_model=MyOpParameters,
                    example_configuration=MyOpParameters(),
                    type=DiscoveryOperationEnum.EXPLORE,
                )
        """
        raise NotImplementedError(f"{cls.__name__} must implement operator_metadata().")


class UnaryDiscoveryOperation(metaclass=abc.ABCMeta):
    """A discovery operation that processes a single space"""

    # async def run(self, discoveryState):
    #     pass

    @abc.abstractmethod
    async def run(self) -> OperationOutput | None:
        pass


class MultivariateDiscoveryOperation(metaclass=abc.ABCMeta):
    """A discovery operation that processes  multiple spaces"""

    # async def run(self, *discoveryStates):
    #     pass

    @abc.abstractmethod
    async def run(self) -> OperationOutput | None:
        pass


class DiscoverySpaceSubscribingDiscoveryOperation(
    DiscoveryOperationBase,
    DiscoverySpaceUpdateSubscriber,
    metaclass=abc.ABCMeta,
):
    """Instances of this class can receive notifications when measurements are added to a DiscoverySpace"""

    def __init__(
        self,
        operationActorName: str,
        namespace: str | None,
        discovery_space_manager: "DiscoverySpaceManagerActor",
        actuators: dict[str, "ado.modules.actuators.base.ActuatorBase"],
    ) -> None:
        self.actorName = operationActorName
        self.namespace = namespace
        self.ds_manager = discovery_space_manager
        self.actuators = actuators
        # noinspection PyUnresolvedReferences
        self.ds_manager.subscribeToUpdates.remote(subscriberName=self.actorName)

        super().__init__()


class Explore(
    DiscoverySpaceSubscribingDiscoveryOperation,
    UnaryDiscoveryOperation,
    metaclass=abc.ABCMeta,
):
    """Subclasses sample entities from a DiscoverySpace and run measurements on them

    The general pattern is that on calling run() the subclass
    1. Samples a set of entities from the discovery space
    2. Submits them for measurement using measure_or_replay()
    3. Waits for notifications that measurements are completed
    4. Returns to 1 or finishes

    Subclasses define different sampling strategies and submission strategies
    e.g. wait for all measurements to complete before sending new batch or
    submit new measurements as soon as one completes.
    """


def measure_or_replay(
    requestIndex: int,
    requesterid: str,
    entities: list[Entity],
    experimentReference: ExperimentReference,
    actuators: dict[str, "ActuatorBase"],
    measurement_queue: MeasurementQueue,
    memoize: bool,
) -> list[str]:
    """Checks if entities have been measured by experimentReference before and takes appropriate action

    If an entity has been measured by referenced experiment and memoize is True:
        The existing measurement results for the experiment on that entity are reused i.e. the experiment is not performed again
    If an entity has not been measured by the referenced experiment and/or memoize is False:
        The entity is submitted to the relevant actuator for measurement

    Params:
        requestIndex: The index of this request
        requesterid: The id of the requester
        entities: A list entities to be measured
        experimentReference: Indicates the experiment to perform
        actuators: Dictionary of actuators containing one that can execute experimentReference
        measurement_queue: Used to replay memoized results
        memoize: If True and the experiment has already been applied to the entity
             existing results are reused i.e. the experiment is not performed again.

    Returns:
        The identifiers of the MeasurementRequests created to perform (or reuse) the measurements

    Raises:
        KeyError: If there is no actuator in actuators that can executed experimentReference
        MeasurementError: If the experimentReference cannot be executed by the actuator as it is
            deprecated w.r.t the actuator version being used.
    """
    from ado.modules.actuators.errors import (
        DeprecatedExperimentError,
        MeasurementError,
        MissingConfigurationForExperimentError,
    )

    moduleLog.debug(
        f"Checking application of {experimentReference} to {[e.identifier for e in entities]}. Memoize is: {memoize}"
    )

    request_ids: list[str] = []
    if memoize:
        # This will only memoize entities that can be memoized
        replayed_requests = ado.modules.actuators.replay.replay(
            requesterid=requesterid,
            requestIndex=requestIndex,
            entities=entities,
            experiment_reference=experimentReference,
            measurement_queue=measurement_queue,
        )
        request_ids.extend([r.requestid for r in replayed_requests])

        memoized_entity_ids = {
            e.identifier for r in replayed_requests for e in r.entities
        }

        moduleLog.debug(f"The following entities were memoized: {memoized_entity_ids}")
    else:
        memoized_entity_ids = {}

    entities_to_be_measured = [
        e for e in entities if e.identifier not in memoized_entity_ids
    ]

    if len(entities_to_be_measured) > 0:
        actuator = actuators[experimentReference.actuatorIdentifier]
        try:
            # noinspection PyUnresolvedReferences
            measurement_request_ids = ray.get(
                actuator.submit.remote(
                    entities=entities_to_be_measured,
                    experimentReference=experimentReference,
                    requesterid=requesterid,
                    requestIndex=requestIndex,
                )
            )
        # All exceptions coming from Ray are wrapped in a RayTaskError
        # We access the cause and
        # - If it is one of the exception types we support we raise a MeasurementError wrapping it
        # - If it is not we raise the RayTaskError
        except ray.exceptions.RayTaskError as e:
            if isinstance(
                e.cause,
                DeprecatedExperimentError | MissingConfigurationForExperimentError,
            ):
                raise MeasurementError(
                    f"Cannot apply experiment {experimentReference}. Reason: {e.cause}"
                ) from e.cause
            raise

        request_ids.extend(measurement_request_ids)

    return request_ids


def add_operation_output_to_metastore(
    operation: "OperationResource",
    output: "OperationOutput",
    metastore: "ResourceStore",
) -> None:

    if output:
        resource: ADOResource
        for resource in output.resources:
            try:
                metastore.addResourceWithRelationships(
                    resource, relatedIdentifiers=[operation.identifier]
                )
            except ValueError:  # noqa: PERF203
                # Assume already added
                metastore.addRelationship(operation.identifier, resource.identifier)

        if output.metadata:
            # Add the OperationOutput metadata to the OperationResource
            operation.metadata.update(output.metadata)
            metastore.updateResource(operation)


def add_operation_and_output_to_metastore(
    operation_resource_configuration: DiscoveryOperationResourceConfiguration,
    output: OperationOutput,
    metastore: SQLStore,
) -> OperationResource:
    """Creates an operation resource from the given configuration and adds it and its outputs to the resource store"""
    from ado.core.operation.resource import OperationProvenanceInfo
    from ado.modules.operators.collections import provenance_for_operator

    operator_module = operation_resource_configuration.operation.module
    operators = {}
    if isinstance(operator_module, OperatorReference):
        operator_provenance = provenance_for_operator(
            operator_module.operatorName, operator_module.operationType
        )
        if operator_provenance is not None:
            operators[operator_module.operatorIdentifier] = operator_provenance

    operation = OperationResource(
        operationType=operator_module.operationType,
        operatorIdentifier=operator_module.operatorIdentifier,
        config=operation_resource_configuration,
        status=[output.exitStatus],
        provenance=OperationProvenanceInfo(operators=operators),
    )

    # ValueError means the resource has already been added
    with contextlib.suppress(ValueError):
        metastore.addResourceWithRelationships(
            resource=operation,
            relatedIdentifiers=[
                ref.identifier
                for ref in operation_resource_configuration.inputs.values()
            ],
        )

    add_operation_output_to_metastore(operation, output, metastore)

    return operation


def create_operation_and_add_to_metastore(
    inputs: dict[str, ADOResourceReference],
    operator_module: OperatorModuleConf | OperatorReference,
    operation_parameters: dict,
    operation_info: FunctionOperationInfo,
    metastore: SQLStore,
    operation_identifier: str | None = None,
) -> OperationResource:
    """Creates an operation resource and adds it to the metastore.

    This function creates an OperationResource from the provided operator module,
    parameters, and operation info, then adds it to the metastore with relationships
    to the associated ado resources and any actuator configurations.

    Args:
        inputs: A dictionary mapping the ado resources the operation worked on to
            the operators input parameters.
        operator_module: Configuration for the operator (module or function-based).
        operation_parameters: Dictionary of parameters for the operation.
        operation_info: Information about the operation including metadata and actuator
            configuration identifiers.
        metastore: The SQL store to add the operation resource to.

        operation_identifier: Optional pre-existing identifier for the operation resource.
            If not provided, a new identifier will be generated.

    Returns:
        OperationResource instance that was created and added to the metastore.
    """
    from ado.core.operation.config import DiscoveryOperationConfiguration
    from ado.core.operation.resource import OperationProvenanceInfo
    from ado.core.resources import CoreResourceKinds
    from ado.modules.operators.collections import provenance_for_operator

    # Derive space_ids for metastore relationship linking.
    space_ids = [
        e.identifier
        for e in inputs.values()
        if e.kind == CoreResourceKinds.DISCOVERYSPACE
    ]

    operation_resource_configuration = DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=operator_module,
            parameters=operation_parameters,
        ),
        metadata=operation_info.metadata,
        actuatorConfigurationIdentifiers=operation_info.actuatorConfigurationIdentifiers,
        inputs=inputs,
    )

    op_module = operation_resource_configuration.operation.module
    operators = {}
    if isinstance(op_module, OperatorReference):
        operator_provenance = provenance_for_operator(
            op_module.operatorName, op_module.operationType
        )
        if operator_provenance is not None:
            operators[op_module.operatorIdentifier] = operator_provenance

    operation = OperationResource(
        identifier=operation_identifier,
        operationType=op_module.operationType,
        operatorIdentifier=op_module.operatorIdentifier,
        config=operation_resource_configuration,
        provenance=OperationProvenanceInfo(operators=operators),
    )

    related_identifiers = [
        *space_ids,
        *operation_resource_configuration.actuatorConfigurationIdentifiers,
    ]

    # ValueError means the resource has already been added
    with contextlib.suppress(ValueError):
        metastore.addResourceWithRelationships(
            resource=operation,
            relatedIdentifiers=related_identifiers,
        )

    return operation


def warn_deprecated_operator_parameters_model_in_use(
    affected_operator: str,
    deprecated_from_operator_version: str,
    removed_from_operator_version: str,
    deprecated_fields: str | list[str] | None = None,
    latest_format_documentation_url: str | None = None,
) -> None:
    from rich.console import Console

    resource_name = "operation"
    doc_url = (
        f": {latest_format_documentation_url}"
        if latest_format_documentation_url
        else ""
    )

    if deprecated_fields:
        fields_causing_issues = f"fields [b magenta]{deprecated_fields}[/b magenta]"
        if isinstance(deprecated_fields, str):
            fields_causing_issues = f"field [b magenta]{deprecated_fields}[/b magenta]"
        elif isinstance(deprecated_fields, list) and len(deprecated_fields) == 1:
            fields_causing_issues = (
                f"field [b magenta]{deprecated_fields[0]}[/b magenta]"
            )

        warning_preamble = (
            f"The use of {fields_causing_issues} in the parameters of the {affected_operator} operator "
            f"is deprecated as of {affected_operator} [b cyan]{deprecated_from_operator_version}[/b cyan]."
        )
    else:
        warning_preamble = (
            f"The parameters for the {affected_operator} operator have been updated "
            f"as of {affected_operator} [b cyan]{deprecated_from_operator_version}[/b cyan]."
        )

    autoupgrade_notice = (
        "They are being temporarily auto-upgraded to the latest version."
    )
    autoupgrade_removal_warning = (
        f"[b]This behavior will be removed with {affected_operator} "
        f"[b cyan]{removed_from_operator_version}[/b cyan][/b]."
    )
    manual_upgrade_hint = (
        f"Run [b cyan]ado upgrade {resource_name}s[/b cyan] to upgrade the stored {resource_name}s.\n\t"
        f"Update your {resource_name} YAML files to use the latest format{doc_url}."
    )

    Console(stderr=True).print(
        f"[b yellow]WARN[/b yellow]:\t{warning_preamble}\n\t"
        f"{autoupgrade_notice}\n\t{autoupgrade_removal_warning}\n"
        f"[b magenta]HINT[/b magenta]:\t{manual_upgrade_hint}",
        overflow="ignore",
        crop=False,
    )


class InterruptedOperationError(KeyboardInterrupt):
    """Exception raised when an operation is interrupted (e.g., by SIGINT)

    This exception inherits from KeyboardInterrupt and includes the operation identifier
    to provide context about which operation was interrupted.
    """

    def __init__(
        self, operation_identifier: str, resources: list["ADOResource"] | None = None
    ) -> None:
        self.operation_identifier = operation_identifier
        self.resources = resources or []
        super().__init__(f"Operation {operation_identifier} was interrupted")


if typing.TYPE_CHECKING:
    from ray.actor import ActorHandle

    OperatorActor = type[ActorHandle[DiscoverySpaceSubscribingDiscoveryOperation]]
