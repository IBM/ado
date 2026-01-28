# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""Defines the interfaces to the operations that can be performed on DiscoverySpaces"""

import abc
import contextlib
import logging
import typing

import pydantic
import ray
import ray.exceptions

import orchestrator.core.metadata
import orchestrator.core.operation.resource
import orchestrator.core.resources
import orchestrator.metastore.project
import orchestrator.modules
import orchestrator.modules.actuators.replay
import orchestrator.schema.reference
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    FunctionOperationInfo,
    OperatorFunctionConf,
    OperatorModuleConf,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.core.operation.resource import OperationResource
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.operators.discovery_space_manager import (
    DiscoverySpaceManager,
    DiscoverySpaceUpdateSubscriber,
)
from orchestrator.schema.entity import Entity
from orchestrator.schema.reference import ExperimentReference

if typing.TYPE_CHECKING:
    from orchestrator.core.resources import ADOResource
    from orchestrator.metastore.base import ResourceStore
    from orchestrator.modules.actuators.base import ActuatorBase

moduleLog = logging.getLogger("operation_base")


# Some operations are RayActors: These operations use Actuators and StateUpdateQueue and require Ray
# Some operations are not RayActors: They don't have to use Actuators and StateUpdateQueue or Ray. They can use ray-workers


class DiscoveryOperationBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def operationIdentifier(self) -> str:
        """A unique id for the operation instance being run by the operator

        should have form $operatorIdentifier-$version-$uid"""

    @classmethod
    @abc.abstractmethod
    def operatorIdentifier(cls) -> str:
        """The identifier of this operator

        should have form method-version"""

    @classmethod
    @abc.abstractmethod
    def operationType(cls) -> DiscoveryOperationEnum:
        """The type of operation this operator applies"""

    @classmethod
    @abc.abstractmethod
    def defaultOperationParameters(
        cls,
    ) -> pydantic.BaseModel:
        """A default pytdantic model for this operations parameters with this operator"""

    @classmethod
    @abc.abstractmethod
    def validateOperationParameters(
        cls,
        parameters: dict,
    ) -> pydantic.BaseModel:
        """If the parameters are valid returns a model for them.

        Otherwise, will raise ValidationErrors"""


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


# Note: We need async and sync versions because depending on agent
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
    from orchestrator.modules.actuators.base import (
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
        replayed_requests = orchestrator.modules.actuators.replay.replay(
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


class DiscoverySpaceSubscribingDiscoveryOperation(
    DiscoveryOperationBase,
    DiscoverySpaceUpdateSubscriber,
    metaclass=abc.ABCMeta,
):
    """Instances of this class can cause updates the state and receives details of updates via the StateUpdateSubscriber interface

    Instances of this class are RayActors and must run in Ray.
    They work on a Ray wrapped instance of the DiscoveryState (models.actors.InternalState)
    """

    def __init__(
        self,
        operationActorName: str,
        namespace: str | None,
        state: DiscoverySpaceManager,
        # Will actually be ray.actor.ActorHandle accessing InternalState
        actuators: dict[str, "orchestrator.modules.actuators.base.ActuatorBase"],
        params: dict | None = None,
        metadata: orchestrator.core.metadata.ConfigurationMetadata | None = None,
    ) -> None:
        # Common code for StateSubscribingDiscoveryOperations
        self.state = state
        self.actorName = operationActorName
        self.namespace = namespace
        # noinspection PyUnresolvedReferences
        self.state.subscribeToUpdates.remote(subscriberName=self.actorName)

        super().__init__()

    # async def run(self, discoveryState: orchestrator.model.actors.InternalState):
    #
    #     pass


class Characterize(
    DiscoverySpaceSubscribingDiscoveryOperation,
    UnaryDiscoveryOperation,
    metaclass=abc.ABCMeta,
):
    pass


class Search(
    DiscoverySpaceSubscribingDiscoveryOperation,
    UnaryDiscoveryOperation,
    metaclass=abc.ABCMeta,
):
    pass


class Compare(
    DiscoveryOperationBase, MultivariateDiscoveryOperation, metaclass=abc.ABCMeta
):
    pass


class Modify(DiscoveryOperationBase, UnaryDiscoveryOperation, metaclass=abc.ABCMeta):
    pass


class Fuse(
    DiscoveryOperationBase, MultivariateDiscoveryOperation, metaclass=abc.ABCMeta
):
    pass


class Learn(DiscoveryOperationBase, UnaryDiscoveryOperation, metaclass=abc.ABCMeta):
    pass


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

    operation = OperationResource(
        operationType=operation_resource_configuration.operation.module.operationType,
        operatorIdentifier=operation_resource_configuration.operation.module.operatorIdentifier,
        config=operation_resource_configuration,
        status=[output.exitStatus],
    )

    # ValueError means the resource has already been added
    with contextlib.suppress(ValueError):
        metastore.addResourceWithRelationships(
            resource=operation,
            relatedIdentifiers=[operation_resource_configuration.spaces[0]],
        )

    add_operation_output_to_metastore(operation, output, metastore)

    return operation


def create_operation_and_add_to_metastore(
    discovery_space: DiscoverySpace,
    operator_module: OperatorModuleConf | OperatorFunctionConf,
    operation_parameters: dict,
    operation_info: FunctionOperationInfo,
    metastore: SQLStore,
    operation_identifier: str | None = None,
) -> OperationResource:
    """Creates an operation resource and adds it to the metastore

    This function creates an OperationResource from the provided operator module,
    parameters, and operation info, then adds it to the metastore with relationships
    to the discovery space and any actuator configurations.

    Params:
        discovery_space: The discovery space the operation will operate on
        operator_module: Configuration for the operator (either module or function-based)
        operation_parameters: Dictionary of parameters for the operation
        operation_info: Information about the operation including metadata and actuator
            configuration identifiers
        metastore: The SQL store to add the operation resource to
        operation_identifier: Optional pre-existing identifier for the operation resource.
            If not provided, a new identifier will be generated

    Returns:
        OperationResource instance that was created and added to the metastore
    """

    from orchestrator.core.operation.config import DiscoveryOperationConfiguration

    operation_resource_configuration = DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=operator_module,
            parameters=operation_parameters,
        ),
        metadata=operation_info.metadata,
        actuatorConfigurationIdentifiers=operation_info.actuatorConfigurationIdentifiers,
        spaces=[discovery_space.resource.identifier],
    )

    operation = OperationResource(
        identifier=operation_identifier,
        operationType=operation_resource_configuration.operation.module.operationType,
        operatorIdentifier=operation_resource_configuration.operation.module.operatorIdentifier,
        config=operation_resource_configuration,
    )

    related_identifiers = [
        operation_resource_configuration.spaces[0],
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
        self.resources = resources if resources else []
        super().__init__(f"Operation {operation_identifier} was interrupted")


if typing.TYPE_CHECKING:
    from ray.actor import ActorHandle

    OperatorActor = type[ActorHandle[DiscoverySpaceSubscribingDiscoveryOperation]]
