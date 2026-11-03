# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Sequential predictive acquisition and final-model orchestration."""

from __future__ import annotations

import typing

from ado.core.operation.config import FunctionOperationInfo
from ado.core.operation.operation import OperationOutput
from ado.modules.operators.randomwalk import RandomWalk
from ado.schema.request import MeasurementRequestStateEnum

if typing.TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from queue import Queue

    from active_learning.regression._shared import (
        _PredictiveOperatorParameters,
        _PredictiveParameters,
        _PredictiveSampleSelector,
    )
    from ado.core.discoveryspace.space import DiscoverySpace
    from ado.modules.actuators.base import ActuatorBase
    from ado.modules.operators.discovery_space_manager import DiscoverySpaceManager
    from ado.modules.operators.randomwalk import RandomWalkParameters
    from ado.schema.entity import Entity
    from ado.schema.measurementspace import MeasurementSpace
    from ado.schema.request import MeasurementRequest


class _SequentialPredictiveWalk(RandomWalk):
    """Complete an entity's measurements before requesting the next entity."""

    def __init__(
        self,
        operationActorName: str,
        namespace: str,
        discovery_space_manager: DiscoverySpaceManager,
        actuators: dict[str, ActuatorBase],
        params: dict | None = None,
    ) -> None:
        """Initialize the completed-request count and measurement scheduler."""

        self._finished_requests = 0
        super().__init__(
            operationActorName=operationActorName,
            namespace=namespace,
            discovery_space_manager=discovery_space_manager,
            actuators=actuators,
            params=params,
        )

    async def run(self) -> None:
        """Run measurements and propagate critical infrastructure failures."""

        await super().run()
        if self.criticalError:
            raise RuntimeError(
                "measurement operation stopped after a critical infrastructure error"
            )

    def _processCompletedMeasurement(
        self,
        measurementRequest: MeasurementRequest,
        measurementSpace: MeasurementSpace,
        continuousBatchingQueue: Queue,
    ) -> bool:
        """Count completed attempts and report exhausted experiment failures."""

        if measurementRequest.operation_id != self.operationIdentifier():
            return False
        completed = super()._processCompletedMeasurement(
            measurementRequest, measurementSpace, continuousBatchingQueue
        )
        self._finished_requests += 1
        if (
            completed
            and measurementRequest.status != MeasurementRequestStateEnum.SUCCESS
        ):
            identifiers = [entity.identifier for entity in measurementRequest.entities]
            raise RuntimeError(
                f"experiment {measurementRequest.experimentReference} exhausted "
                f"its retry limit ({self.params.maxRetries}) for entities {identifiers}"
            )
        return completed

    async def _sampleEntityAndAddMeasurementsToQueue(
        self,
        continuousBatchingQueue: Queue,
        iterator: AsyncGenerator[list[Entity], None],
        measurementSpace: MeasurementSpace,
    ) -> list[Entity]:
        """Sample only after all submitted measurement attempts have finished."""

        if self._finished_requests < self._experimentsRequested:
            return []
        return await super()._sampleEntityAndAddMeasurementsToQueue(
            continuousBatchingQueue, iterator, measurementSpace
        )


def _random_walk_parameters(
    parameters: _PredictiveParameters,
    *,
    number_entities: int,
    sampler_class: type[_PredictiveSampleSelector],
) -> RandomWalkParameters:
    """Build the fixed sequential RandomWalk configuration for a selector."""

    from ado.modules.operators.randomwalk import (
        CustomSamplerConfiguration,
        EntityFilter,
        FilterModeEnum,
        RandomWalkParameters,
        SamplerModuleConf,
    )

    sampler_configuration = CustomSamplerConfiguration(
        module=SamplerModuleConf(
            moduleClass=sampler_class.__name__,
            moduleName=sampler_class.__module__,
        ),
        parameters=parameters.model_dump(
            include=set(sampler_class.parameters_model().model_fields)
        ),
    )
    return RandomWalkParameters(
        samplerConfig=sampler_configuration,
        numberEntities=number_entities,
        batchSize=1,
        singleMeasurement=True,
        filter=EntityFilter(filterMode=FilterModeEnum.noFilter),
    )


def _outer_operation_output(
    nested_output: OperationOutput,
    *,
    final_model: dict[str, typing.Any] | None = None,
) -> OperationOutput:
    """Expose a nested RandomWalk operation as a resource of the outer one."""

    return OperationOutput(
        other=[],
        resources=[nested_output.operation] if nested_output.operation else [],
        metadata={"finalModel": final_model} if final_model is not None else {},
        exitStatus=nested_output.exitStatus,
    )


def _run_predictive_operator(
    *,
    discovery_space: DiscoverySpace,
    operation_info: FunctionOperationInfo | None,
    parameters: _PredictiveOperatorParameters,
    number_entities: int,
    sampler_class: type[_PredictiveSampleSelector],
    output_directory: str,
) -> OperationOutput:
    """Run predictive acquisition with sequential measurement scheduling."""

    # Lazy import avoids a circular import while ADO loads operator plugins.
    from active_learning.regression._finalize import (
        _expected_training_count,
        _final_model_directory,
        _finalize_predictive_model,
    )
    from active_learning.regression._predictor import _validate_final_predictor
    from ado.core.operation.resource import OperationExitStateEnum
    from ado.modules.operators.collections import explore
    from ado.modules.operators.orchestrate import orchestrate_explore_operation

    directory = _final_model_directory(output_directory)
    expected_training_count = _expected_training_count(
        discovery_space, parameters=parameters, number_entities=number_entities
    )
    _validate_final_predictor(
        parameters, expected_training_count=expected_training_count
    )
    random_walk_parameters = _random_walk_parameters(
        parameters,
        number_entities=number_entities,
        sampler_class=sampler_class,
    )
    random_walk_metadata = explore.operators["random_walk"].model_copy(
        update={"cls": _SequentialPredictiveWalk}
    )
    nested_operation_info = FunctionOperationInfo(
        metadata={
            "implementation": (
                f"{_SequentialPredictiveWalk.__module__}."
                f"{_SequentialPredictiveWalk.__name__}"
            )
        },
        actuatorConfigurationIdentifiers=list(
            operation_info.actuatorConfigurationIdentifiers
            if operation_info is not None
            else []
        ),
    )
    nested_output = orchestrate_explore_operation(
        operator_metadata=random_walk_metadata,
        inputs={"discoverySpace": discovery_space},
        operation_info=nested_operation_info,
        parameters=random_walk_parameters,
    )
    if (
        nested_output.exitStatus is None
        or nested_output.exitStatus.exit_state != OperationExitStateEnum.SUCCESS
    ):
        raise RuntimeError("measurement operation did not complete successfully")
    final_model = _finalize_predictive_model(
        discovery_space,
        parameters=parameters,
        directory=directory,
        minimum_training_count=expected_training_count,
    )
    return _outer_operation_output(nested_output, final_model=final_model)
