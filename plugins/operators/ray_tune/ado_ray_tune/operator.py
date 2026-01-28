# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import pydantic
import ray
import ray.air
import ray.tune
from pydantic import ConfigDict
from ray.actor import ActorHandle

import orchestrator.core
import orchestrator.modules
from orchestrator.core.datacontainer.resource import (
    DataContainer,
    DataContainerResource,
)
from orchestrator.core.discoveryspace.space import (
    SpaceInconsistencyError,
)
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.core.operation.resource import (
    OperationExitStateEnum,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.operators.base import (
    Search,
    measure_or_replay,
)
from orchestrator.modules.operators.discovery_space_manager import DiscoverySpaceManager
from orchestrator.schema.domain import PropertyDomain
from orchestrator.schema.entity import (
    Entity,
)
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.measurementspace import (
    MeasurementSpace,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.utilities.environment import enable_ray_actor_coverage
from orchestrator.utilities.support import prepare_dependent_experiment_input

from .config import (
    OrchRunConfig,
    OrchSearchAlgorithm,
    OrchTuneConfig,
    RayTuneConfiguration,
    RayTuneOrchestratorConfiguration,
)
from .samplers import LhuSampler

if TYPE_CHECKING:  # pragma: nocover
    from ray.tune.search.sample import Domain

    import orchestrator.modules.actuators.base


def run_dependent_experiments(
    request: MeasurementRequest,
    measurement_space: MeasurementSpace,
    actuators: dict[str, "orchestrator.modules.actuators.base.ActuatorBase"],
    queue: MeasurementQueue,
    requestIndex: int,
    singleMeasurement: bool,
    log: logging.Logger,
) -> list[str]:
    """Checks what dependent experiments can run based on a completed MeasureRequest and executes them

    Parameters:
        request: A completed MeasurementRequest.
        measurement_space: A MeasurementSpace instance detailing the experiments to apply
        actuators: A list of actuators instances that can execute the experiments
        queue: The StateUpdateQueue used to communicate the results of any launched experiment
        requestIndex: The request index to associate with any launched experiment
        singleMeasurement: If True existing measurements found for dependent experiments on the entity
             are reused
        log: logger to use

    Returns:
        A list of request_ids for the submitted experiments.
        If no dependent experiment can be executed the list will be empty"""

    request_ids = []

    prepared_inputs = prepare_dependent_experiment_input(
        measurement_request=request,
        measurement_space=measurement_space,
    )

    for d in prepared_inputs:
        log.debug(f"Requesting dependent experiment {d.experimentReference}")
        waiting_request_ids = measure_or_replay(
            requestIndex=d.requestIndex,
            requesterid=request.operation_id,
            entities=d.entities,
            experimentReference=d.experimentReference,
            actuators=actuators,
            measurement_queue=queue,
            memoize=singleMeasurement,
        )

        request_ids.extend(waiting_request_ids)

    return request_ids


def retrieve_results(
    entity: Entity,
    experimentReference: ExperimentReference,
    mode: Literal["target", "observed"] = "target",
) -> dict[str, Any]:
    """Returns a dictionary mapping property identifiers to their measured values.

    Args:
        entity: The entity that was measured
        experimentReference: Reference to the experiment that was applied
        mode: "target" returns target property identifiers as keys,
              "observed" returns observed property identifiers as keys

    Returns:
        Dictionary mapping property identifiers to their measured values
    """
    property_values = entity.propertyValuesFromExperimentReference(
        experimentReference=experimentReference
    )

    if mode == "target":
        return {p.property.targetProperty.identifier: p.value for p in property_values}
    # observed
    return {p.property.identifier: p.value for p in property_values}


class OrchTrainableParameters(pydantic.BaseModel):
    """Model for the information the orchestrator needs to pass to tune() and tune_trainable"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation_id: str
    ray_tune_actor_name: str
    ray_tune_actor_namespace: str | None
    measurement_space: MeasurementSpace
    entity_space: EntitySpaceRepresentation
    actuators: dict
    state: (
        ActorHandle  # We need the state to access the sample store which can't be sent
    )
    debugging: bool
    target_metric: str | list[str]  # Accept single or multiple target metrics
    orchestrator_config: RayTuneOrchestratorConfiguration


def process_metric(
    metric: str,
    all_results: dict[str, list[Any]],
    entity: Entity,
    trainable_params: OrchTrainableParameters,
) -> float | int | str | None:
    """
    Processes a single metric for a given entity.

    If the metric is in all_results, it returns the last result.
    If the metric is not in the all_results, it checks if it is a virtual property.
    If it is, it returns the value of the virtual property.
    If it is not, it returns the failed metric value.


    Args:
        metric (str): Name or identifier of the metric to process.
        all_results (dict[str, list[Any]]): A dictionary of all results, keyed by metric name.
        entity (Entity): The entity for which the metric is being processed.
        trainable_params (OrchTrainableParameters): Parameters/configuration for the trainable/orchestrator.

    Returns:
        Any: The processed metric value, or the failed metric value if the metric could not be found or computed.

    Raises:
        ValueError: If the metric is a virtual property and there are multiple observed properties with the same identifier.
    """

    log = logging.getLogger(f"trainable-{entity.identifier}")

    if all_results.get(metric):
        # We use the last result
        return all_results[metric][-1]

    # The metric is not in the results, so we need to process it
    # Check if this is a virtual metric
    log.debug(f"No measured properties match {metric} - checking if a virtual property")
    try:
        properties = entity.virtualObservedPropertiesFromIdentifier(metric)
    except ValueError:
        log.warning(
            f"No experiment measured {metric} and it's not a valid virtual property.  "
            f"Will set value of {metric} for {entity.identifier} to {trainable_params.orchestrator_config.failed_metric_value} "
        )
        processed_metric = trainable_params.orchestrator_config.failed_metric_value
    else:
        if properties is not None:
            if len(properties) == 1:
                value = entity.valueForProperty(property=properties[0])
                processed_metric = (
                    value.value
                    if value is not None
                    else trainable_params.orchestrator_config.failed_metric_value
                )
            else:
                raise ValueError(
                    f"Ambiguous virtual target metric provided - matches multiple observed properties. "
                    f"{[p.identifier for p in properties]}"
                )
        else:
            log.warning(
                f"{metric} is a valid virtual property name "
                f"however no experiment measured an underlying property with the required identifier. "
                f"Will set value of {metric} for {entity.identifier} to {trainable_params.orchestrator_config.failed_metric_value}"
            )
            processed_metric = trainable_params.orchestrator_config.failed_metric_value

    return processed_metric


def tune_trainable(config: dict, parameters: dict) -> dict[str, Any]:
    """
    Applies the experiments in the measurement space to a single entity

    Params:
        config: A dictionary of constitutive property name: value pairs (the entity)
        parameters: A dict representation of an OrchTrainableParameters instance

    Returns:
        A dict whose keys are the target properties in the measurement space
        and whose values are the measured values for the input entity

    Exceptions:

        Note: A valid trial that fails is not expected to cause an Exception to be raised from this function
        The actuator should handle this via an InvalidMeasurementResult
        All Exceptions raised from this function indicate that RayTune should stop.

        SystemError: if it detects a critical error notification has been sent to the operator
        UnknownExperimentError: If the requested experiment is unknown to the actuator
        DeprecatedExperimentError: If the requested experiment cannot be executed by the actuator because it is deprecated
    """

    trainable_params = OrchTrainableParameters(**parameters)
    single_measurement = (
        trainable_params.orchestrator_config.single_measurement_per_property
    )
    metric_mode = trainable_params.orchestrator_config.metric_format

    entity_space = trainable_params.entity_space
    actuators = trainable_params.actuators
    measurement_space = trainable_params.measurement_space
    measurement_queue = ray.get(trainable_params.state.measurement_queue.remote())
    property_values = [
        ConstitutivePropertyValue(value=config[cp.identifier], property=cp.descriptor())
        for cp in entity_space.constitutiveProperties
    ]
    has_dependent_experiments = len(measurement_space.dependentExperiments) > 0

    ops = []
    for experiment in measurement_space.experiments:
        ops.extend(experiment.observedProperties)

    cpid = Entity.identifier_from_property_values(property_values)

    log = logging.getLogger(f"trainable-{cpid}")

    log.debug(
        f"Getting driver using name {trainable_params.ray_tune_actor_name} and "
        f"namespace {trainable_params.ray_tune_actor_namespace}"
    )

    driver = ray.get_actor(
        trainable_params.ray_tune_actor_name,
        namespace=trainable_params.ray_tune_actor_namespace,
    )

    log.debug("START: running tune trainable")
    if ray.get(driver.isCriticalError.remote()):
        raise SystemError(
            "Exiting trainable as notification of critical error received by operator"
        )
    # Create Entity
    log.debug("Checking if entity is in discovery space")
    try:
        # We compare on constitutive properties
        retval = (
            trainable_params.state.storedEntitiesWithConstitutivePropertyValues.remote(
                property_values
            )
        )

        entity_list = ray.get(retval)
    except SpaceInconsistencyError:
        log.critical(
            "There are multiple entities with the same constitutive property value set"
        )
        raise
    except Exception as exc:
        log.critical(f"Could not obtain entities due to {exc}")
        raise
    else:
        entity = entity_list[0] if len(entity_list) > 0 else None

    if entity is None:
        log.debug("Entity not found - generating now")
        entity = Entity(
            identifier=cpid,
            constitutive_property_values=tuple(property_values),
            generatorid="ado_ray_tune",
        )
    else:
        log.debug("Found entity")

    requestIndex = ray.get(driver.getNextRequestIndex.remote())
    waitingRequestIds = []
    allResults = defaultdict(list)

    # First Phase: Independent Experiment
    log.debug("BEGIN PHASE ONE: independent experiments")
    log.debug(
        f"There are {len(measurement_space.independentExperiments)} independent experiments"
    )
    for experiment in measurement_space.independentExperiments:
        log.debug(f"Requesting experiment {experiment.identifier}")
        request_ids = measure_or_replay(
            requestIndex=requestIndex,
            requesterid=trainable_params.operation_id,
            entities=[entity],
            experimentReference=experiment.reference,
            actuators=actuators,
            measurement_queue=measurement_queue,
            memoize=single_measurement,
        )

        waitingRequestIds.extend(request_ids)

    log.debug(f"Experiments launched for {entity.identifier}: {waitingRequestIds}")
    log.debug(f"FINISHED PHASE ONE - launched {len(waitingRequestIds)} experiments")
    completedRequests = 0

    log.debug(
        "BEGIN PHASE TWO: collection of PHASE1 experiments results and launch "
        "of dependent experiments"
    )
    log.debug(
        f"Waiting on completion of {len(waitingRequestIds)} experiments from PHASE 1"
    )

    submittedExperiments = len(waitingRequestIds)
    waitingOnExperiments = True
    counter = 0
    while waitingOnExperiments:
        time.sleep(1)
        newDependentRequests = []
        newCompletedRequests = []

        log.debug(f"Checking {len(waitingRequestIds)} outstanding experiments")

        # Once a requestid is found completed it's removed from waitingRequestIds
        # So the dependent experiment will not execute more than once.
        for requestid in waitingRequestIds:
            log.debug(f"Checking if request {requestid} is complete")
            request = ray.get(
                driver.getRequest.remote(requestid)
            )  # type: MeasurementRequest
            if request is not None:
                log.debug(
                    f"Request {request} for {request.experimentReference} complete"
                )
                completedRequests += 1
                newCompletedRequests.append(request.requestid)
                if isinstance(request.measurements[0], ValidMeasurementResult):
                    for k, v in retrieve_results(
                        request.entities[0],
                        request.experimentReference,
                        mode=metric_mode,
                    ).items():
                        allResults[k].append(v)

                    if has_dependent_experiments:
                        newRequests = run_dependent_experiments(
                            request=request,
                            measurement_space=measurement_space,
                            actuators=actuators,
                            queue=measurement_queue,
                            requestIndex=requestIndex,
                            singleMeasurement=single_measurement,
                            log=log,
                        )
                        newDependentRequests.extend(newRequests)
                        submittedExperiments += len(newRequests)
                else:
                    log.warning(
                        f"Experiment {request.experimentReference} did produce any property measurements. "
                        f"Will set values of all properties to {trainable_params.orchestrator_config.failed_metric_value}"
                    )
                    # Record failure for all observed properties
                    experiment = measurement_space.experimentForReference(
                        request.experimentReference
                    )
                    for op in experiment.observedProperties:
                        allResults[op.targetProperty.identifier].append(
                            trainable_params.orchestrator_config.failed_metric_value
                        )
            else:
                log.debug(
                    f"Request {requestid} is not finished. {completedRequests}/{len(waitingRequestIds)}"
                )

        # Update waiting request ids
        waitingRequestIds = list(
            filter(lambda i: i not in newCompletedRequests, waitingRequestIds)
        )
        waitingRequestIds.extend(newDependentRequests)

        if counter % 120 == 0:
            log.debug(f"PHASE TWO STATE SUMMARY UPDATE {counter % 120}:")
            log.debug(
                f"{len(newCompletedRequests)} new experiments completed. "
                f"{len(newDependentRequests)} new experiments added"
            )

            if len(waitingRequestIds) > 0:
                log.debug(f"Currently waiting on {len(waitingRequestIds)} experiments.")

            log.info(
                f"{completedRequests} experiments completed in total. "
                f"{submittedExperiments} experiments launched in total"
            )

        counter += 1

        waitingOnExperiments = completedRequests < submittedExperiments

    log.info(f"Measurement properties obtained: {allResults}")
    log.debug("FINISHED PHASE TWO")

    # There may have been multiple results for some properties.
    # The trainable can only return one.
    # The following code either returns the last available value or if the metric is virtual, aggregates it.
    # It also handles the case where no value of target metric is available
    # All results will be in the same format (target or observed) based on metric_mode
    final_results = {}

    target_metrics = (
        trainable_params.target_metric
        if isinstance(trainable_params.target_metric, list)
        else [trainable_params.target_metric]
    )
    for metric in target_metrics:
        final_results[metric] = process_metric(
            metric=metric,
            all_results=allResults,
            entity=entity,
            trainable_params=trainable_params,
        )
    # Add non-target metrics to final results in the same format - skip any already handled
    skip_metrics = list(target_metrics)
    for k, v in allResults.items():
        if k not in skip_metrics:
            final_results[k] = v[-1]
    return final_results


class TuneOutput(NamedTuple):
    result: ray.tune.Result
    exit_state: OperationExitStateEnum
    error: str = None


@ray.remote
def tune(
    search_space: dict,
    config: RayTuneConfiguration,
    parameters: OrchTrainableParameters,
) -> TuneOutput:
    """ "
    Parameters:
        search_space: A ray tune search-space dict
        config: A RayTuneConfiguration object with options for the run
            This includes orchestrator specific options
        parameters: A dict with internal parameters that enable the tune_trainable function
            to interact with the orchestrator while being called by ray tune.

    Returns:
        ray.tune.Result instance containing information on the best result found
        If no valid result was found before failure the result object will be for the entity that caused the error
    """

    import logging

    log = logging.getLogger("tune")

    ray_tune_config = config.tuneConfig.rayTuneConfig()
    ray_runtime_config = config.runtimeConfig.rayRuntimeConfig()

    configured_stoppers = (
        [e.name for e in config.runtimeConfig.stop] if config.runtimeConfig.stop else []
    )
    data_columns = list(search_space.keys())
    if "InformationGainStopper" in configured_stoppers:
        # If there is more than one stopper then runtimeConfig will have a CombinedStopper
        if len(configured_stoppers) > 1:
            IG_stopper = ray_runtime_config.stop._stoppers[
                configured_stoppers.index("InformationGainStopper")
            ]
        else:
            IG_stopper = ray_runtime_config.stop

        search_columns = [
            k.identifier
            for k in parameters.entity_space.constitutiveProperties
            if k.propertyDomain.size > 1
        ]

        import math

        try:
            total_size = parameters.entity_space.size
        except AttributeError:
            total_size = math.inf

        targeted_value = parameters.target_metric
        IG_stopper.configure_details(
            data_columns=data_columns,
            targeted_value=targeted_value,
            search_columns=search_columns,
            total_size=total_size,
        )

    parameters.target_metric = ray_tune_config.metric  # still supports str or list

    ## LhuSampler requires entity space
    if isinstance(ray_tune_config.search_alg, LhuSampler):
        ray_tune_config.search_alg.set_entity_space(parameters.entity_space)

    try:
        tuner = ray.tune.Tuner(
            ray.tune.with_parameters(
                tune_trainable, parameters=parameters.model_dump()
            ),
            param_space=search_space,
            tune_config=ray_tune_config,
            run_config=ray_runtime_config,
        )
    except Exception as e:
        log.critical(f"got unexpected exception during Tuner initialisation: {e}")
        raise e

    try:
        # The tuner is configured via run_config to fail on first exception raised in a trial
        # As the tune_trainable() should only raise Exceptions on cases that the whole run should exit
        # "Normal" failure cases should be handled via the actuator and ados InvalidMeasurementResult reporting
        results = tuner.fit()
    except Exception as e:
        log.critical(f"got unexpected exception during search: {e}")
        raise e

    # Check if the run was stop - this is if any trial raised an exception
    failed_trials = [result for result in results if result.error is not None]
    if failed_trials:
        log.critical(f"There were {len(failed_trials)} failed trials")

    for result in results:
        if result.error:
            log.debug(f"Failed: {result}")

    error = None
    if failed_trials:
        # The type of error is Exception but
        error: ray.exceptions.RayTaskError = failed_trials[0].error
        error = str(error.cause) if error.cause else str(error)
        log.debug(f"Error is {error}")

    operation_status = (
        OperationExitStateEnum.FAIL if failed_trials else OperationExitStateEnum.SUCCESS
    )

    return TuneOutput(
        exit_state=operation_status,
        error=error,
        result=results.get_best_result(
            metric=ray_tune_config.metric, mode=ray_tune_config.mode
        ),
    )


def property_domain_to_ray_distribution(
    domain: PropertyDomain,
) -> "Domain":
    # Later we can use sample_from to support any distribution

    from orchestrator.schema.domain import ProbabilityFunctionsEnum, VariableTypeEnum

    if domain.variableType == VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE:
        # Categorical values always return choice - currently no method to describe a non-uniform distribution
        retval = ray.tune.choice(domain.values)
    elif domain.variableType == VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE:
        # Continuous variables
        # "uniform" => uniform
        # "normal" => randn
        # Others not supported yet
        if domain.probabilityFunction.identifier == ProbabilityFunctionsEnum.UNIFORM:
            # TODO: Check range is enforced for uniform dist
            retval = ray.tune.uniform(
                lower=min(domain.domainRange), upper=max(domain.domainRange)
            )
        elif domain.probabilityFunction.identifier == ProbabilityFunctionsEnum.NORMAL:
            retval = ray.tune.randn(
                mean=domain.probabilityFunction.parameters["mean"],
                sd=domain.probabilityFunction.parameters["std"],
            )
        else:
            raise ValueError(
                f"Probability function {domain.probabilityFunction.identifier} "
                "not currently supported with ray tune for continuous variables"
            )
    elif domain.variableType == VariableTypeEnum.DISCRETE_VARIABLE_TYPE:
        # Discrete variables
        # "uniform" => randint if interval is 1
        # "uniform" => qrandint if interval is > 1
        if domain.values is not None:
            # Ray grid-search method changes the meaning of `num-samples` parameter to RayTune
            # Instead of being the num-samples selected it's the number of times to repeat measure each grid point
            # This is confusing so we use choice
            retval = ray.tune.choice(domain.values)
        elif domain.probabilityFunction.identifier == ProbabilityFunctionsEnum.UNIFORM:
            if domain.interval == 1:
                retval = ray.tune.randint(
                    lower=min(domain.domainRange), upper=max(domain.domainRange)
                )
            else:
                # This makes the sampling inclusive of upper bound
                retval = ray.tune.qrandint(
                    lower=min(domain.domainRange),
                    upper=max(domain.domainRange),
                    q=int(domain.interval),
                )
        else:
            raise ValueError(
                f"Unable to determine ray tune configuration for domain {domain}"
            )
    else:
        raise ValueError(f"Probability domains, {domain}, variable type is not valid")

    return retval


def search_space_from_explicit_entity_space(
    entitySpace: EntitySpaceRepresentation,
) -> dict:
    """Returns a ray tune search space dictionary from an explicit entity space"""

    space = {}
    for cp in entitySpace.constitutiveProperties:
        space[cp.identifier] = property_domain_to_ray_distribution(cp.propertyDomain)

    return space


@ray.remote
class RayTune(Search):
    """Uses raytune optimization algorithm to search through entities in a space"""

    @classmethod
    def defaultOperationParameters(
        cls,
    ) -> RayTuneConfiguration:
        return RayTuneConfiguration(
            tuneConfig=OrchTuneConfig(
                metric="wallclock_time", search_alg=OrchSearchAlgorithm(name="bayesopt")
            ),
            runtimeConfig=OrchRunConfig(),
        )

    @classmethod
    def validateOperationParameters(cls, parameters: dict) -> RayTuneConfiguration:
        return RayTuneConfiguration.model_validate(parameters)

    @classmethod
    def description(cls) -> str:
        return """RayTune provides capabilities for sampling points in an entity space and applying
               measurements to them via optimization algorithms.

              All Ray Tune optimizers (except Ax) e.g. Nevergrad, BOHB, HyperBand are supported
              and accept the same configuration parameters as in raytune.

              Support for Ax has been removed due to incompatibilities with newer Numpy versions.
              """

    def __init__(
        self,
        operationActorName: str,
        namespace: str,
        state: DiscoverySpaceManager,
        actuators: dict[str, "orchestrator.modules.actuators.base.ActuatorBase"],
        params: dict | None = None,
    ) -> None:
        import os

        enable_ray_actor_coverage("ado_ray_tune")
        FORMAT = "%(levelname)-9s %(threadName)-30s %(name)-30s: %(funcName)-20s %(asctime)-15s: %(message)s"
        LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
        logging.basicConfig(level=LOGLEVEL, format=FORMAT)

        self.runid = str(uuid.uuid4())[:6]
        params = params if params is not None else {}
        self.log = logging.getLogger("RayTune")

        self.params = RayTuneConfiguration(**params)

        self.actuators = actuators
        self._entitiesSubmitted = 0
        self._finishedMeasurements = {}
        self._requestIndex = 0
        self.received_critical_error_notification = False
        self.criticalError = None  # Will store the critical error if we receive one

        # Sets state, actorName ivars and subscribes to the state
        super().__init__(
            operationActorName=operationActorName,
            namespace=namespace,
            state=state,
            actuators=actuators,
        )

    def onUpdate(self, measurementRequest: MeasurementRequest) -> None:
        self._finishedMeasurements[measurementRequest.requestid] = measurementRequest

    def onCompleted(self) -> None:
        pass

    def onError(self, error: Exception) -> None:
        self.criticalError = error
        self.received_critical_error_notification = True

    def isCriticalError(self) -> bool:
        return self.received_critical_error_notification

    def isRequestCompleted(self, requestid: str) -> bool:
        return self._finishedMeasurements.get(requestid) is not None

    def getRequest(self, requestid: str) -> MeasurementRequest:
        return self._finishedMeasurements.get(requestid)

    def getNextRequestIndex(self) -> int:
        retval = self._requestIndex
        self._requestIndex += 1

        return retval

    async def run(self) -> OperationOutput:
        try:
            # noinspection PyUnresolvedReferences
            entity_space = (
                await self.state.entitySpace.remote()
            )  # type: EntitySpaceRepresentation
            # noinspection PyUnresolvedReferences
            measurement_space = (
                await self.state.measurementSpace.remote()
            )  # type: MeasurementSpace

            metric_or_metrics = self.params.tuneConfig.metric
            metric_mode = self.params.orchestratorConfig.metric_format

            # Validate metrics match the configured mode
            metrics_to_check = (
                metric_or_metrics
                if isinstance(metric_or_metrics, list)
                else [metric_or_metrics]
            )

            metrics_valid = all(
                measurement_space.propertyWithIdentifierInSpace(m, format=metric_mode)
                for m in metrics_to_check
            )

            if not metrics_valid:
                error_message = (
                    f"Metric(s) {metric_or_metrics} not found as {metric_mode} "
                    f"properties in measurement space"
                )
                self.log.error(error_message)

            if metrics_valid:
                internal_parameters = OrchTrainableParameters(
                    operation_id=self.operationIdentifier(),
                    ray_tune_actor_name=self.actorName,
                    ray_tune_actor_namespace=self.namespace,
                    measurement_space=measurement_space,
                    entity_space=entity_space,
                    actuators=self.actuators,
                    state=self.state,
                    orchestrator_config=self.params.orchestratorConfig,
                    target_metric=self.params.tuneConfig.metric,
                    debugging=False,
                )

                self.log.debug(internal_parameters)

                search_space = search_space_from_explicit_entity_space(entity_space)

                self.log.debug(search_space)

                # Create the tune instance

                # noinspection PyArgumentList
                output: TuneOutput = await tune.remote(
                    search_space, config=self.params, parameters=internal_parameters
                )

                self.log.debug(f"Tune Result: {output}")
                result_dict = {
                    "config": output.result.config,
                    "metrics": output.result.metrics,
                    "error": output.result.error if output.result.error else None,
                }
                resources = [
                    DataContainerResource(
                        config=DataContainer(data={"best_result": result_dict})
                    )
                ]

                if output.exit_state == OperationExitStateEnum.FAIL:
                    self.log.debug("tune() exited reporting failure")
                    if self.received_critical_error_notification:
                        self.log.debug("failure due to external notification")
                        operation_output = OperationOutput(
                            resources=resources,
                            exitStatus=OperationResourceStatus(
                                message=f"Ray Tune operation exited as the operator received notification "
                                f"there was a critical error external to the operation: {self.criticalError}",
                                exit_state=OperationExitStateEnum.FAIL,
                                event=OperationResourceEventEnum.FINISHED,
                            ),
                        )
                    else:
                        self.log.debug("failure likely due to actuator")
                        operation_output = OperationOutput(
                            resources=resources,
                            exitStatus=OperationResourceStatus(
                                message=f"Ray Tune operation exited due to an exception from an actuator:  {output.error}",
                                exit_state=OperationExitStateEnum.FAIL,
                                event=OperationResourceEventEnum.FINISHED,
                            ),
                        )
                else:
                    operation_output = OperationOutput(
                        resources=resources,
                        exitStatus=OperationResourceStatus(
                            message="Ray Tune operation completed successfully",
                            exit_state=OperationExitStateEnum.SUCCESS,
                            event=OperationResourceEventEnum.FINISHED,
                        ),
                    )

            else:
                operation_output = OperationOutput(
                    exitStatus=OperationResourceStatus(
                        message=f"Ray Tune operation did not start. {error_message}",
                        exit_state=OperationExitStateEnum.FAIL,
                        event=OperationResourceEventEnum.FINISHED,
                    ),
                )
        except Exception as error:
            operation_output = OperationOutput(
                exitStatus=OperationResourceStatus(
                    message=f"Unexpected exception when running RayTune operation: {error}",
                    exit_state=OperationExitStateEnum.ERROR,
                    event=OperationResourceEventEnum.FINISHED,
                ),
            )

        # noinspection PyUnresolvedReferences
        self.state.unsubscribeFromUpdates.remote(subscriberName=self.actorName)

        return operation_output

    def numberEntitiesSampled(self) -> int:
        return self._requestIndex

    def numberMeasurementsRequested(self) -> int:
        # FIXME: This is not correct. Its request*experiments per entity
        return self._requestIndex

    def operationIdentifier(self) -> str:
        return f"{self.__class__.operatorIdentifier()}-{self.params.tuneConfig.search_alg.name}-{self.runid}"

    @classmethod
    def operatorIdentifier(cls) -> str:
        from importlib.metadata import version

        version = version("ado-core")

        return f"raytune-{version}"

    @classmethod
    def operationType(cls) -> DiscoveryOperationEnum:
        return DiscoveryOperationEnum.SEARCH
