# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import inspect
import typing
import uuid
from functools import wraps

import pydantic
import ray

import orchestrator.modules.actuators.catalog
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.base import (
    ActuatorBase,
    DeprecatedExperimentError,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.entity import (
    CheckRequiredObservedPropertyValuesPresent,
    Entity,
)
from orchestrator.schema.experiment import Experiment
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import ConstitutiveProperty
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import ValidMeasurementResult
from orchestrator.utilities.environment import enable_ray_actor_coverage
from orchestrator.utilities.logging import configure_logging

configure_logging()

# Module-level catalog for custom experiments
_custom_experiments_catalog = orchestrator.modules.actuators.catalog.ExperimentCatalog(
    catalogIdentifier="CustomExperiments"
)


def custom_experiment(properties: list[ConstitutiveProperty]):
    """
    Decorator for custom experiment functions.

    Args:
        properties: List of ConstitutiveProperty instances defining the input values the Entity must have

    Returns:
        A decorator that wraps a function to work with ADO's custom experiment system

    Example:
        from orchestrator.schema.property import ConstitutiveProperty

        @custom_experiment([
            ConstitutiveProperty(identifier="temperature"),
            ConstitutiveProperty(identifier="pressure")
        ])
        def my_experiment(temperature: float, pressure: float) -> dict[str, float]:
            # Function takes temperature and pressure as parameters
            # Returns a dict of observed property values
            return {
                "density": temperature * pressure * 0.001,
                "viscosity": temperature / pressure * 0.1
            }
    """

    def decorator(func):
        @wraps(func)
        def wrapper(
            entity: Entity, experiment: Experiment
        ) -> list[ObservedPropertyValue]:
            """
            Wrapper function that converts Entity+Experiment to dict and calls the wrapped function.

            Args:
                entity: The entity to measure
                experiment: The experiment configuration

            Returns:
                List of ObservedPropertyValue instances
            """
            # Convert Entity+Experiment to dict using propertyValuesFromEntity
            input_values = experiment.propertyValuesFromEntity(entity)

            # Call the wrapped function with the input values
            result_dict = func(**input_values)

            # Convert the result dict to ObservedPropertyValue list
            observed_property_values = []
            for property_identifier, value in result_dict.items():
                # Create ObservedProperty for this result
                observed_property = ObservedProperty(identifier=property_identifier)

                # Create ObservedPropertyValue
                observed_property_value = ObservedPropertyValue(
                    property=observed_property, value=value
                )
                observed_property_values.append(observed_property_value)

            return observed_property_values

        # Validate that the properties match the function parameters
        func_signature = inspect.signature(func)
        func_param_names = set(func_signature.parameters.keys())
        property_identifiers = {prop.identifier for prop in properties}

        if func_param_names != property_identifiers:
            raise ValueError(
                f"Function parameter names {func_param_names} do not match "
                f"property identifiers {property_identifiers}"
            )

        # Store decorator arguments as function attributes
        wrapper._decorator_properties = properties
        wrapper._decorator_func = func
        wrapper._is_custom_experiment = True

        # Create and store the Experiment instance
        experiment = Experiment(
            actuatorIdentifier="custom_experiments",
            identifier=func.__name__,
            requiredProperties=tuple(properties),
            optionalProperties=(),
            defaultParameterization=(),
            deprecated=False,
        )
        wrapper._experiment = experiment

        # Add the experiment to the module-level catalog
        _custom_experiments_catalog.addExperiment(experiment)

        return wrapper

    return decorator


def get_custom_experiment_info(func) -> dict:
    """
    Helper function to access decorator arguments and experiment from a decorated function.

    Args:
        func: A function decorated with @custom_experiment

    Returns:
        Dict containing decorator properties and experiment instance

    Raises:
        ValueError: If the function is not decorated with @custom_experiment
    """
    if not hasattr(func, "_is_custom_experiment") or not func._is_custom_experiment:
        raise ValueError("Function is not decorated with @custom_experiment")

    return {
        "properties": func._decorator_properties,
        "experiment": func._experiment,
        "original_func": func._decorator_func,
    }


def load_custom_experiments_from_entry_points():
    """
    Load custom experiments from entry points.

    This function searches for entry points under 'ado.custom_experiments' and loads
    any decorated functions from those modules.
    """
    try:
        import importlib
        import importlib.metadata

        # Get all entry points for ado.custom_experiments
        entry_points = importlib.metadata.entry_points()
        custom_experiment_groups = entry_points.get("ado.custom_experiments", [])

        for entry_point in custom_experiment_groups:
            entry_point.load()

    except ImportError:
        # importlib.metadata not available (Python < 3.8)
        pass


def get_custom_experiments_catalog() -> (
    orchestrator.modules.actuators.catalog.ExperimentCatalog
):
    """
    Get the module-level catalog of custom experiments.

    Returns:
        The ExperimentCatalog containing all registered custom experiments
    """
    return _custom_experiments_catalog


async def custom_experiment_wrapper(
    function: typing.Callable,
    parameters: dict,
    measurementRequest: MeasurementRequest,
    targetExperiment: Experiment,
    queue: MeasurementQueue,
):
    """
    :param function: The function to call
    :param parameters: The custom parameters to the function
    :param measurementRequest: The entity and custom experiment to be measured
    :param targetExperiment: The experiment to execute.
        Required as the measurementRequest only includes an ExperimentReference
    :param queue: The queue to put the result on
    :return:
    """

    measurement_results = []
    for entity in measurementRequest.entities:
        values = function(entity, targetExperiment, parameters=parameters)

        # Record the results in the entity
        if len(values) > 0:
            measurement_result = ValidMeasurementResult(
                entityIdentifier=entity.identifier, measurements=values
            )
            measurement_results.append(measurement_result)

    if len(measurement_results) > 0:
        measurementRequest.measurements = measurement_results
        measurementRequest.status = MeasurementRequestStateEnum.SUCCESS
    else:
        measurementRequest.status = MeasurementRequestStateEnum.FAILED

    await queue.put_async(measurementRequest, block=False)


@ray.remote
class CustomExperiments(ActuatorBase):
    identifier = "custom_experiments"

    """Actuator for applying user supplied custom experiments
    """

    def __init__(self, queue, params: dict | None = None):
        """

        :param queue: The StateUpdates queue instance
        :param params: The params for the objective-function

        """

        enable_ray_actor_coverage("custom_experiments")
        super().__init__(queue=queue, params=params)

        params = params if params else {}
        self.log.debug(f"Queue is {self._stateUpdateQueue}")
        self.log.debug(f"Params are {params}")

        # Use the module-level catalog by calling the class method
        self._catalog = type(self).catalog()
        self.log.debug(f"Catalog is {self._catalog}")

        self._functionImplementations = {}
        for experiment in self._catalog.experiments:
            # For custom experiments, we need to find the decorated function
            # The experiment identifier should match the function name
            function_name = experiment.identifier

            # Search for the function in the current module and loaded modules
            found_function = None

            # First, try to find it in the current module
            import sys

            current_module = sys.modules[__name__]
            if hasattr(current_module, function_name):
                potential_function = getattr(current_module, function_name)
                if (
                    hasattr(potential_function, "_is_custom_experiment")
                    and potential_function._is_custom_experiment
                ):
                    found_function = potential_function

            # If not found, search in other loaded modules
            if not found_function:
                for module_name, module in sys.modules.items():
                    if module and hasattr(module, function_name):
                        potential_function = getattr(module, function_name)
                        if (
                            hasattr(potential_function, "_is_custom_experiment")
                            and potential_function._is_custom_experiment
                        ):
                            found_function = potential_function
                            break

            if found_function:
                self._functionImplementations[experiment.identifier] = found_function
                self.log.info(
                    f"Experiment name: {experiment.identifier}. "
                    f"Function Implementation: {self._functionImplementations[experiment.identifier]}. "
                    f"Experiment: {experiment}"
                )
            else:
                self.log.warning(
                    f"Could not find function implementation for experiment {experiment.identifier}"
                )

        self.log.debug("Completed init")

    def loadedExperiment(
        self,
        experimentReference: ExperimentReference,
    ):

        return (
            self._functionImplementations.get(experimentReference.experimentIdentifier)
            is not None
        )

    async def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ):

        self.log.debug(
            f"Received a request to measure {experimentReference} on {[e.identifier for e in entities]}"
        )

        if self._catalog.experimentForReference(experimentReference) is None:
            if self._catalog.experiments:
                raise ValueError(
                    f"Requested experiments {experimentReference} is not in the CustomExperiments actuator catalog. "
                    f"Known experiments are {list(self._catalog.experimentsMap.keys())}"
                )
            raise ValueError(
                f"Requested experiments {experimentReference} is not in the CustomExperiments actuator catalog (which is empty). "
            )

        targetExperiment = self._catalog.experimentForReference(experimentReference)
        if targetExperiment.deprecated:
            raise DeprecatedExperimentError(
                f"{targetExperiment.actuatorIdentifier}.{targetExperiment.identifier} is deprecated."
            )

        # Check all required property values are present to actuate on
        for entity in entities:
            if not CheckRequiredObservedPropertyValuesPresent(
                entity, targetExperiment, exactMatch=False
            ):
                raise ValueError(
                    f"Entity {entity.identifier} does not have values for properties required "
                    f"as inputs for experiment {experimentReference.experimentIdentifier}"
                )

        # Create Measurement Request
        requestid = str(uuid.uuid4())[:6]
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=requestid,
        )

        self.log.debug(f"Create measurement request {request}")
        # TODO: Allow functions to specify if they should be remote
        experiment = self._catalog.experimentForReference(request.experimentReference)
        function = experiment.metadata.get("function", experiment.identifier)
        self.log.debug(f"Calling custom experiment {function}")

        await custom_experiment_wrapper(
            self._functionImplementations[
                request.experimentReference.experimentIdentifier
            ],
            self._catalog.experimentForReference(
                request.experimentReference
            ).metadata.get("parameters", {}),
            request,  # The request - contains experiment reference
            targetExperiment,  # Experiment to execute
            self._stateUpdateQueue,
        )

        # We only send one request
        return [requestid]


def load_custom_experiments_legacy(identifier):
    import importlib.resources
    import logging
    import pkgutil
    from pathlib import Path

    import ado_actuators as plugins
    import yaml

    from orchestrator.modules.actuators.catalog import ActuatorCatalogExtension
    from orchestrator.modules.actuators.registry import (
        CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME,
    )

    logger = logging.getLogger("custom_experiments")

    for module in pkgutil.iter_modules(plugins.__path__, f"{plugins.__name__}."):
        module_contents = {
            entry.name for entry in importlib.resources.files(module.name).iterdir()
        }

        if CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME in module_contents:
            logger.debug(f"Found {CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME}")

            experiments_configuration_file = Path(
                str(importlib.resources.files(module.name))
            ) / Path(CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME)

            try:
                catalog_extension = ActuatorCatalogExtension.model_validate(
                    yaml.safe_load(experiments_configuration_file.read_text())
                )
            except pydantic.ValidationError:
                logger.exception(
                    f"{module.name}'s {CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME} raised a validation error"
                )
                raise

            logger.debug(f"Adding catalog extension {catalog_extension!s}")
            # Check if catalog extension is for this actuator
            if catalog_extension.actuatorIdentifier == identifier:
                logger.debug(
                    f"Adding catalog extension {catalog_extension!s} for actuator {identifier}"
                )
                _custom_experiments_catalog.update(catalog_extension)

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:

        load_custom_experiments_legacy(cls.identifier)
        # Load custom experiments from entry points before returning catalog
        load_custom_experiments_from_entry_points()
        return get_custom_experiments_catalog()

    def current_catalog(
        self,
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        return self._catalog
