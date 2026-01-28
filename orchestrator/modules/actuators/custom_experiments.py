# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import inspect
import logging
import typing
import uuid
from collections.abc import Callable
from typing import Annotated, Any

import pydantic
import ray

import orchestrator.modules.actuators.catalog
from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.base import (
    ActuatorBase,
    DeprecatedExperimentError,
)
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)
from orchestrator.schema.entity import (
    CheckRequiredObservedPropertyValuesPresent,
    Entity,
)
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest, MeasurementRequestStateEnum
from orchestrator.schema.result import InvalidMeasurementResult, ValidMeasurementResult
from orchestrator.utilities.environment import enable_ray_actor_coverage
from orchestrator.utilities.logging import configure_logging

configure_logging()

# Module-level catalog for custom experiments
_custom_experiments_catalog = orchestrator.modules.actuators.catalog.ExperimentCatalog(
    catalogIdentifier="CustomExperiments"
)


class RayRemoteOptions(pydantic.BaseModel):
    num_cpus: float | None = None
    num_gpus: float | None = None
    resources: dict | None = None
    runtime_env: dict | None = None


class ExperimentModuleConf(ModuleConf):
    moduleType: Annotated[ModuleTypeEnum, pydantic.Field()] = ModuleTypeEnum.EXPERIMENT


def _infer_domain_and_property(
    identifier: str, annotation: type, default: Any  # noqa: ANN401
) -> ConstitutiveProperty:
    """This function infers the domain of a parameter from its type and default value.
    Parameters:
    - identifier: The name of the parameter
    - annotation: The type of the parameter. Must be a valid python type
    - default: The default value of the parameter
    Returns:
    - A ConstitutiveProperty instance with the inferred domain
    Exceptions:
    - ValueError: If the parameter is not supported i.e. the domain cannot be inferred
    """
    import logging

    logger = logging.getLogger("custom_experiments")
    from typing import get_args, get_origin

    from orchestrator.schema.domain import PropertyDomain, VariableTypeEnum

    if annotation is int:
        domain = PropertyDomain(
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE, interval=1
        )
    elif annotation is float:
        domain = PropertyDomain(variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE)
    elif annotation is bool:
        domain = PropertyDomain(variableType=VariableTypeEnum.BINARY_VARIABLE_TYPE)
    elif annotation is str:
        domain = PropertyDomain(
            variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE,
            values=[default],
        )
    elif get_origin(annotation) in [
        getattr(typing, "Literal", None),
        getattr(__import__("typing_extensions"), "Literal", None),
    ] or str(annotation).startswith("typing.Literal"):
        vals = list(get_args(annotation))
        domain = PropertyDomain(
            variableType=VariableTypeEnum.CATEGORICAL_VARIABLE_TYPE, values=vals
        )
    else:
        logger.warning(
            f"Error parameter '{identifier}' - unsupported annotation: {annotation}"
        )
        raise ValueError(f"Unsupported annotation: {annotation}")

    return ConstitutiveProperty(identifier=identifier, propertyDomain=domain)


def derive_required_properties_from_signature(
    func: Callable, optional_property_identifiers: list[str]
) -> list[ConstitutiveProperty]:
    """This function derives the required properties from the function signature.

    The required properties are the positional parameters of the function that are not in optional_property_identifiers.

    Parameters:
    - func: The function to derive the required properties from
    - optional_property_identifiers: The identifiers of the optional properties
    Returns:
    - A list of ConstitutiveProperty instances
    """

    func_signature = inspect.signature(func)
    required_props = []
    for param in func_signature.parameters.values():
        if param.name in optional_property_identifiers:
            continue
        if (
            param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and param.default is inspect.Parameter.empty
        ):
            inferred_prop = _infer_domain_and_property(
                param.name, param.annotation, None
            )
            required_props.append(inferred_prop)
    return required_props


def get_parameterization(
    properties: list[ConstitutiveProperty], func_signature: inspect.Signature
) -> dict[str, Any]:
    """This function derives the parameterization of properties and function signature.

    The parameterization of a property is the default value of the corresponding parameter in func_signature

    Parameters:
    - properties: The properties to derive the parameterization for
    - func_signature: The function signature to derive the parameterization from
    Returns:
    - A dictionary of property identifiers and their default values
    Exceptions:
    - ValueError: If a parameterization cannot be derived for a property"""
    param_map = {p.name: p for p in func_signature.parameters.values()}
    results = {}
    missing = []
    for prop in properties:
        param = param_map.get(prop.identifier, None)
        if param and param.default is not inspect.Parameter.empty:
            results[prop.identifier] = param.default
        else:
            missing.append(prop.identifier)
    if missing:
        raise ValueError(f"Parameterization missing for: {missing}")
    return results


def derive_optional_properties_and_parameterization(
    func: Callable, required_properties: list[ConstitutiveProperty]
) -> tuple[list[ConstitutiveProperty], dict[str, Any]]:
    """This function derives the optional properties and their parameterization from the function signature.

    The optional properties are the keyword parameters of the function that are not in required_properties.
    The parameterization of an optional property is the default value of the corresponding parameter in func_signature.

    Parameters:
    - func: The function to derive the optional properties and parameterization from
    - required_properties: The properties that are required input values.
    Returns:
    - A tuple. The first element is a list of optional properties, the second element is a dictionary of property identifiers and their default values
    Exceptions:
    - ValueError: If a parameterization cannot be derived for any optional property (unexpected)
    - ValueError: If a domain cannot be inferred for any optional property"""
    optional_properties = []
    for param in inspect.signature(func).parameters.values():
        if param.name in {prop.identifier for prop in required_properties}:
            continue
        if (
            param.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and param.default is not inspect.Parameter.empty
        ):
            inferred_prop = _infer_domain_and_property(
                param.name, param.annotation, param.default
            )
            optional_properties.append(inferred_prop)

    return optional_properties, get_parameterization(
        optional_properties, inspect.signature(func)
    )


def check_parameters_and_infer(
    func: Callable[..., Any],
    _required_properties: list[ConstitutiveProperty] | None,
    _optional_properties: list[ConstitutiveProperty] | None = None,
    _parameterization: dict | None = None,
) -> tuple[
    list[ConstitutiveProperty] | list[Any],
    dict | None | dict[str, Any],
    list[ConstitutiveProperty],
]:
    logger = logging.getLogger("custom_experiment_decorator")

    # Set up dynamic optional_properties and parameterization if none were provided
    _optional_properties = _optional_properties if _optional_properties else []

    if not _required_properties:
        try:
            _required_properties = derive_required_properties_from_signature(
                func, {prop.identifier for prop in _optional_properties}
            )
        except ValueError as error:
            logger.critical(
                f"No required properties provided and they could not be derived from signature: {error}"
            )
            raise error
    if _optional_properties and not _parameterization:
        try:
            _parameterization = get_parameterization(
                _optional_properties, inspect.signature(func)
            )
        except ValueError as error:
            logger.critical(
                f"Optional properties provided but parameterization could not be derived from signature: {error}"
            )
            raise error
    if not _optional_properties and not _parameterization:
        try:
            _optional_properties, _parameterization = (
                derive_optional_properties_and_parameterization(
                    func, _required_properties
                )
            )
        except ValueError as error:
            logger.critical(
                f"No optional properties provided and theycould not be derived from signature: {error}"
            )
            raise error

    return _optional_properties, _parameterization, _required_properties


def check_parameters_valid(
    func: Callable[..., Any],
    _required_properties: list[ConstitutiveProperty | ObservedProperty] | None,
    _optional_properties: list[ConstitutiveProperty] | None,
) -> None:
    # Validate that the property identifiers match the function parameters
    func_signature = inspect.signature(func)
    func_param_names = set(func_signature.parameters.keys())
    req_property_identifiers = {prop.identifier for prop in _required_properties}
    opt_property_identifiers = (
        {prop.identifier for prop in _optional_properties}
        if _optional_properties
        else set()
    )
    experiment_prop_identifiers = req_property_identifiers | opt_property_identifiers
    if not experiment_prop_identifiers.issubset(func_param_names):
        raise ValueError(
            f"{func.__name__} parameter names {func_param_names} must include all property identifiers {experiment_prop_identifiers}. "
            f"Missing identifiers: {experiment_prop_identifiers - func_param_names}"
        )


def custom_experiment(
    output_property_identifiers: list[str],
    required_properties: list[ConstitutiveProperty | ObservedProperty] | None = None,
    optional_properties: list[ConstitutiveProperty] | None = None,
    parameterization: dict[str, Any] | None = None,  # noqa: ANN401
    metadata: dict[str, Any] | None = None,  # noqa: ANN401
    use_ray: bool = True,
    ray_options: dict | None = None,
) -> Callable[
    [Callable[..., Any]],  # noqa: ANN401
    Callable[[tuple[Any, ...], dict[str, Any]], Any],  # noqa: ANN401
]:
    """
    Decorator for custom experiment functions.

    Args:
        required_properties: List of ConstitutiveProperty instances that are required input values.
        output_property_identifiers: List of strings identifying the output property names.
        optional_properties: List of ConstitutiveProperty instances that are optional input values.
        parameterization: Tuple of parameters for default parameterization.
        metadata: Metadata for the experiment
        use_ray: If True the CustomExperiments actuator will launch the experiment as a ray remote task
        ray_options: A dictionary containing ray remote task options.
            The keys and allowed values are defined by RayRemoteOptions

    Returns:
        A decorator that wraps a function to work with ado's custom experiment system

    Raises:
        ValueError: If Unable to generate custom function via decorator e.g.
        - No required properties provided, and they could not be derived from signature:
        - Optional properties provided but parameterization could not be derived from signature:
        - No optional properties provided and they could not be derived from signature
        - Function parameter names did not include all property identifiers

    Example:

    mass = ConstitutiveProperty(identifier="mass", propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0, 100],
    ))
    volume = ConstitutiveProperty(identifier="volume", propertyDomain=PropertyDomain(
        variableType=VariableTypeEnum.CONTINUOUS_VARIABLE_TYPE,
        domainRange=[0, 100],
    ))

    @custom_experiment(
        required_properties=[mass, volume],
        output_properties=["density"]
    )
    def calculate_density(mass, volume):
        density_value = mass / volume if volume else None
        return {
            "density": density_value
        }
    """

    metadata = metadata if metadata else {}
    required_properties = required_properties if required_properties else []

    ray_options_model = None
    if ray_options is not None:
        try:
            ray_options_model = RayRemoteOptions.model_validate(ray_options)
        except pydantic.ValidationError as e:
            raise ValueError("Invalid ray_options") from e

    def decorator(
        func: Callable[..., Any],  # noqa: ANN401
    ) -> Callable[[tuple[Any, ...], dict[str, Any]], Any]:  # noqa: ANN401
        # If we were not given information on required/optional properties
        # or parameterization try to infer it
        # This function will log a critical error message and raise exception
        # if inference is required (because user did not provide explicit information)
        # but it could not be done (missing annotation, invalid annotation etc.)

        try:
            _optional_properties, _parameterization, _required_properties = (
                check_parameters_and_infer(
                    func=func,
                    _required_properties=required_properties,
                    _optional_properties=optional_properties,
                    _parameterization=parameterization,
                )
            )

            check_parameters_valid(
                func,
                _required_properties=required_properties,
                _optional_properties=_optional_properties,
            )
        except ValueError as error:
            raise ValueError(
                f"Unable to generate custom experiment for {func.__name__} via decorator.\n\t{error}"
            ) from error

        # Create an ExperimentModuleConf instance describing where the function is
        metadata["module"] = ExperimentModuleConf(
            moduleType=ModuleTypeEnum.EXPERIMENT,
            moduleName=func.__module__,
            moduleFunction=func.__name__,
        )

        # Create and store the Experiment instance
        experiment = Experiment(
            actuatorIdentifier="custom_experiments",
            identifier=func.__name__,
            requiredProperties=tuple(_required_properties),
            optionalProperties=tuple(_optional_properties),
            targetProperties=[
                AbstractPropertyDescriptor(identifier=p)
                for p in output_property_identifiers
            ],
            defaultParameterization=tuple(
                [
                    ConstitutivePropertyValue(
                        property=ConstitutivePropertyDescriptor(identifier=k), value=v
                    )
                    for k, v in _parameterization.items()
                ]
            ),
            deprecated=False,
            metadata=metadata,
        )
        func._experiment = experiment

        # Add the experiment to the module-level catalog
        _custom_experiments_catalog.addExperiment(experiment)

        from functools import wraps

        @wraps(func)
        def validated_func(
            *args: Any, **kwargs: Any  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            # Build property dict from either kwargs or args
            # Prefer kwargs, but support positional for backwards compatibility
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            param_dict = dict(bound_args.arguments)

            # Validate using SpacePoint and Experiment.validate_entity
            spoint = SpacePoint(entity=param_dict)
            entity = spoint.to_entity()
            if not experiment.validate_entity(entity, verbose=True):
                raise ValueError(
                    f"Arguments {param_dict} do not match required/optional properties for experiment '{experiment.identifier}'. "
                    f"See logs/stderr for reasons, or check experiment.requiredProperties/optionalProperties."
                )
            # Call the original with the unpacked arguments
            return func(*args, **kwargs)

        # Attach metadata to validated_func not func, so end users get the right attributes
        validated_func._decorator_required_properties = _required_properties
        validated_func._decorator_optional_properties = _optional_properties
        validated_func._decorator_parameterization = _parameterization
        validated_func._original_func = func
        validated_func._is_custom_experiment = True
        validated_func._experiment = experiment
        validated_func._use_ray = use_ray
        validated_func._ray_options = ray_options_model
        return validated_func

    return decorator


def load_custom_experiments_from_catalog_extensions(identifier: str) -> None:
    import importlib.resources
    import logging
    import pkgutil
    from pathlib import Path

    import yaml

    from orchestrator.modules.actuators.catalog import ActuatorCatalogExtension
    from orchestrator.modules.actuators.registry import (
        CATALOG_EXTENSIONS_CONFIGURATION_FILE_NAME,
    )

    try:
        import ado_actuators as plugins
    except ImportError:
        logging.getLogger("custom_experiments").info(
            "ado_actuators namespace package has not been created yet"
        )
        return

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
            for experiment in catalog_extension.experiments:
                if experiment.actuatorIdentifier == "custom_experiments":
                    _custom_experiments_catalog.addExperiment(experiment)
                else:
                    logger.warning(
                        f"Cannot add catalog extension for {experiment.actuatorIdentifier} - only custom_experiments supports catalog extensions"
                    )


def load_custom_experiments_from_entry_points() -> None:
    """
    Load custom experiments from entry points.

    This function searches for entry points under 'ado.custom_experiments' and loads
    any decorated functions from those modules.
    """

    import importlib.metadata

    # Get all entry points for ado.custom_experiments
    entry_points = importlib.metadata.entry_points()
    custom_experiment_groups = entry_points.select(group="ado.custom_experiments")
    for entry_point in custom_experiment_groups:
        try:
            entry_point.load()
        except ImportError as error:  # noqa: PERF203
            logging.getLogger("load_custom_experiments").warning(
                f"Unable to import custom experiments from {entry_point.value}: {error}"
            )
        except ValueError as error:
            logging.getLogger("load_custom_experiments").warning(
                f"Error when creating custom experiments defined in {entry_point.value}: {error}"
            )


def get_custom_experiments_catalog() -> (
    orchestrator.modules.actuators.catalog.ExperimentCatalog
):
    """
    Get the module-level catalog of custom experiments.

    Returns:
        The ExperimentCatalog containing all registered custom experiments
    """
    return _custom_experiments_catalog


def _call_decorated_custom_experiment(
    function: Callable, target_experiment: Experiment, entity: Entity
) -> list[ObservedPropertyValue]:

    # Build input dict using experiment values from entity
    input_values = target_experiment.propertyValuesFromEntity(entity)
    # Call function with unpacked parameters
    result_dict = function(**input_values)

    # Drop all keys not in output_property_identifiers
    allowed_keys = {tp.identifier for tp in target_experiment.targetProperties}
    filtered_result = {k: v for k, v in result_dict.items() if k in allowed_keys}
    dropped_keys = {k: v for k, v in result_dict.items() if k not in allowed_keys}
    if dropped_keys:
        import logging

        logger = logging.getLogger("custom_experiments")
        logger.debug(
            f"Dropped keys from return of {target_experiment.identifier}: {dropped_keys}"
        )

    # Check at least one valid output property was returned
    if not filtered_result:
        raise ValueError(
            f"No valid output properties (from set {allowed_keys}) returned by experiment '{target_experiment.identifier}'"
        )

    # Create observed property values
    observed_property_values = []
    for property_identifier, value in filtered_result.items():
        observed_property = target_experiment.observedPropertyForTargetIdentifier(
            property_identifier
        )
        if not observed_property:
            raise ValueError(
                f"{target_experiment.identifier} returned a property called {property_identifier}, however "
                f"the experiment definition does not define an output property with this name"
            )
        observed_property_value = ObservedPropertyValue(
            property=observed_property, value=value
        )
        observed_property_values.append(observed_property_value)

    return observed_property_values


def _call_legacy_custom_experiment(
    function: Callable,
    target_experiment: Experiment,
    entity: Entity,
    parameters: dict | None = None,
) -> list[ObservedPropertyValue]:
    # For legacy case or other functions, check for parameters kwarg else pass entity/experiment
    func_signature = inspect.signature(function)
    if "parameters" in func_signature.parameters:
        values = function(entity, target_experiment, parameters=parameters)
    else:
        values = function(entity, target_experiment)

    return values


def custom_experiment_executor(
    function: Callable,
    parameters: dict,
    measurement_request: MeasurementRequest,
    target_experiment: Experiment,
    queue: MeasurementQueue,
) -> None:
    """
    :param function: The function to call
    :param parameters: The custom parameters to the function
    :param measurement_request: The entity and custom experiment to be measured
    :param target_experiment: The experiment to execute.
        Required as the measurementRequest only includes an ExperimentReference
    :param queue: The queue to put the result on
    :return:
    """

    measurement_results = []
    for entity in measurement_request.entities:
        try:
            # Check if this is a custom experiment decorated function
            if getattr(function, "_is_custom_experiment", False):
                values = _call_decorated_custom_experiment(
                    function=function,
                    target_experiment=target_experiment,
                    entity=entity,
                )
            else:
                values = _call_legacy_custom_experiment(
                    function=function,
                    target_experiment=target_experiment,
                    entity=entity,
                    parameters=parameters,
                )

            if len(values) > 0:
                measurement_result = ValidMeasurementResult(
                    entityIdentifier=entity.identifier, measurements=values
                )
                measurement_results.append(measurement_result)
        except Exception as error:  # noqa: PERF203
            measurement_result = InvalidMeasurementResult(
                entityIdentifier=entity.identifier,
                experimentReference=target_experiment.reference,
                reason=f"Unexpected exception: {error}",
            )
            measurement_results.append(measurement_result)

    if len(measurement_results) > 0:
        measurement_request.measurements = measurement_results
        measurement_request.status = MeasurementRequestStateEnum.SUCCESS
    else:
        measurement_request.status = MeasurementRequestStateEnum.FAILED

    queue.put(measurement_request, block=False)


@ray.remote
class CustomExperiments(ActuatorBase):
    identifier = "custom_experiments"

    """Actuator for applying user supplied custom experiments
    """

    def __init__(self, queue: "MeasurementQueue", params: dict | None = None) -> None:
        """

        :param queue: The MeasurementQueue instance
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

            function = None
            if module := experiment.metadata.get("module"):
                experiment_module_conf = ExperimentModuleConf.model_validate(module)
                function = (
                    load_module_class_or_function(experiment_module_conf)
                    if experiment_module_conf
                    else None
                )

            if function:
                self._functionImplementations[experiment.identifier] = function
                self.log.info(
                    f"Experiment name: {experiment.identifier}. "
                    f"Function Implementation: {self._functionImplementations[experiment.identifier]}. "
                    f"Experiment: {experiment}"
                )
            else:
                self.log.error(
                    f"Experiment in custom_experiment catalog is missing required metadata (either experiment_function or module): {experiment}"
                )

        self.log.debug("Completed init")

    def loadedExperiment(
        self,
        experimentReference: ExperimentReference,
    ) -> bool:

        return (
            self._functionImplementations.get(experimentReference.experimentIdentifier)
            is not None
        )

    def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:

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

        if experimentReference.parameterization:
            targetExperiment = ParameterizedExperiment(
                parameterization=experimentReference.parameterization,
                **targetExperiment.model_dump(),
            )

        # Fetch custom_experiment function for this identifier
        fn = self._functionImplementations[
            request.experimentReference.experimentIdentifier
        ]
        use_ray = getattr(fn, "_use_ray", True)
        ray_options_model = getattr(fn, "_ray_options", None)
        if use_ray:
            remote_kwargs = (
                ray_options_model.model_dump(exclude_none=True)
                if getattr(fn, "_ray_options", None)
                else {}
            )
            # Dispatch as Ray task. Pass ray options if present.
            ray.remote(custom_experiment_executor, **remote_kwargs).remote(
                fn,
                self._catalog.experimentForReference(
                    request.experimentReference
                ).metadata.get("parameters", {}),
                request,
                targetExperiment,
                self._stateUpdateQueue,
            )
        else:
            custom_experiment_executor(
                fn,
                self._catalog.experimentForReference(
                    request.experimentReference
                ).metadata.get("parameters", {}),
                request,
                targetExperiment,
                self._stateUpdateQueue,
            )
        return [requestid]

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:

        load_custom_experiments_from_catalog_extensions(cls.identifier)
        # Load custom experiments from entry points before returning catalog
        load_custom_experiments_from_entry_points()
        return get_custom_experiments_catalog()

    def current_catalog(
        self,
    ) -> orchestrator.modules.actuators.catalog.ExperimentCatalog:
        return self._catalog
