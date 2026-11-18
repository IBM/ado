# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import inspect
import logging
import typing
import uuid
from collections.abc import Callable
from typing import Annotated, Any

import pydantic
import ray

import ado.modules.actuators.catalog
from ado.core.actuatorconfiguration.config import GenericActuatorParameters
from ado.modules.actuators.errors import UnknownExperimentError
from ado.modules.actuators.executor_supervisor import (
    ExperimentExecutorSupervisorParameters,
)
from ado.modules.actuators.measurement_queue import MeasurementQueue, NullQueue
from ado.modules.actuators.standard import StandardActuator
from ado.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
)
from ado.schema.entity import Entity
from ado.schema.experiment import Experiment
from ado.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from ado.schema.point import SpacePoint
from ado.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference
from ado.schema.request import MeasurementRequest
from ado.schema.result import InvalidMeasurementResult, ValidMeasurementResult
from ado.utilities.environment import enable_ray_actor_coverage
from ado.utilities.logging import configure_logging
from ado.utilities.support import compute_measurement_status

configure_logging()

# Module-level catalog for custom experiments
_custom_experiments_catalog = ado.modules.actuators.catalog.ExperimentCatalog(
    catalogIdentifier="CustomExperiments"
)

# Maps experiment identifier to the callable most recently decorated with
# @custom_experiment under that identifier
_custom_experiment_functions: dict[str, Callable[..., Any]] = {}


class RayRemoteOptions(pydantic.BaseModel):
    num_cpus: float | None = None
    num_gpus: float | None = None
    resources: dict | None = None
    runtime_env: dict | None = None


# Ray runtime_env keys that provision an isolated Python environment for the task.
_SANDBOX_VENV_RUNTIME_ENV_KEYS = frozenset({"pip", "conda", "uv"})


def runtime_env_creates_sandbox_venv(runtime_env: dict | None) -> bool:
    """Return True if ``runtime_env`` asks Ray to create a sandboxed Python env.

    Args:
        runtime_env: A Ray ``runtime_env`` mapping, or None.

    Returns:
        True when ``runtime_env`` includes package-manager keys that create a
        task-local virtual environment (``pip``, ``conda``, or ``uv``).
    """
    if not runtime_env:
        return False
    return bool(_SANDBOX_VENV_RUNTIME_ENV_KEYS.intersection(runtime_env))


class ExperimentModuleConf(ModuleConf):
    moduleType: Annotated[ModuleTypeEnum, pydantic.Field()] = ModuleTypeEnum.EXPERIMENT


def _infer_domain_and_property(
    identifier: str,
    annotation: type,
    default: Any,  # noqa: ANN401
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

    from ado.schema.domain import PropertyDomain, VariableTypeEnum

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
        param = param_map.get(prop.identifier)
        if param and param.default is not inspect.Parameter.empty:
            results[prop.identifier] = param.default
        else:
            missing.append(prop.identifier)
    if missing:
        raise ValueError(f"Parameterization missing for: {missing}")
    return results


def _parameter_name_for_property(
    prop: ConstitutiveProperty | ObservedProperty,
) -> str:
    """Return the name of the function parameter that receives a property's value.

    Args:
        prop: A required or optional property of a custom experiment.

    Returns:
        The target property identifier for an ObservedProperty, otherwise the
        property identifier.
    """
    if isinstance(prop, ObservedProperty):
        return prop.targetProperty.identifier
    return prop.identifier


def derive_optional_properties_and_parameterization(
    func: Callable,
    required_properties: list[ConstitutiveProperty | ObservedProperty],
) -> tuple[list[ConstitutiveProperty], dict[str, Any]]:
    """This function derives the optional properties and their parameterization from the function signature.

    The optional properties are the keyword parameters of the function that do not receive a required property.
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
    required_parameter_names = {
        _parameter_name_for_property(prop) for prop in required_properties
    }
    for param in inspect.signature(func).parameters.values():
        if param.name in required_parameter_names:
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
    dict | dict[str, Any] | None,
    list[ConstitutiveProperty],
]:
    logger = logging.getLogger("custom_experiment_decorator")

    # Set up dynamic optional_properties and parameterization if none were provided
    _optional_properties = _optional_properties or []

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
    """Check each property maps to a distinct parameter of func.

    An ObservedProperty maps to the parameter named by its target property
    identifier. Other properties map to the parameter named by their identifier.

    Args:
        func: The decorated function.
        _required_properties: The experiment's required properties.
        _optional_properties: The experiment's optional properties.

    Raises:
        ValueError: If two properties map to the same parameter name, or a
            property's parameter is not in func's signature.
    """
    func_param_names = set(inspect.signature(func).parameters.keys())
    experiment_properties = [
        *(_required_properties or []),
        *(_optional_properties or []),
    ]

    properties_by_parameter_name: dict[str, list[str]] = {}
    for prop in experiment_properties:
        properties_by_parameter_name.setdefault(
            _parameter_name_for_property(prop), []
        ).append(prop.identifier)
    colliding_properties = {
        parameter_name: property_identifiers
        for parameter_name, property_identifiers in properties_by_parameter_name.items()
        if len(property_identifiers) > 1
    }
    if colliding_properties:
        raise ValueError(
            f"Properties of {func.__name__} map to the same function parameter: {colliding_properties}"
        )

    experiment_parameter_names = set(properties_by_parameter_name)
    if not experiment_parameter_names.issubset(func_param_names):
        raise ValueError(
            f"{func.__name__} parameter names {func_param_names} must include the parameter names "
            f"of all properties {experiment_parameter_names}. "
            f"Missing parameters: {experiment_parameter_names - func_param_names}"
        )


def custom_experiment(
    output_property_identifiers: list[str],
    required_properties: list[ConstitutiveProperty | ObservedProperty] | None = None,
    optional_properties: list[ConstitutiveProperty] | None = None,
    parameterization: dict[str, Any] | None = None,  # noqa: ANN401
    metadata: dict[str, Any] | None = None,  # noqa: ANN401
    use_ray: bool = True,
    ray_options: dict | None = None,
    version: str | None = None,
) -> Callable[
    [Callable[..., Any]],  # noqa: ANN401
    Callable[[tuple[Any, ...], dict[str, Any]], Any],  # noqa: ANN401
]:
    """Decorator for custom experiment functions.

    Args:
        required_properties: List of ConstitutiveProperty and ObservedProperty instances
            that are required input values. The value of a ConstitutiveProperty is passed
            to the function parameter named by its identifier. The value of an
            ObservedProperty is passed to the function parameter named by its target
            property identifier.
        output_property_identifiers: List of strings identifying the output property names.
        optional_properties: List of ConstitutiveProperty instances that are optional input values.
        parameterization: Dictionary mapping optional property identifiers to their default values.
        metadata: Metadata for the experiment.
        use_ray: If True the CustomExperiments actuator will launch the experiment as a ray remote task.
        ray_options: A dictionary containing ray remote task options.
            The keys and allowed values are defined by RayRemoteOptions.
        version: Algorithm version string in strict SemVer format (``"MAJOR.MINOR.PATCH"``).
            Can be None.

    Returns:
        A decorator that wraps a function to work with ado's custom experiment system.

    Raises:
        ValueError: If Unable to generate custom experiment via decorator e.g.
            - No required properties provided, and they could not be derived from signature.
            - Optional properties provided but parameterization could not be derived from signature.
            - No optional properties provided and they could not be derived from signature.
            - Function parameter names did not include the parameter name of every property.
            - Two properties map to the same function parameter.

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
        output_property_identifiers=["density"],
        version="1.0.0",
    )
    def calculate_density(mass, volume):
        density_value = mass / volume if volume else None
        return {
            "density": density_value
        }
    """

    metadata = metadata or {}
    required_properties = required_properties or []

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
            version=version,
        )
        func._experiment = experiment

        # Add the experiment to the module-level catalog
        _custom_experiments_catalog.addExperiment(experiment)

        from functools import wraps

        @wraps(func)
        def validated_func(
            *args: Any,  # noqa: ANN401
            **kwargs: Any,  # noqa: ANN401
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
        _custom_experiment_functions[experiment.identifier] = validated_func
        return validated_func

    return decorator


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


def get_custom_experiments_catalog() -> ado.modules.actuators.catalog.ExperimentCatalog:
    """
    Get the module-level catalog of custom experiments.

    Returns:
        The ExperimentCatalog containing all registered custom experiments
    """
    return _custom_experiments_catalog


def _call_decorated_custom_experiment(
    function: Callable, target_experiment: Experiment, entity: Entity
) -> list[ObservedPropertyValue]:
    """Call a decorated custom experiment function on one entity.

    Each input value is passed as a keyword argument. The value of a required
    observed property is passed under its target property identifier.

    Args:
        function: The decorated custom experiment function.
        target_experiment: The experiment the function implements.
        entity: The entity to measure.

    Returns:
        The measured observed property values.

    Raises:
        ValueError: If the entity has more than one value for a required observed
            property, is missing an input value, or the function returns no
            target property values.
    """

    for observed_property in target_experiment.requiredObservedProperties:
        number_of_values = len(entity.valuesForProperty(observed_property))
        if number_of_values > 1:
            raise ValueError(
                f"Entity {entity.identifier} has {number_of_values} values for required "
                f"observed property {observed_property.identifier}. "
                f"Experiment '{target_experiment.identifier}' requires exactly one."
            )

    # Build input dict using experiment values from entity
    input_values = target_experiment.propertyValuesFromEntity(entity, target=True)
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


def custom_experiment_executor(
    function: Callable,
    measurement_request: MeasurementRequest,
    target_experiment: Experiment,
    queue: MeasurementQueue | NullQueue,
    custom_experiments_actor: "CustomExperimentsActor | None" = None,
) -> MeasurementRequest:
    """Run a custom experiment and place the completed request on ``queue``.

    Args:
        function: The decorated custom experiment function to call.
        measurement_request: The entity and custom experiment to be measured.
        target_experiment: The experiment to execute. Required as the
            measurementRequest only includes an ExperimentReference.
        queue: The queue to put the result on. Use ``NullQueue`` when the
            caller only needs the returned request.
        custom_experiments_actor: Optional handle to CustomExperiments actuator
            to notify after the result is queued.

    Returns:
        The completed ``MeasurementRequest`` (also written to ``queue``).
    """

    measurement_results = []
    for entity in measurement_request.entities:
        try:
            values = _call_decorated_custom_experiment(
                function=function,
                target_experiment=target_experiment,
                entity=entity,
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
    measurement_request.status = compute_measurement_status(measurement_results)

    queue.put(measurement_request, block=False)
    if custom_experiments_actor:
        custom_experiments_actor.mark_measurement_request_completed.remote(
            measurement_request.requestid
        )
    return measurement_request


def _custom_experiment_executor_remote(
    fn: Callable[..., Any],  # noqa: ANN401
) -> ray.remote_function.RemoteFunction:
    """Build the Ray remote function that runs a custom experiment.

    Args:
        fn: Decorated custom experiment function. Its ``_ray_options``
            attribute supplies the Ray task options when set.

    Returns:
        A remote function wrapping ``custom_experiment_executor``.
    """
    ray_options_model = getattr(fn, "_ray_options", None)
    remote_kwargs = (
        ray_options_model.model_dump(exclude_none=True)
        if ray_options_model is not None
        else {}
    )
    if remote_kwargs:
        return ray.remote(**remote_kwargs)(custom_experiment_executor)
    return ray.remote(custom_experiment_executor)


class CustomExperimentsParameters(ExperimentExecutorSupervisorParameters):
    """Configuration parameters for the CustomExperiments actuator."""

    model_config = pydantic.ConfigDict(extra="forbid")


class CustomExperiments(StandardActuator):
    """Actuator for applying user supplied custom experiments.

    Subclasses :class:`~ado.modules.actuators.standard.StandardActuator` for
    shared queue/supervisor setup and the ``execute`` / ``submit`` API surface.
    All entities in a request are measured by a single call to
    ``custom_experiment_executor``. That call runs either in the current
    process or as one Ray remote task configured with the experiment's
    ``ray_options``. ``execute`` selects between them with its ``use_ray``
    argument, and ``submit`` with the experiment's ``use_ray`` setting.
    """

    identifier = "custom_experiments"
    parameters_class = CustomExperimentsParameters

    def __init__(
        self,
        queue: MeasurementQueue | NullQueue | None = None,
        params: dict | None = None,
    ) -> None:
        """Initialise the actuator and load custom experiment implementations.

        Args:
            queue: MeasurementQueue for submit()-based async operation.
                If None, a NullQueue is used — suitable for execute()-only use.
            params: Actuator configuration parameters.

        Raises:
            ValueError: If a catalog experiment has no registered function.
        """

        enable_ray_actor_coverage("custom_experiments")
        super().__init__(
            queue=queue if queue is not None else NullQueue(), params=params
        )

        self.log.debug(f"Queue is {self._stateUpdateQueue}")
        self.log.debug(f"Params are {self._parameters}")

        # Use the module-level catalog by calling the class method
        self._catalog = type(self).catalog()
        self.log.debug(f"Catalog is {self._catalog}")

        self._functionImplementations = {}
        for experiment in self._catalog.experiments:
            function = _custom_experiment_functions.get(experiment.identifier)
            if function is None:
                raise ValueError(
                    f"Custom experiment '{experiment.identifier}' is in the catalog "
                    f"but has no registered function."
                )
            self._functionImplementations[experiment.identifier] = function
            self.log.info(
                f"Experiment name: {experiment.identifier}. "
                f"Function Implementation: {function}. "
                f"Experiment: {experiment}"
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

    def experiment_requires_ray_sandbox_venv(
        self,
        experimentReference: ExperimentReference,
    ) -> bool:
        """Return True if the experiment needs a Ray-managed sandbox venv.

        Experiments whose ``ray_options.runtime_env`` includes ``pip``,
        ``conda``, or ``uv`` cannot run via ``execute(use_ray=False)``.

        Args:
            experimentReference: Reference to the experiment to inspect.

        Returns:
            True when the decorated experiment uses Ray and its runtime
            environment creates a sandboxed Python environment.
        """
        fn = self._functionImplementations.get(experimentReference.experimentIdentifier)
        if fn is None:
            return False
        if not getattr(fn, "_use_ray", True):
            return False
        ray_options = getattr(fn, "_ray_options", None)
        if ray_options is None:
            return False
        return runtime_env_creates_sandbox_venv(ray_options.runtime_env)

    def _resolve_experiment_and_function(
        self,
        experimentReference: ExperimentReference,
    ) -> tuple[Experiment, Callable]:
        """Resolve the experiment and its function.

        Args:
            experimentReference: Reference to the experiment to run.

        Returns:
            Tuple of ``(target_experiment, function)``.

        Raises:
            ValueError: If the experiment is unknown.
        """
        try:
            target_experiment = self._catalog.experimentForReference(
                experimentReference, resolve=True
            )
        # ExperimentVersionMismatchError can't be raised as we don't use FQI
        # DeprecatedExperimentError can't be raised as there is no custom experiment deprecation
        except UnknownExperimentError as error:
            if self._catalog.experiments:
                raise ValueError(
                    f"Requested experiments {experimentReference} is not in the CustomExperiments actuator catalog. "
                    f"Known experiments are {self._catalog.experiment_major_version_identifiers}"
                ) from error
            raise ValueError(
                f"Requested experiments {experimentReference} is not in the CustomExperiments actuator catalog (which is empty). "
            ) from error

        fn = self._functionImplementations[experimentReference.experimentIdentifier]
        return target_experiment, fn

    def execute(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
        use_ray: bool = False,
    ) -> MeasurementRequest:
        """Run the experiment and return the completed MeasurementRequest.

        Args:
            entities: Entities to measure.
            experimentReference: Reference to the experiment to run.
            requesterid: ID of the requesting operation.
            requestIndex: Index of this request within the operation.
            use_ray: When True, dispatch execution to a Ray remote task
                (honouring the experiment's ``ray_options``).

        Returns:
            A completed MeasurementRequest with measurements and status set.
        """
        target_experiment, fn = self._resolve_experiment_and_function(
            experimentReference
        )
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=str(uuid.uuid4())[:6],
        )

        if not use_ray:
            return custom_experiment_executor(
                fn,
                request,
                target_experiment,
                NullQueue(),
            )

        remote_fn = _custom_experiment_executor_remote(fn)
        return ray.get(
            remote_fn.remote(
                fn,
                request,
                target_experiment,
                NullQueue(),
                None,
            )
        )

    def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:
        """Submit the experiment and place the completed request on the queue.

        If the experiment's ``use_ray`` setting is True (the default), the
        request is dispatched as a Ray remote task configured with the
        experiment's ``ray_options`` and this method returns without waiting
        for it to finish. If ``use_ray`` is False, the request is executed
        synchronously in the current process before this method returns.

        Args:
            entities: Entities to measure.
            experimentReference: Reference to the experiment to run.
            requesterid: ID of the requesting operation.
            requestIndex: Index of this request within the operation.

        Returns:
            A list containing the single request ID.
        """

        self.log.debug(
            f"Received a request to measure {experimentReference} on {[e.identifier for e in entities]}"
        )

        target_experiment, fn = self._resolve_experiment_and_function(
            experimentReference
        )

        requestid = str(uuid.uuid4())[:6]
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=requestid,
        )

        self.log.debug(f"Create measurement request {request}")

        use_ray = getattr(fn, "_use_ray", True)
        if use_ray:
            # Dispatch as Ray task. Pass ray options if present.
            # Pass the actor handle (not self) so the executor can call
            # mark_measurement_request_completed.remote() without pickling the actor object.
            # self is not serialisable (contains threading.Lock from the supervisor).
            try:
                _notifier = ray.get_runtime_context().current_actor
            except Exception:
                _notifier = None
            remote_fn = _custom_experiment_executor_remote(fn)
            executor_ref = remote_fn.remote(
                fn,
                request,
                target_experiment,
                self._stateUpdateQueue,
                _notifier,
            )
            self._launch_supervisor.supervise_experiment_executor(request, executor_ref)
        else:
            custom_experiment_executor(
                fn,
                request,
                target_experiment,
                self._stateUpdateQueue,
            )
        return [requestid]

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ado.modules.actuators.catalog.ExperimentCatalog:

        # Load custom experiments from entry points before returning catalog
        load_custom_experiments_from_entry_points()
        return get_custom_experiments_catalog()

    def current_catalog(
        self,
    ) -> ado.modules.actuators.catalog.ExperimentCatalog:
        return self._catalog


if typing.TYPE_CHECKING:
    from ray.actor import ActorHandle

    CustomExperimentsActor = type[ActorHandle[CustomExperiments]]
