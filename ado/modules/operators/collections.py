# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import inspect
import logging
import typing
from typing import Annotated

import pydantic

import ado.core.operation.config
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.metadata import PackageProvenance
from ado.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    GenericOperatorParameters,
    OperatorMetadata,
    OperatorReference,
)
from ado.core.operation.inputs import resource_inputs_from_operator_function
from ado.core.resources import ADOResourcePropertyDescriptor
from ado.metastore.sqlstore import SQLStore
from ado.modules.operators.base import (
    DiscoveryOperationBase,
    DiscoverySpaceSubscribingDiscoveryOperation,
    OperationOutput,
    OperatorFunction,
    validate_operator_registration,
)
from ado.modules.operators.errors import OperatorVersionMismatchError

moduleLog = logging.getLogger("operation_collections")


def _warn_if_operator_name_reused(
    collection_label: str, name: str, operators: dict[str, OperatorMetadata]
) -> None:
    """Log a warning when registering under a name that is already in use."""
    if name in operators:
        moduleLog.warning(
            "Operator %r is already registered in %s; replacing the existing entry",
            name,
            collection_label,
        )


class OperatorCollection(pydantic.BaseModel):
    """A registry of operators of a single discovery operation type.

    Operators are added via the decorator functions (e.g. ``characterize_operation``,
    ``explore_operation``, ``summarize_operation``).  Each registered name maps to an
    :class:`~ado.core.operation.config.OperatorMetadata` instance that
    carries the function, version, description, configuration model,
    example configuration, and optional actor class.

    Attributes:
        type: The discovery operation type all operators in this collection belong to.
        operators: Mapping of operator name to :class:`~ado.core.operation.config.OperatorMetadata` instance.
    """

    type: DiscoveryOperationEnum
    operators: Annotated[
        dict[str, OperatorMetadata], pydantic.Field(default_factory=dict)
    ]

    def list_operators(self) -> list[str]:
        """Returns all registered operator names."""
        return list(self.operators.keys())

    def __getattr__(self, item: str) -> OperatorFunction | None:
        """Returns the operator function for the given registered name.

        Args:
            item: Registered operator name.

        Returns:
            The callable registered under that name
            Or None if no callable registered.

        Raises:
            AttributeError: If no operator is registered with that name.
        """
        if item in self.operators:
            return self.operators[item].function
        raise AttributeError(f"Unknown attribute {item}")


characterize = OperatorCollection(
    type=ado.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE
)
explore = OperatorCollection(
    type=ado.core.operation.config.DiscoveryOperationEnum.EXPLORE
)
modify = OperatorCollection(
    type=ado.core.operation.config.DiscoveryOperationEnum.MODIFY
)
export = OperatorCollection(
    type=ado.core.operation.config.DiscoveryOperationEnum.EXPORT
)
compare = OperatorCollection(
    type=ado.core.operation.config.DiscoveryOperationEnum.COMPARE
)
fuse = OperatorCollection(type=ado.core.operation.config.DiscoveryOperationEnum.FUSE)
study = OperatorCollection(type=ado.core.operation.config.DiscoveryOperationEnum.STUDY)
learn = OperatorCollection(type=ado.core.operation.config.DiscoveryOperationEnum.LEARN)
summarize = OperatorCollection(
    type=ado.core.operation.config.DiscoveryOperationEnum.SUMMARIZE
)
operationCollectionMap = {
    ado.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE: characterize,
    ado.core.operation.config.DiscoveryOperationEnum.EXPLORE: explore,
    ado.core.operation.config.DiscoveryOperationEnum.MODIFY: modify,
    ado.core.operation.config.DiscoveryOperationEnum.EXPORT: export,
    ado.core.operation.config.DiscoveryOperationEnum.COMPARE: compare,
    ado.core.operation.config.DiscoveryOperationEnum.STUDY: study,
    ado.core.operation.config.DiscoveryOperationEnum.LEARN: learn,
    ado.core.operation.config.DiscoveryOperationEnum.FUSE: fuse,
    ado.core.operation.config.DiscoveryOperationEnum.SUMMARIZE: summarize,
}

#
# Decorators for registering operator functions
#


def _make_general_orchestration_wrapper(
    *,
    collection: OperatorCollection,
    name: str,
    user_fn: typing.Callable[..., OperationOutput],
    required_resource_inputs: list[ADOResourcePropertyDescriptor],
    configuration_model: type[GenericOperatorParameters],
) -> typing.Callable[..., OperationOutput]:
    """Build a wrapper that starts orchestration and then runs *user_fn*.

    The wrapper accepts the same call shape as *user_fn*: resource inputs,
    then ``operationInfo``, then ``parameters``.

    Args:
        collection: Registry the operator is stored in.
        name: Canonical operator name.
        user_fn: Operator implementation used to bind the call.
        required_resource_inputs: Resource inputs deduced from *user_fn*.
        configuration_model: Model used to validate ``parameters``.

    Returns:
        A wrapper with the same call shape as *user_fn*.
    """
    input_ids = [descriptor.identifier for descriptor in required_resource_inputs]
    user_sig = inspect.signature(user_fn)

    @functools.wraps(user_fn)
    def wrapper(*args: object, **kwargs: object) -> OperationOutput:
        bound = user_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)

        from ado.core.operation.inputs import (
            resolve_operation_project_context,
            verify_inputs_exist_in_metastore,
        )
        from ado.modules.operators._general_orchestration import (
            orchestrate_general_operation,
        )

        missing = [input_id for input_id in input_ids if input_id not in arguments]
        if missing:
            raise ValueError(
                f"Operator {name!r} missing required resource input(s): {missing!r}."
            )

        inputs = {input_id: arguments.pop(input_id) for input_id in input_ids}
        operation_info = arguments.pop("operationInfo", None) or FunctionOperationInfo()
        raw_parameters = arguments.pop("parameters", None)
        if arguments:
            raise ValueError(
                f"Operator {name!r} received unexpected arguments: {list(arguments)!r}."
            )

        parameters = configuration_model.model_validate(raw_parameters)
        project_context = resolve_operation_project_context(operation_info, inputs)  # type: ignore[arg-type]
        if operation_info.projectContext is None:
            operation_info = operation_info.model_copy(
                update={"projectContext": project_context}
            )
        metastore = SQLStore(project_context=project_context)
        verify_inputs_exist_in_metastore(inputs, metastore)  # type: ignore[arg-type]
        return orchestrate_general_operation(
            operator_metadata=collection.operators[name],
            inputs=inputs,  # type: ignore[arg-type]
            operation_parameters=parameters,
            operation_info=operation_info,
            metastore=metastore,
        )

    return wrapper


def _register_general_operator(
    *,
    collection: OperatorCollection,
    collection_label: str,
    operation_type: DiscoveryOperationEnum,
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None,
    func: OperatorFunction,
) -> OperatorFunction:
    """Validate, wrap, and register a non-explore operator.

    Resource inputs are deduced from *func*.

    Args:
        collection: Registry to store the operator in.
        collection_label: Name used in reuse warnings.
        operation_type: Collection the operator belongs to.
        name: Canonical operator name.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.
        func: Operator implementation.

    Returns:
        The registered wrapper.
    """
    required_resource_inputs = resource_inputs_from_operator_function(func)
    wrapper = _make_general_orchestration_wrapper(
        collection=collection,
        name=name,
        user_fn=func,
        required_resource_inputs=required_resource_inputs,
        configuration_model=configuration_model,
    )
    validate_operator_registration(
        user_fn=func,
        required_resource_inputs=required_resource_inputs,
        operation_type=operation_type,
        configuration_model=configuration_model,
        stored_fn=wrapper,
    )
    _warn_if_operator_name_reused(collection_label, name, collection.operators)
    collection.operators[name] = OperatorMetadata(
        name=name,
        function=wrapper,
        version=version,
        description=description,
        configuration_model=configuration_model,
        example_configuration=example_configuration,
        type=operation_type,
        provenance=PackageProvenance.from_module_name(func.__module__),
        required_resource_inputs=tuple(required_resource_inputs),
    )
    return typing.cast("OperatorFunction", wrapper)


def characterize_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a characterize operation.

    Resource inputs are deduced from the decorated function's signature.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        return _register_general_operator(
            collection=characterize,
            collection_label="characterize",
            operation_type=DiscoveryOperationEnum.CHARACTERIZE,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            func=func,
        )

    return _register


def _validate_explore_cls(t: type, metadata: OperatorMetadata) -> None:
    """Validate a class-decorated explore operator and its metadata.

    Args:
        t: The decorated class.
        metadata: The :class:`~ado.core.operation.config.OperatorMetadata`
            returned by ``t.operator_metadata()``.

    Raises:
        TypeError: If ``t`` is not a
            :class:`~ado.modules.operators.base.DiscoverySpaceSubscribingDiscoveryOperation`
            subclass, or if ``metadata.cls`` is set to a class other than ``t``.
    """
    if not issubclass(t, DiscoverySpaceSubscribingDiscoveryOperation):
        raise TypeError(
            f"@explore_operation: {t.__name__} must be a subclass of "
            "DiscoverySpaceSubscribingDiscoveryOperation (e.g. subclass "
            "`Explore` or another discovery operation that subscribes to the space)."
        )
    if metadata.cls is not None and metadata.cls is not t:
        raise TypeError(
            f"@explore_operation on {t.__name__}: operator_metadata().cls is "
            f"{metadata.cls!r} but the decorated class is {t!r}. "
            "Leave cls as None in operator_metadata() — the decorator sets it."
        )


def explore_operation(
    cls: "type[DiscoveryOperationBase]",
) -> "type[DiscoveryOperationBase]":
    """Decorator that registers an explore operator class.

    All metadata is sourced from the class's ``operator_metadata()``
    classmethod.  The decorator generates an :class:`OperatorFunction`,
    validates its signature, registers it in the explore collection, and
    returns the **original class unchanged**::

        @explore_operation
        class MyOp(Explore):
            @classmethod
            def operator_metadata(cls) -> OperatorMetadata:
                return OperatorMetadata(
                    name="my_op",
                    version="0.1.0",
                    configuration_model=MyOpParameters,
                    example_configuration=MyOpParameters(),
                    type=DiscoveryOperationEnum.EXPLORE,
                )

            async def run(self) -> OperationOutput | None: ...

    The generated operator function is accessible via
    ``explore.operators[name].function``; the class name continues to refer
    to the class itself.

    Args:
        cls: The operator class to register.  Must be a subclass of
            :class:`~ado.modules.operators.base.DiscoverySpaceSubscribingDiscoveryOperation`
            and must implement ``operator_metadata()``.

    Returns:
        *cls* unchanged.

    Raises:
        NotImplementedError: If ``cls.operator_metadata()`` is not implemented.
        TypeError: If ``cls`` fails :func:`_validate_explore_cls`.
    """
    metadata = cls.operator_metadata()
    _validate_explore_cls(cls, metadata)
    op_name = metadata.name

    configuration_model = metadata.configuration_model

    def _generated(
        discoverySpace: DiscoverySpace,
        operationInfo: FunctionOperationInfo | None = None,
        *,
        parameters: GenericOperatorParameters,
    ) -> OperationOutput:
        from ado.core.operation.inputs import (
            resolve_operation_project_context,
            verify_inputs_exist_in_metastore,
        )
        from ado.modules.operators.orchestrate import (
            orchestrate_explore_operation,
        )

        inputs = {"discoverySpace": discoverySpace}
        operation_info = operationInfo or FunctionOperationInfo()
        project_context = resolve_operation_project_context(operation_info, inputs)
        if operation_info.projectContext is None:
            operation_info = operation_info.model_copy(
                update={"projectContext": project_context}
            )
        metastore = SQLStore(project_context=project_context)
        verify_inputs_exist_in_metastore(inputs, metastore)
        return orchestrate_explore_operation(
            operator_metadata=explore.operators[op_name],
            inputs=inputs,
            parameters=configuration_model.model_validate(parameters),
            operation_info=operation_info,
        )

    _generated.__annotations__["parameters"] = configuration_model
    _generated.__name__ = op_name
    _generated.__qualname__ = op_name
    stored_inputs = resource_inputs_from_operator_function(_generated)
    validate_operator_registration(
        user_fn=_generated,
        required_resource_inputs=stored_inputs,
        operation_type=DiscoveryOperationEnum.EXPLORE,
        configuration_model=configuration_model,
    )
    _generated = typing.cast("OperatorFunction", _generated)
    _warn_if_operator_name_reused("explore", op_name, explore.operators)
    explore.operators[op_name] = metadata.model_copy(
        update={
            "function": _generated,
            "cls": cls,
            "provenance": PackageProvenance.from_module_name(cls.__module__),
            "required_resource_inputs": tuple(stored_inputs),
        }
    )
    return cls


def modify_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a modify operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        return _register_general_operator(
            collection=modify,
            collection_label="modify",
            operation_type=DiscoveryOperationEnum.MODIFY,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            func=func,
        )

    return _register


def export_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as an export operation.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        return _register_general_operator(
            collection=export,
            collection_label="export",
            operation_type=DiscoveryOperationEnum.EXPORT,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            func=func,
        )

    return _register


def compare_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a compare operation.

    A compare operator takes at least two discovery-space or data-container
    inputs. Resource inputs are deduced from the decorated function.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        return _register_general_operator(
            collection=compare,
            collection_label="compare",
            operation_type=DiscoveryOperationEnum.COMPARE,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            func=func,
        )

    return _register


def fuse_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a fuse operation.

    A fuse operator takes at least two discovery-space inputs. Resource inputs
    are deduced from the decorated function.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        return _register_general_operator(
            collection=fuse,
            collection_label="fuse",
            operation_type=DiscoveryOperationEnum.FUSE,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            func=func,
        )

        return _register


def summarize_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a summarize operation.

    A summarize operator takes at least one list of discovery-space or
    data-container inputs, and may take additional scalar resource inputs
    such as a baseline. Resource inputs are deduced from the decorated
    function.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        return _register_general_operator(
            collection=summarize,
            collection_label="summarize",
            operation_type=DiscoveryOperationEnum.SUMMARIZE,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            func=func,
        )

    return _register


def operator_metadata_for_reference(ref: OperatorReference) -> OperatorMetadata:
    """Return registry metadata for an operator reference.

    Args:
        ref: Operator reference carrying ``operatorName`` and ``operationType``.

    Returns:
        The :class:`~ado.core.operation.config.OperatorMetadata` for the
        referenced operator.

    Raises:
        ValueError: If the operation type is unknown or the operator is not registered.
    """
    collection = operationCollectionMap.get(ref.operationType)
    if collection is None:
        raise ValueError(f"Unknown operation type {ref.operationType}")

    metadata = collection.operators.get(ref.operatorName)
    if metadata is None:
        raise ValueError(
            f"Operator {ref.operatorName} had no functions of type {ref.operationType}"
        )
    return metadata


def resolve_operator_reference(ref: OperatorReference) -> OperatorReference:
    """Resolves an operator reference against the available operators and returns the result

    Resolution means:
    - Check the operator exists.
    - If the reference specifies a version: check this version exists
    - If the reference does not specify a version: set it to the version in the catalog

    Args:
        ref: Operator reference to resolve.

    Returns:
        Returns an OperatorReference instance with operatorVersion set.
        Note: This will be a different object if the input ref.operatorVersion is None

    Raises:
        ValueError: If the operator is not registered.
        OperatorVersionMismatchError: If ``ref.operatorVersion`` is set and does not
            match the registry.
    """

    metadata = operator_metadata_for_reference(ref)
    if ref.operatorVersion is None:
        ref = ref.model_copy()
        ref.operatorVersion = metadata.version
    elif ref.operatorVersion != metadata.version:
        raise OperatorVersionMismatchError(
            f"Algorithm version mismatch for operator {ref.operatorName!r} "
            f"of type {ref.operationType.value!r}. Reference requires version "
            f"{ref.operatorVersion!r} but registry provides {metadata.version!r}."
        )

    return ref


def load_operators() -> None:
    """Load all operator plugins via ``ado.operators`` entry points."""
    from importlib.metadata import entry_points

    for operator_plugin in entry_points(group="ado.operators"):
        try:
            operator_plugin.load()
            moduleLog.debug(
                f"Loaded plugin: {operator_plugin.name} from {operator_plugin.value}"
            )
        except Exception as e:  # noqa: PERF203
            moduleLog.error(f"Failed to load plugin {operator_plugin.name}: {e}")


# Load the operator plugins
load_operators()
