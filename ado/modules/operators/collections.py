# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import inspect
import logging
import typing
from collections.abc import Callable
from typing import Annotated, ParamSpec, TypeVar

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
from ado.core.operation.context import (
    assert_inputs_in_metastore,
    resolve_operation_project_context,
)
from ado.core.operation.inputs import resource_inputs_from_operator_function
from ado.core.resources import (
    ADOResourcePropertyDescriptor,
)
from ado.metastore.sqlstore import SQLStore
from ado.modules.operators.base import (
    DiscoveryOperationBase,
    DiscoverySpaceSubscribingDiscoveryOperation,
    OperationOutput,
    OperatorCallable,
    validate_operator_registration,
)
from ado.modules.operators.errors import OperatorVersionMismatchError
from ado.modules.operators.orchestrate import (
    orchestrate_explore_operation,
    orchestrate_general_operation,
)

moduleLog = logging.getLogger("operation_collections")

P = ParamSpec("P")
F = TypeVar("F", bound=Callable[..., OperationOutput])

_EMPTY_OPERATOR_PARAMETERS = GenericOperatorParameters()


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
    ``explore_operation``).  Each registered name maps to an
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

    def __getattr__(self, item: str) -> OperatorCallable | None:
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
operationCollectionMap = {
    ado.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE: characterize,
    ado.core.operation.config.DiscoveryOperationEnum.EXPLORE: explore,
    ado.core.operation.config.DiscoveryOperationEnum.MODIFY: modify,
    ado.core.operation.config.DiscoveryOperationEnum.EXPORT: export,
    ado.core.operation.config.DiscoveryOperationEnum.COMPARE: compare,
    ado.core.operation.config.DiscoveryOperationEnum.STUDY: study,
    ado.core.operation.config.DiscoveryOperationEnum.LEARN: learn,
    ado.core.operation.config.DiscoveryOperationEnum.FUSE: fuse,
}


def _make_general_orchestration_wrapper(
    *,
    collection: OperatorCollection,
    name: str,
    user_fn: Callable[..., OperationOutput],
    required_resource_inputs: list[ADOResourcePropertyDescriptor],
    configuration_model: type[GenericOperatorParameters],
) -> Callable[..., OperationOutput]:
    """Build a wrapper that starts ado orchestration then runs *user_fn*.

    The wrapper accepts the same call shape as *user_fn* (resource inputs,
    then ``operationInfo``, then ``parameters``). Callers — nested operators
    and :func:`~ado.modules.operators.orchestrate.orchestrate` — invoke it with
    ``fn(**inputs, operationInfo=..., parameters=...)``.
    """
    input_ids = [d.identifier for d in required_resource_inputs]
    user_sig = inspect.signature(user_fn)

    @functools.wraps(user_fn)
    def wrapper(*args: object, **kwargs: object) -> OperationOutput:
        bound = user_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)

        missing = [iid for iid in input_ids if iid not in arguments]
        if missing:
            raise ValueError(
                f"Operator {name!r} missing required resource input(s): {missing!r}."
            )

        inputs = {iid: arguments.pop(iid) for iid in input_ids}
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
        assert_inputs_in_metastore(inputs, metastore)  # type: ignore[arg-type]

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
    required_properties: list[str] | None,
    func: Callable[P, OperationOutput],
) -> Callable[P, OperationOutput]:
    """Validate, wrap, and register a general (non-explore) operator.

    Resource inputs are deduced from *func*'s signature.
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
        required_properties=required_properties,
    )
    return typing.cast("Callable[P, OperationOutput]", wrapper)


#
# Decorators for registering operator functions
#


def characterize_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
    required_properties: list[str] | None = None,
) -> Callable[[Callable[P, OperationOutput]], Callable[P, OperationOutput]]:
    """Decorator that registers a function as a characterize operation.

    Resource inputs are deduced from the decorated function's signature
    (parameters before ``operationInfo`` / ``parameters``).

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.
        required_properties: Target property identifiers this operator reads
            from a discovery space.  Used for ``ado get operators --space``
            filtering in Phase 2.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(
        func: Callable[P, OperationOutput],
    ) -> Callable[P, OperationOutput]:
        return _register_general_operator(
            collection=characterize,
            collection_label="characterize",
            operation_type=DiscoveryOperationEnum.CHARACTERIZE,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            required_properties=required_properties,
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
    classmethod.  The decorator generates an :class:`OperatorCallable`,
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
    to the class itself. Resource inputs are deduced from that generated
    function (exactly one ``discoverySpace``).

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
        parameters: GenericOperatorParameters = _EMPTY_OPERATOR_PARAMETERS,
    ) -> OperationOutput:
        op_meta = explore.operators[op_name]
        params_model = op_meta.configuration_model.model_validate(parameters)
        operation_info = operationInfo or FunctionOperationInfo()
        inputs = {"discoverySpace": discoverySpace}
        project_context = resolve_operation_project_context(operation_info, inputs)
        if operation_info.projectContext is None:
            operation_info = operation_info.model_copy(
                update={"projectContext": project_context}
            )
        metastore = SQLStore(project_context=project_context)
        assert_inputs_in_metastore(inputs, metastore)
        return orchestrate_explore_operation(
            operator_metadata=op_meta,
            discovery_space=discoverySpace,
            parameters=params_model,
            operation_info=operation_info,
        )

    # Pin ``parameters`` to the operator's concrete configuration model so
    # registration validation is as strict as for general operators.
    _generated.__annotations__["parameters"] = configuration_model
    _generated.__name__ = op_name
    _generated.__qualname__ = op_name
    _warn_if_operator_name_reused("explore", op_name, explore.operators)

    stored_inputs = resource_inputs_from_operator_function(_generated)
    validate_operator_registration(
        user_fn=_generated,
        required_resource_inputs=stored_inputs,
        operation_type=DiscoveryOperationEnum.EXPLORE,
        configuration_model=configuration_model,
    )

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
    required_properties: list[str] | None = None,
) -> Callable[[Callable[P, OperationOutput]], Callable[P, OperationOutput]]:
    """Decorator that registers a function as a modify operation.

    Resource inputs are deduced from the decorated function's signature.

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.
        required_properties: Target property identifiers used for operator
            discoverability filtering.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(
        func: Callable[P, OperationOutput],
    ) -> Callable[P, OperationOutput]:
        return _register_general_operator(
            collection=modify,
            collection_label="modify",
            operation_type=DiscoveryOperationEnum.MODIFY,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            required_properties=required_properties,
            func=func,
        )

    return _register


def export_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
    required_properties: list[str] | None = None,
) -> Callable[[Callable[P, OperationOutput]], Callable[P, OperationOutput]]:
    """Decorator that registers a function as an export operation.

    Resource inputs are deduced from the decorated function's signature
    (exactly one discoveryspace).

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.
        required_properties: Target property identifiers used for operator
            discoverability filtering.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(
        func: Callable[P, OperationOutput],
    ) -> Callable[P, OperationOutput]:
        return _register_general_operator(
            collection=export,
            collection_label="export",
            operation_type=DiscoveryOperationEnum.EXPORT,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            required_properties=required_properties,
            func=func,
        )

    return _register


def compare_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
    required_properties: list[str] | None = None,
) -> Callable[[Callable[P, OperationOutput]], Callable[P, OperationOutput]]:
    """Decorator that registers a function as a compare operation.

    Compare operators relate two or more artifacts.  They may accept any
    mix of ``discoveryspace`` and ``datacontainer`` inputs; the signature
    must declare **at least two** resource parameters.

    Example::

        @compare_operation(
            name="my_compare",
            version="0.1.0",
            configuration_model=MyCompareOptions,
            example_configuration=MyCompareOptions(),
        )
        def my_compare(
            baseline: DataContainerResource,
            candidate: DataContainerResource,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: MyCompareOptions,
        ) -> OperationOutput: ...

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.
        required_properties: Target property identifiers used for operator
            discoverability filtering.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.

    Raises:
        ValueError: If the deduced resource inputs violate compare category
            rules or the function signature is invalid.
    """

    def _register(
        func: Callable[P, OperationOutput],
    ) -> Callable[P, OperationOutput]:
        return _register_general_operator(
            collection=compare,
            collection_label="compare",
            operation_type=DiscoveryOperationEnum.COMPARE,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            required_properties=required_properties,
            func=func,
        )

    return _register


def fuse_operation(
    name: str,
    version: str,
    configuration_model: type[GenericOperatorParameters],
    example_configuration: GenericOperatorParameters,
    description: str | None = None,
    required_properties: list[str] | None = None,
) -> Callable[[Callable[P, OperationOutput]], Callable[P, OperationOutput]]:
    """Decorator that registers a function as a fuse operation.

    Fuse operators merge **two or more discovery spaces** into one.
    The signature must declare at least two ``DiscoverySpace`` parameters.

    Example::

        @fuse_operation(
            name="merge_spaces",
            version="0.1.0",
            configuration_model=MergeOptions,
            example_configuration=MergeOptions(),
        )
        def merge_spaces(
            spaceA: DiscoverySpace,
            spaceB: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            *,
            parameters: MergeOptions,
        ) -> OperationOutput: ...

    Args:
        name: Canonical operator name used in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        example_configuration: Example parameter model instance for templating.
        description: Human-readable description shown in the registry.
        required_properties: Target property identifiers used for operator
            discoverability filtering.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.

    Raises:
        ValueError: If the deduced resource inputs violate fuse category rules
            or the function signature is invalid.
    """

    def _register(
        func: Callable[P, OperationOutput],
    ) -> Callable[P, OperationOutput]:
        return _register_general_operator(
            collection=fuse,
            collection_label="fuse",
            operation_type=DiscoveryOperationEnum.FUSE,
            name=name,
            version=version,
            configuration_model=configuration_model,
            example_configuration=example_configuration,
            description=description,
            required_properties=required_properties,
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


def provenance_for_operator(
    name: str, op_type: DiscoveryOperationEnum
) -> PackageProvenance | None:
    """Return the package provenance for a registered operator.

    Looks up the operator in the collection for ``op_type`` and returns the
    :class:`~ado.core.metadata.PackageProvenance` recorded on its
    registry metadata at registration time.

    Args:
        name: Canonical operator name.
        op_type: The discovery operation type the operator belongs to.

    Returns:
        A :class:`~ado.core.metadata.PackageProvenance` instance,
        or ``None`` if provenance is unavailable.
    """
    collection = operationCollectionMap.get(op_type)
    if collection is None:
        return None
    metadata = collection.operators.get(name)
    if metadata is None:
        return None
    return metadata.provenance


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
