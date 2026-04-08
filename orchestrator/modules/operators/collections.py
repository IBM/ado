# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import functools
import logging
import typing
import warnings
from typing import Annotated

import pydantic
from pydantic import ConfigDict

import orchestrator.core.operation.config
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    OperatorMetadata,
    OperatorReference,
)
from orchestrator.modules.operators.base import (
    DiscoveryOperationBase,
    OperationOutput,
    OperatorFunction,
)
from orchestrator.modules.operators.orchestrate import (
    orchestrate_explore_operation,
    orchestrate_general_operation,
)

moduleLog = logging.getLogger("operation_collections")


def _validate_operator_cls(cls: type, context: str) -> None:
    """Assert that ``cls`` is a subclass of :class:`DiscoveryOperationBase`.

    ``OperatorMetadata.cls`` is typed as ``type | None`` in the core layer to
    keep ``config.py`` decoupled from ``base.py``.  This function is the single
    place in ``modules/operators`` where the concrete constraint is enforced.

    Args:
        cls: The class to validate.
        context: A short description of the call site, included in the error
            message.

    Raises:
        TypeError: If ``cls`` is not a subclass of ``DiscoveryOperationBase``.
    """
    if not (isinstance(cls, type) and issubclass(cls, DiscoveryOperationBase)):
        raise TypeError(
            f"{context}: expected a DiscoveryOperationBase subclass, got {cls!r}."
        )


class OperatorCollection(pydantic.BaseModel):
    """A registry of operators of a single discovery operation type.

    Operators are added via the decorator functions (e.g. ``characterize_operation``,
    ``explore_operation``).  Each registered name maps to an
    :class:`~orchestrator.core.operation.config.OperatorMetadata` instance that
    carries the function, version, description, configuration model,
    example configuration, and optional actor class.

    Attributes:
        type: The discovery operation type all operators in this collection belong to.
        operators: Mapping of operator name to :class:`~orchestrator.core.operation.config.OperatorMetadata` instance.
    """

    type: DiscoveryOperationEnum
    operators: Annotated[
        dict[str, OperatorMetadata], pydantic.Field(default_factory=dict)
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def list_operators(self) -> list[str]:
        """Returns all registered operator names."""
        return list(self.operators.keys())

    def list_operations(self) -> list[str]:
        """Returns all registered operator names.

        Deprecated:
            Use :meth:`list_operators` instead.
        """
        warnings.warn(
            "list_operations() is deprecated; use list_operators() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_operators()

    def configuration_model_for_operation(
        self, name: str
    ) -> type[pydantic.BaseModel] | None:
        """Returns the configuration model for the named operator.

        Deprecated:
            Access ``collection.operators[name].configuration_model`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The pydantic model class used to validate parameters, or None.

        Raises:
            ValueError: If the operator name is not registered.
        """
        warnings.warn(
            "configuration_model_for_operation() is deprecated; "
            "access collection.operators[name].configuration_model directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators:
            raise ValueError(f"Unknown operator {name}")
        return self.operators[name].configuration_model

    def default_configuration_model_for_operation(
        self, name: str
    ) -> pydantic.BaseModel | None:
        """Returns the example configuration instance for the named operator.

        Deprecated:
            Access ``collection.operators[name].example_configuration`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The default pydantic model instance, or None.

        Raises:
            ValueError: If the operator name is not registered.
        """
        warnings.warn(
            "default_configuration_model_for_operation() is deprecated; "
            "access collection.operators[name].example_configuration directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators:
            raise ValueError(f"Unknown operator {name}")
        return self.operators[name].example_configuration

    def description_for_operation(self, name: str) -> str | None:
        """Returns the description for the named operator.

        Deprecated:
            Access ``collection.operators[name].description`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The description string, or None.

        Raises:
            ValueError: If the operator name is not registered.
        """
        warnings.warn(
            "description_for_operation() is deprecated; "
            "access collection.operators[name].description directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators:
            raise ValueError(f"Unknown operator {name}")
        return self.operators[name].description

    def operator_class_for_operation(self, name: str) -> type[DiscoveryOperationBase]:
        """Returns the actor class registered for the named explore operator.

        Deprecated:
            Access ``collection.operators[name].cls`` directly.

        Args:
            name: The registered operator name.

        Returns:
            The DiscoveryOperationBase subclass for this operator.

        Raises:
            ValueError: If no class has been registered for the given name.
        """
        warnings.warn(
            "operator_class_for_operation() is deprecated; "
            "access collection.operators[name].cls directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name not in self.operators or self.operators[name].cls is None:
            raise ValueError(f"No operator class registered for {name}")
        return self.operators[name].cls

    def __getattr__(self, item: str) -> OperatorFunction:
        """Returns the operator function for the given registered name.

        Args:
            item: Registered operator name.

        Returns:
            The callable registered under that name.

        Raises:
            AttributeError: If no operator is registered with that name.
        """
        if item in self.operators:
            return self.operators[item].function
        raise AttributeError(f"Unknown attribute {item}")


characterize = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE
)
explore = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH
)
modify = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY
)
export = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT
)
compare = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.COMPARE
)
fuse = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.FUSE
)
study = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.STUDY
)
learn = OperatorCollection(
    type=orchestrator.core.operation.config.DiscoveryOperationEnum.LEARN
)
operationCollectionMap = {
    orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE: characterize,
    orchestrator.core.operation.config.DiscoveryOperationEnum.SEARCH: explore,
    orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY: modify,
    orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT: export,
    orchestrator.core.operation.config.DiscoveryOperationEnum.COMPARE: compare,
    orchestrator.core.operation.config.DiscoveryOperationEnum.STUDY: study,
    orchestrator.core.operation.config.DiscoveryOperationEnum.LEARN: learn,
    orchestrator.core.operation.config.DiscoveryOperationEnum.FUSE: fuse,
}

#
# Decorators for registering operator functions
#


def characterize_operation(
    name: str,
    description: str | None = None,
    version: str | None = "v0.1",
    configuration_model: type[pydantic.BaseModel] | None = None,
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a characterize operation.

    Args:
        name: Canonical operator name used in the registry.
        description: Human-readable description shown in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        configuration_model_default: Default parameter model instance.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        @functools.wraps(func)
        def wrapper(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            **kwargs: object,
        ) -> OperationOutput:
            return orchestrate_general_operation(
                operator_function=func,
                operation_parameters=kwargs,
                parameters_model=characterize.operators[name].configuration_model,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
                operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.CHARACTERIZE,
            )

        characterize.operators[name] = OperatorMetadata(
            name=name,
            function=typing.cast("OperatorFunction", wrapper),
            version=version or "v0.1",
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.CHARACTERIZE,
        )
        return typing.cast("OperatorFunction", wrapper)

    return _register


def explore_operation(
    target: "type[DiscoveryOperationBase] | None" = None,
    *,
    name: str | None = None,
    operator_class: "type[DiscoveryOperationBase] | None" = None,
    description: str | None = None,
    configuration_model: type[pydantic.BaseModel] | None = None,
    version: str | None = "v0.1",
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable:
    """Decorator that registers an explore (search) operator.

    Supports two usage patterns:

    **Class decoration** (no arguments) — all metadata is taken from the
    class's :meth:`~orchestrator.modules.operators.base.DiscoveryOperationBase.operator_metadata`
    classmethod.  The decorator generates the :data:`~orchestrator.modules.operators.base.OperatorFunction`
    body, fills in ``function`` and ``cls`` on the returned
    :class:`~orchestrator.core.operation.config.OperatorMetadata`, and
    registers it.  The name bound in the decorated module's scope becomes the
    generated function::

        @explore_operation
        class MyOp(Search):
            @classmethod
            def operator_metadata(cls) -> OperatorMetadata:
                return OperatorMetadata(
                    name="my_op",
                    version="v0.1",
                    description="...",
                    configuration_model=MyOpParameters,
                    example_configuration=MyOpParameters(),
                    type=DiscoveryOperationEnum.SEARCH,
                )
            ...

    **Function decoration** (with keyword arguments) — all metadata is
    supplied via decorator arguments, matching the classic pattern::

        @explore_operation(name="my_op", operator_class=MyOp, ...)
        def my_op(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            **kwargs: object,
        ) -> OperationOutput:
            return orchestrate_explore_operation(...)

    Args:
        target: Set automatically when the decorator is used without parentheses
            (class path).  Do not pass this argument explicitly.
        name: Canonical operator name (function path only; required).
        operator_class: The class implementing the operation (function path
            only; optional).  Unwrapped from ``@ray.remote`` if necessary.
        description: Human-readable description shown in the registry.
        configuration_model: Pydantic model used to validate operation
            parameters (function path only).
        version: Semantic version string included in the
            :attr:`~orchestrator.core.operation.config.OperatorMetadata.operatorIdentifier`
            (e.g. ``"v0.1"``).
        configuration_model_default: Default parameter model instance
            (function path only).

    Returns:
        The generated :data:`~orchestrator.modules.operators.base.OperatorFunction`
        (class path) or a decorator that registers and returns the decorated
        function (function path).

    Raises:
        NotImplementedError: (class path) If the decorated class has not
            implemented ``operator_metadata()``.
        TypeError: (function path) If ``name`` is not provided.
    """

    def _register(
        t: "OperatorFunction | type[DiscoveryOperationBase]",
    ) -> OperatorFunction:
        import inspect

        from orchestrator.utilities.ray import extract_base_class

        if inspect.isclass(t) and issubclass(t, DiscoveryOperationBase):
            # ------------------------------------------------------------------
            # Class decoration path — metadata from cls.operator_metadata()
            # Class-decorated operators are never also @ray.remote, so t is
            # always the plain Python class — no Ray unwrapping needed.
            # ------------------------------------------------------------------
            metadata = (
                t.operator_metadata()
            )  # raises NotImplementedError if not overridden
            op_name = metadata.name
            if metadata.configuration_model is None:
                raise TypeError(
                    f"@explore_operation on {t.__name__}: configuration_model must be "
                    "set in operator_metadata(). Provide it via the "
                    "configuration_model field."
                )

            def _generated(
                discoverySpace: DiscoverySpace,
                operationInfo: FunctionOperationInfo | None = None,
                **kwargs: object,
            ) -> OperationOutput:
                return orchestrate_explore_operation(
                    discovery_space=discoverySpace,
                    operator_reference=OperatorReference(
                        operationType=DiscoveryOperationEnum.SEARCH,
                        operatorName=op_name,
                    ),
                    parameters=kwargs,
                    operation_info=operationInfo or FunctionOperationInfo(),
                )

            _generated.__name__ = op_name
            _generated.__qualname__ = op_name

            explore.operators[op_name] = metadata.model_copy(
                update={
                    "function": typing.cast("OperatorFunction", _generated),
                    "cls": t,
                }
            )
            return typing.cast("OperatorFunction", _generated)

        # ------------------------------------------------------------------
        # Function decoration path — metadata from decorator arguments,
        # supplemented by operator_class.operator_metadata() when available.
        # ------------------------------------------------------------------
        if name is None:
            raise TypeError(
                "explore_operation: 'name' must be provided when decorating a function."
            )
        func = typing.cast("OperatorFunction", t)
        resolved_cls = None
        base_meta: OperatorMetadata | None = None
        if operator_class is not None:
            resolved_cls = extract_base_class(operator_class, DiscoveryOperationBase)
            _validate_operator_cls(
                resolved_cls,
                f"@explore_operation(name={name!r}, operator_class=...)",
            )
            import contextlib

            with contextlib.suppress(NotImplementedError):
                base_meta = resolved_cls.operator_metadata()

        def _log_override(field: str, base_val: object, new_val: object) -> None:
            if base_val is not None and new_val != base_val:
                moduleLog.info(
                    "explore_operation %r: decorator arg '%s' overrides "
                    "class operator_metadata() value %r with %r",
                    name,
                    field,
                    base_val,
                    new_val,
                )

        if configuration_model is not None:
            _log_override(
                "configuration_model",
                base_meta.configuration_model if base_meta else None,
                configuration_model,
            )
            final_config_model = configuration_model
        else:
            final_config_model = base_meta.configuration_model if base_meta else None

        if configuration_model_default is not None:
            _log_override(
                "example_configuration",
                base_meta.example_configuration if base_meta else None,
                configuration_model_default,
            )
            final_example = configuration_model_default
        else:
            final_example = base_meta.example_configuration if base_meta else None

        if description is not None:
            _log_override(
                "description",
                base_meta.description if base_meta else None,
                description,
            )
            final_description = description
        else:
            final_description = base_meta.description if base_meta else None

        # version defaults to "v0.1" in the signature; prefer the class value
        # when the decorator relies on the sentinel default.
        if version != "v0.1":
            _log_override(
                "version",
                base_meta.version if base_meta else None,
                version,
            )
            final_version = version
        else:
            final_version = (base_meta.version if base_meta else None) or "v0.1"

        if final_config_model is None:
            raise TypeError(
                f"explore_operation {name!r}: configuration_model must be set. "
                "Supply it via the decorator arg or operator_class.operator_metadata()."
            )

        explore.operators[name] = OperatorMetadata(
            name=name,
            cls=resolved_cls,
            function=func,
            version=final_version or "v0.1",
            description=final_description,
            configuration_model=final_config_model,
            example_configuration=final_example,
            type=DiscoveryOperationEnum.SEARCH,
        )
        return func

    if target is not None:
        # @explore_operation without parentheses — must be a class
        return _register(target)
    return _register


def modify_operation(
    name: str,
    description: str | None = None,
    version: str | None = "v0.1",
    configuration_model: type[pydantic.BaseModel] | None = None,
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as a modify operation.

    Args:
        name: Canonical operator name used in the registry.
        description: Human-readable description shown in the registry.
        version: Version string included in the operator identifier.
        configuration_model: Pydantic model used to validate operation parameters.
        configuration_model_default: Default parameter model instance.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        @functools.wraps(func)
        def wrapper(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            **kwargs: object,
        ) -> OperationOutput:
            return orchestrate_general_operation(
                operator_function=func,
                operation_parameters=kwargs,
                parameters_model=modify.operators[name].configuration_model,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
                operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.MODIFY,
            )

        modify.operators[name] = OperatorMetadata(
            name=name,
            function=typing.cast("OperatorFunction", wrapper),
            version=version or "v0.1",
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.MODIFY,
        )
        return typing.cast("OperatorFunction", wrapper)

    return _register


def export_operation(
    name: str,
    description: str | None = None,
    configuration_model: type[pydantic.BaseModel] | None = None,
    version: str | None = "v0.1",
    configuration_model_default: pydantic.BaseModel | None = None,
) -> typing.Callable[[OperatorFunction], OperatorFunction]:
    """Decorator that registers a function as an export operation.

    Args:
        name: Canonical operator name used in the registry.
        description: Human-readable description shown in the registry.
        configuration_model: Pydantic model used to validate operation parameters.
        version: Version string included in the operator identifier.
        configuration_model_default: Default parameter model instance.

    Returns:
        A decorator that wraps and registers the decorated function under ``name``.
    """

    def _register(func: OperatorFunction) -> OperatorFunction:
        @functools.wraps(func)
        def wrapper(
            discoverySpace: DiscoverySpace,
            operationInfo: FunctionOperationInfo | None = None,
            **kwargs: object,
        ) -> OperationOutput:
            return orchestrate_general_operation(
                operator_function=func,
                operation_parameters=kwargs,
                parameters_model=export.operators[name].configuration_model,
                discovery_space=discoverySpace,
                operation_info=operationInfo or FunctionOperationInfo(),
                operation_type=orchestrator.core.operation.config.DiscoveryOperationEnum.EXPORT,
            )

        export.operators[name] = OperatorMetadata(
            name=name,
            function=typing.cast("OperatorFunction", wrapper),
            version=version or "v0.1",
            description=description,
            configuration_model=configuration_model,
            example_configuration=configuration_model_default,
            type=DiscoveryOperationEnum.EXPORT,
        )
        return typing.cast("OperatorFunction", wrapper)

    return _register


def load_operators() -> None:
    from importlib.metadata import entry_points

    import orchestrator.modules.operators.randomwalk  # noqa: F401

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
