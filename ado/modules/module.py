# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
import logging
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

if typing.TYPE_CHECKING:
    from types import ModuleType

moduleLog = logging.getLogger("module")


class ModuleTypeEnum(enum.Enum):
    OPERATION = "operation"
    ACTUATOR = "actuator"
    SAMPLE_STORE = "sample_store"
    GENERIC = "generic"
    SAMPLER = "sampler"
    EXPERIMENT = "experiment"


class ModuleConf(pydantic.BaseModel):
    """Represents a dynamically loadable python resource"""

    model_config = ConfigDict(extra="forbid")

    moduleType: Annotated[
        ModuleTypeEnum,
        pydantic.Field(description="The type of resource the module contains"),
    ]
    moduleName: Annotated[
        str | None,
        pydantic.Field(
            validate_default=True,
            description="The name or module path of the python module "
            "with the resource. If None a guess will be made based on the type",
        ),
    ] = None
    modulePath: Annotated[
        str,
        pydantic.Field(
            description="The location of the module on filesystem. Required if its not in sys.path"
        ),
    ] = "."
    moduleClass: Annotated[
        str | None,
        pydantic.Field(
            validate_default=True,
            description="A class in the module that provides the resource. "
            "Some module may not supply resources in a class. "
            "If None a guess will be made based on moduleType",
        ),
    ] = None
    moduleFunction: Annotated[
        str | None,
        pydantic.Field(
            validate_default=True,
            description="The function for the function actuators.",
        ),
    ] = None

    @classmethod
    def _warn_legacy_module_name(cls, old_name: str, new_name: str) -> None:
        """Emit a deprecation warning when a legacy orchestrator.* moduleName is rewritten.

        Subclasses should override this to emit a resource-specific rich
        callout via warn_deprecated_resource_model_in_use. The base
        implementation falls back to a plain log warning.
        """
        moduleLog.warning(
            "Legacy moduleName %r has been auto-rewritten to %r. "
            "Update your resource to use the new name. "
            "See https://ibm.github.io/ado/migration/1x-to-2x/"
            "#renamed-python-import-package-orchestrator-ado",
            old_name,
            new_name,
        )

    @pydantic.field_validator("moduleName", mode="before")
    @classmethod
    def migrate_orchestrator_module_name(cls, value: str | None) -> str | None:
        """Rewrite legacy 'orchestrator.*' module names to 'ado.*'.

        Resources stored before the orchestrator→ado rename (commit 5687809)
        have moduleName values prefixed with 'orchestrator.'. This validator
        transparently rewrites them so existing stored resources remain usable.
        """
        if isinstance(value, str) and value.startswith("orchestrator."):
            rewritten = "ado." + value[len("orchestrator.") :]
            cls._warn_legacy_module_name(value, rewritten)
            return rewritten
        return value

    @pydantic.field_validator("moduleName")
    def set_default_module_name_for_type(
        cls, value: str | None, values: "pydantic.FieldValidationInfo"
    ) -> str | None:

        if value is None:
            if values.data.get("moduleType") == ModuleTypeEnum.OPERATION:
                value = "ado.modules.operators.randomwalk"
            elif values.data.get("moduleType") == ModuleTypeEnum.ACTUATOR:
                value = "ado.modules.actuators.base"
            elif values.data.get("moduleType") == ModuleTypeEnum.SAMPLE_STORE:
                value = "ado.core.samplestore.sql"

        return value

    @pydantic.field_validator("moduleClass")
    def set_default_class_for_type(
        cls, value: str | None, values: "pydantic.FieldValidationInfo"
    ) -> str | None:

        if value is None:
            if values.data.get("moduleType") == ModuleTypeEnum.OPERATION:
                value = "RandomWalk"
            elif values.data.get("moduleType") == ModuleTypeEnum.SAMPLE_STORE:
                value = "SQLSampleStore"

        return value

    def __str__(self) -> str:

        description = f"{self.moduleType}:"
        if self.moduleClass:
            description += f" Class: {self.moduleClass}"
        if self.moduleName:
            description += f" Module: {self.moduleName}"
        if self.modulePath:
            description += f" Path: {self.modulePath}"

        return description


def load_module(conf: ModuleConf) -> "ModuleType":
    """Loads a module and returns the module

    Params:
        conf: A ModuleConf

    A module is a python module at some path
    THe module can already be loaded or be a default class in the ado.

    Returns:
        The module

    """

    import importlib.util
    import sys

    moduleLog.debug(f"Module loading: Loading {conf}")

    # Check if a module with the name is loaded
    if sys.modules.get(conf.moduleName, None) is None:
        if conf.moduleName is not None:
            sys.path.insert(-1, conf.modulePath)

        # Create a spec for the module
        spec = importlib.util.find_spec(conf.moduleName)

        if spec is not None:
            # Convert the spec to a module
            module = importlib.util.module_from_spec(spec)

            # Load and add the module to active modules
            sys.modules[conf.moduleName] = module
            spec.loader.exec_module(module)
            moduleLog.debug(f"Module loading: {module} has been imported")

            # Now we need to connect the parent package (module) to the actuator module
            # If we don't do this, doing "import $conf.moduleName" and then accessing $conf.moduleName will
            # raise Attribute error. see second last line of Approximating import_module at https://docs.python.org/3/library/importlib.html
            absolute_name = importlib.util.resolve_name(conf.moduleName, package=None)
            if "." in absolute_name:
                parent_name, _, child_name = absolute_name.rpartition(".")
                setattr(sys.modules[parent_name], child_name, module)
        else:
            moduleLog.warning(
                f"Module loading: Failure. spec for {conf.moduleName} could not be loaded from {conf.modulePath}"
            )
            raise ValueError(
                f"Module loading: Failure: spec for {conf.moduleName} "
                f"could not be loaded from {conf.modulePath}"
            )
    else:
        moduleLog.debug(
            f"Module loading: module {conf.moduleName} is already present {conf}"
        )

    return sys.modules[conf.moduleName]


def load_module_class_or_function(conf: ModuleConf) -> type | typing.Callable:
    """Loads a module and returns the class or function

    Params:
        conf: A ModuleConf

    A module is a python module at some path with some class (the module class)
    THe module can already be loaded or be a default class in the ado.

    Returns:
        The module class or function

    Raises: ValueError if no class/function found matching conf

    """

    attribute = conf.moduleClass or conf.moduleFunction
    try:
        retval = getattr(load_module(conf), attribute)
    except AttributeError as error:
        raise ValueError(f"Unable to load class or function from {conf}") from error

    return retval
