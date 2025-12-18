# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing

import pydantic
from pydantic import ConfigDict

from orchestrator.core.metadata import ConfigurationMetadata

if typing.TYPE_CHECKING:  # pragma: nocover
    from orchestrator.modules.actuators.base import ActuatorBase


class GenericActuatorParameters(pydantic.BaseModel):
    # AP 5/11/24:
    # We allow extras because this model will be inherited
    # by actuator-specific models and will be used when
    # deserializing. If this wasn't the case, we would
    # ignore all the values
    model_config = ConfigDict(extra="allow")


class ActuatorConfiguration(pydantic.BaseModel):
    model_config = ConfigDict(extra="forbid")

    actuatorIdentifier: str
    # SerializeAsAny to serialise the correct class
    # https://github.com/pydantic/pydantic/discussions/9879#discussioncomment-10102592
    parameters: pydantic.SerializeAsAny[GenericActuatorParameters | None] = (
        pydantic.Field(default=None)
    )
    metadata: ConfigurationMetadata = pydantic.Field(
        default=ConfigurationMetadata(),
        description="User defined metadata about the configuration. A set of keys and values. "
        "Two optional keys that are used by convention are name and description",
    )

    @pydantic.model_validator(mode="after")
    @classmethod
    def validate_model(
        cls,
        model: "ActuatorConfiguration",
    ) -> "ActuatorConfiguration":
        from orchestrator.modules.actuators.registry import (
            ActuatorRegistry,
            UnknownActuatorError,
        )

        def validate_or_default_parameters(
            actuator_instance: "ActuatorBase",
        ) -> GenericActuatorParameters:
            return (
                actuator_instance.validate_parameters(parameters=model.parameters)
                if model.parameters
                else actuator_instance.default_parameters()
            )

        actuator_registry = ActuatorRegistry.globalRegistry()

        try:
            actuator = actuator_registry.actuatorForIdentifier(model.actuatorIdentifier)
        except UnknownActuatorError as error:
            raise ValueError(
                f"Actuator {model.actuatorIdentifier} is not available in the registry. "
                f"Registered actuators are: {','.join(actuator_registry.actuatorIdentifierMap.keys())}"
            ) from error
        else:
            model.parameters = validate_or_default_parameters(actuator)

        return model


def warn_deprecated_actuator_parameters_model_in_use(
    affected_actuator: str,
    deprecated_from_actuator_version: str,
    removed_from_actuator_version: str,
    deprecated_fields: str | list[str] | None = None,
    latest_format_documentation_url: str | None = None,
):
    from rich.console import Console

    resource_name = "actuatorconfiguration"
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
            f"The use of {fields_causing_issues} in the parameters of the {affected_actuator} actuator "
            f"is deprecated as of {affected_actuator} [b cyan]{deprecated_from_actuator_version}[/b cyan]."
        )
    else:
        warning_preamble = (
            f"The parameters for the {affected_actuator} actuator have been updated "
            f"as of {affected_actuator} [b cyan]{deprecated_from_actuator_version}[/b cyan]."
        )

    autoupgrade_notice = (
        "They are being temporarily auto-upgraded to the latest version."
    )
    autoupgrade_removal_warning = (
        f"[b]This behavior will be removed with {affected_actuator} "
        f"[b cyan]{removed_from_actuator_version}[/b cyan][/b]."
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
