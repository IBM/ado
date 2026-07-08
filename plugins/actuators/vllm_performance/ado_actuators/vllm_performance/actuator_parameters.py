# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated, Any

import pydantic
from pydantic import AfterValidator

from ado.core.actuatorconfiguration.config import GenericActuatorParameters
from ado.utilities.pydantic import validate_rfc_1123


# In case we need parameters for our actuator, we create a class
# that inherits from GenericActuatorParameters and reference it
# in the parameters_class class variable of our actuator.
# This class inherits from pydantic.BaseModel.
class VLLMPerformanceTestParameters(GenericActuatorParameters):
    namespace: Annotated[
        str | None,
        pydantic.Field(
            description="K8s namespace for running VLLM pod. If not supplied vllm deployments cannot be created.",
            validate_default=False,
        ),
        AfterValidator(validate_rfc_1123),
    ] = None
    in_cluster: Annotated[
        bool,
        pydantic.Field(
            description="flag to determine whether we are running in K8s cluster or locally",
        ),
    ] = False
    verify_ssl: Annotated[
        bool, pydantic.Field(description="flag to verify SLL when connecting to server")
    ] = False
    image_pull_secret_name: Annotated[
        str, pydantic.Field(description="secret to use when loading image")
    ] = ""
    node_selector: Annotated[
        dict[str, str],
        pydantic.Field(
            default_factory=dict,
            description="dictionary containing node selector key:value pairs",
        ),
    ]
    deployment_template: Annotated[
        str | None, pydantic.Field(description="name of deployment template")
    ] = None
    service_template: Annotated[
        str | None, pydantic.Field(description="name of service template")
    ] = None
    pvc_template: Annotated[
        str | None, pydantic.Field(description="name of pvc template")
    ] = None
    pvc_name: Annotated[
        None | str, pydantic.Field(description="name of pvc to be created/attached")
    ] = None
    benchmark_retries: Annotated[
        int, pydantic.Field(description="number of retries for running benchmark")
    ] = 3
    retries_timeout: Annotated[
        int, pydantic.Field(description="initial timeout between retries")
    ] = 5
    hf_token: Annotated[
        str,
        pydantic.Field(
            validate_default=True,
            description="Huggingface token - can be empty if you are accessing fully open models",
        ),
    ] = ""
    max_environments: Annotated[
        int, pydantic.Field(description="Maximum amount of concurrent environments")
    ] = 1
    developer_mode: Annotated[
        bool,
        pydantic.Field(
            description="If true, disables automatic installation of vllm and guidellm dependencies in Ray task environment. Useful for development when dependencies are already installed."
        ),
    ] = False
    otlp_traces_endpoint: Annotated[
        pydantic.AnyUrl | None,
        pydantic.Field(
            description="OpenTelemetry traces endpoint URL for vLLM observability"
        ),
    ] = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def delete_interpreter(cls, values: Any) -> Any:  # noqa: ANN401

        # We expect either a GenericActuatorParameters or a dict instance
        if not isinstance(values, GenericActuatorParameters) and not isinstance(
            values, dict
        ):
            raise ValueError(f"Unexpected type {type(values)} in validator")

        from ado.core.actuatorconfiguration.config import (
            warn_deprecated_actuator_parameters_model_in_use,
        )

        key = "interpreter"

        if isinstance(values, GenericActuatorParameters):
            # The key is not present - all good
            if not hasattr(values, key):
                return values

            # Notify the user that the 'interpreter'
            # field is deprecated
            warn_deprecated_actuator_parameters_model_in_use(
                affected_actuator="vllm_performance",
                deprecated_from_actuator_version="v1.8.0",
                removed_from_actuator_version="v1.9.0",
                deprecated_fields=key,
            )

            # delete the 'interpreter' field from the actuator configuration
            if hasattr(values, key):
                delattr(values, key)

        else:
            # The key is not present - all good
            if key not in values:
                return values

            # Notify the user that the 'interpreter'
            # field is deprecated
            warn_deprecated_actuator_parameters_model_in_use(
                affected_actuator="vllm_performance",
                deprecated_from_actuator_version="v1.8.0",
                removed_from_actuator_version="v1.9.0",
                deprecated_fields=key,
            )

            # The user has set both the old
            # and the new key - the new key
            # takes precedence.
            if key in values:
                values.pop(key)

        return values
