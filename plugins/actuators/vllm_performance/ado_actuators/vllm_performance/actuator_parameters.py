# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import Annotated

import pydantic
from pydantic import AfterValidator

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.utilities.pydantic import validate_rfc_1123


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
    in_cluster: bool = pydantic.Field(
        default=False,
        description="flag to determine whether we are running in K8s cluster or locally",
    )
    verify_ssl: bool = pydantic.Field(
        default=False, description="flag to verify SLL when connecting to server"
    )
    image_secret: str = pydantic.Field(
        default="", description="secret to use when loading image"
    )
    node_selector: dict[str, str] = pydantic.Field(
        default={}, description="dictionary containing node selector key:value pairs"
    )
    deployment_template: str | None = pydantic.Field(
        default=None, description="name of deployment template"
    )
    service_template: str | None = pydantic.Field(
        default=None, description="name of service template"
    )
    pvc_template: str | None = pydantic.Field(
        default=None, description="name of pvc template"
    )
    pvc_name: None | str = pydantic.Field(
        default=None, description="name of pvc to be created/attached"
    )
    interpreter: str = pydantic.Field(
        default="python3", description="name of python interpreter"
    )
    benchmark_retries: int = pydantic.Field(
        default=3, description="number of retries for running benchmark"
    )
    retries_timeout: int = pydantic.Field(
        default=5, description="initial timeout between retries"
    )
    hf_token: str = pydantic.Field(
        default="",
        validate_default=True,
        description="Huggingface token - can be empty if you are accessing fully open models",
    )
    max_environments: int = pydantic.Field(
        default=1, description="Maximum amount of concurrent environments"
    )
