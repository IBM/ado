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
    in_cluster: Annotated[
        bool,
        pydantic.Field(
            description="flag to determine whether we are running in K8s cluster or locally",
        ),
    ] = False
    verify_ssl: Annotated[
        bool, pydantic.Field(description="flag to verify SLL when connecting to server")
    ] = False
    image_secret: Annotated[
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
    interpreter: Annotated[
        str, pydantic.Field(description="name of python interpreter")
    ] = "python3"
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
