# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters


# In case we need parameters for our actuator, we create a class
# that inherits from GenericActuatorParameters and reference it
# in the parameters_class class variable of our actuator.
# This class inherits from pydantic.BaseModel.
class VLLMPerformanceTestParameters(GenericActuatorParameters):
    namespace: str | None = pydantic.Field(
        default=None,
        description="k8 namespace for running VLLM pod. If not supplied vllm deployments cannot be created.",
    )
    in_cluster: bool = pydantic.Field(
        default=True,
        description="flag to determine whether we are running in k8 cluster or locally",
    )
    verify_ssl: bool = pydantic.Field(
        default=False, description="flag to verify SLL when connecting to server"
    )
    image_secret: str = pydantic.Field(
        default="", description="secret to use when loading image"
    )
    node_selector: str = pydantic.Field(
        default="", description="json string containing node selector (dictionary)"
    )
    deployment_template: str = pydantic.Field(
        default="deployment.yaml", description="name of deployment template"
    )
    service_template: str = pydantic.Field(
        default="service.yaml", description="name of service template"
    )
    pvc_template: str = pydantic.Field(
        default="pvc.yaml", description="name of pvc template"
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
