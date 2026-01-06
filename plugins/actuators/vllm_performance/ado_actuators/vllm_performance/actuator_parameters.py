# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from typing import Any

import pydantic

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.operators.base import (
    warn_deprecated_operator_parameters_model_in_use,
)


# In case we need parameters for our actuator, we create a class
# that inherits from GenericActuatorParameters and reference it
# in the parameters_class class variable of our actuator.
# This class inherits from pydantic.BaseModel.
class VLLMPerformanceTestParameters(GenericActuatorParameters):
    namespace: str | None = pydantic.Field(
        default=None,
        description="K8s namespace for running VLLM pod. If not supplied vllm deployments cannot be created.",
    )
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

    @pydantic.model_validator(mode="before")
    @classmethod
    def make_node_selector_dict(cls, values: Any):
        import json
        from json import JSONDecodeError

        updated = False
        if isinstance(values, dict):
            node_selector = values.get("node_selector", None)
            if node_selector is not None and isinstance(node_selector, str):
                try:
                    values["node_selector"] = (
                        {} if len(node_selector) == 0 else json.loads(node_selector)
                    )
                except JSONDecodeError as error:
                    raise ValueError(
                        "The node_selector field does not contain a valid dict"
                    ) from error
                updated = True
        elif isinstance(values, GenericActuatorParameters):
            try:
                node_selector = values.node_selector
                if isinstance(node_selector, str):
                    values.node_selector = (
                        {} if len(node_selector) == 0 else json.loads(node_selector)
                    )
                    updated = True
            except JSONDecodeError as error:
                raise ValueError(
                    "The node_selector field does not contain a valid dict"
                ) from error
            except AttributeError:
                pass
        if updated:
            warn_deprecated_operator_parameters_model_in_use(
                affected_operator="vllm_performance",
                deprecated_from_operator_version="v1.2.2",
                removed_from_operator_version="v1.3",
                latest_format_documentation_url="https://example.com",
            )
        return values
