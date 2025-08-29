# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import logging
import time
from enum import Enum

import ray
from ado_actuators.vllm_performance.k8.manage_components import (
    ComponentsManager,
)
from ado_actuators.vllm_performance.k8.yaml_support.build_components import (
    ComponentsYaml,
)

logger = logging.getLogger(__name__)


class EnvironmentState(Enum):
    """
    Environment state
    """

    NONE = "None"
    CREATING = "creating"
    READY = "ready"


class Environment:
    """
    environment class
    """

    def __init__(self, model: str):
        """
        Defines an environment for a model
        :param model: LLM model name
        """
        self.k8_name = ComponentsYaml.get_k8_name(model=model)
        self.state = EnvironmentState.NONE
        self.in_use = 0

    def update_creating(self):
        val = copy.deepcopy(self)
        val.state = EnvironmentState.CREATING
        val.in_use = 1
        return val


@ray.remote
class EnvironmentManager:
    """
    This is a Ray actor (singleton) managing environments
    """

    def __init__(
        self,
        namespace: str,
        max_concurrent: int,
        in_cluster: bool = True,
        verify_ssl: bool = False,
    ):
        """
        Initialize
        :param namespace: deployment namespace
        :param max_concurrent: maximum amount of concurrent environment
        :param in_cluster: flag in cluster
        :param verify_ssl: flag verify SSL
        """
        self.environments = {}
        self.namespace = namespace
        self.max_concurrent = max_concurrent
        self.in_cluster = in_cluster
        self.verify_ssl = verify_ssl
        # component manager for cleanup
        self.manager = ComponentsManager(
            namespace=self.namespace,
            in_cluster=self.in_cluster,
            verify_ssl=self.verify_ssl,
        )

    def get_environment(
        self, model: str, definition: str, increment_usage: bool = False
    ) -> Environment:
        """
        Get an environment for definition
        :param model: LLM model name
        :param definition: environment definition - json string containing:
                        model, image, n_gpus, gpu_type, n_cpus, memory, max_batch_tokens,
                        gpu_memory_utilization, dtype, cpu_offload, max_num_seq
        :param increment_usage: increment usage flag
        :return: environment state
        """
        print(
            f"getting environment for model {model}, currently {len(self.environments)} deployments"
        )
        env = self.environments.get(definition, None)
        if env is None:
            if len(self.environments) >= self.max_concurrent:
                # can't create more environments now, need clean up
                available = False
                for key, env in self.environments.items():
                    if env.in_use == 0:
                        available = True
                        start = time.time()
                        self.manager.delete_service(k8_name=env.k8_name)
                        self.manager.delete_deployment(k8_name=env.k8_name)
                        del self.environments[key]
                        print(
                            f"deleted environment {env.k8_name} in {time.time() - start} sec. "
                            f"Environments length {len(self.environments)}"
                        )
                        time.sleep(3)
                        break
                if not available:
                    return None
            # mark new one
            env = Environment(model=model)
            self.environments[definition] = env.update_creating()
            return env
        if increment_usage:
            env = self.environments.get(definition)
            env.in_use += 1
            self.environments[definition] = env
        return env

    def done_creating(self, definition: str) -> None:
        """
        Report creation
        :param definition: environment definition
        :return: None
        """
        env = self.environments.get(definition, None)
        if env is None:
            return
        env.state = EnvironmentState.READY
        self.environments[definition] = env

    def done_using(self, definition: str) -> None:
        """
        Report test completion
        :param definition: environment definition
        :return: None
        """
        env = self.environments.get(definition)
        if env is None:
            return
        env.in_use -= 1
        self.environments[definition] = env

    def cleanup(self) -> None:
        """
        Clean up environment
        :return: None
        """
        print("Cleaning environment manager")
        for env in self.environments.values():
            if env.state == EnvironmentState.READY:
                self.manager.delete_service(k8_name=env.k8_name)
                self.manager.delete_deployment(k8_name=env.k8_name)
