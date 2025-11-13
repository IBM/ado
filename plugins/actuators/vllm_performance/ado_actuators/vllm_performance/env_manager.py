# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import copy
import logging
import time
from enum import Enum

import ray
from ado_actuators.vllm_performance.k8s.manage_components import (
    ComponentsManager,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
)
from kubernetes.client import ApiException

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
        self.k8s_name = ComponentsYaml.get_k8s_name(model=model)
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
        pvc_name: str | None = None,
        pvc_template: str | None = None,
    ):
        """
        Initialize
        :param namespace: deployment namespace
        :param max_concurrent: maximum amount of concurrent environment
        :param in_cluster: flag in cluster
        :param verify_ssl: flag verify SSL
        :param pvc_name: name of the PVC to be created / used
        :param pvc_template: template of the PVC to be created
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
            init_pvc=True,
            pvc_name=pvc_name,
            pvc_template=pvc_template,
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
                        try:
                            self.manager.delete_service(k8s_name=env.k8s_name)
                            self.manager.delete_deployment(k8s_name=env.k8s_name)
                        except ApiException as e:
                            logger.error(f"Error deleting deployment or service {e}")
                        del self.environments[key]
                        print(
                            f"deleted environment {env.k8s_name} in {time.time() - start} sec. "
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

    def get_experiment_pvc_name(self):
        return self.manager.pvc_name

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
        logger.info("Cleaning environments")
        for env in self.environments.values():
            try:
                self.manager.delete_service(k8s_name=env.k8s_name)
            except ApiException as e:
                if e.reason != "Not Found":
                    raise e
            try:
                self.manager.delete_deployment(k8s_name=env.k8s_name)
            except ApiException as e:
                if e.reason != "Not Found":
                    raise e
        # We only delete the PVC if it was created by this actuator
        if self.manager.pvc_created:
            logger.debug("Deleting PVC")
            self.manager.delete_pvc()
        else:
            logger.debug("No PVC was created. Nothing to delete!")
