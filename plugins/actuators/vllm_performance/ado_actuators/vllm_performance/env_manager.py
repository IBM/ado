# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging
from enum import Enum

import ray
from ado_actuators.vllm_performance.deployment_management import (
    DeploymentConflictManager,
)
from ado_actuators.vllm_performance.k8s import K8sEnvironmentCreationError
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
    READY = "ready"


class Environment:
    """
    environment class
    """

    def __init__(self, model: str, configuration: str) -> None:
        """
        Defines an environment for a model
        :param model: LLM model name
        :param configuration: The full deployment configuration
        """
        self.k8s_name = ComponentsYaml.get_k8s_name(model=model)
        self.state = EnvironmentState.NONE
        self.configuration = configuration
        self.model = model


class EnvironmentsQueue:
    def __init__(self) -> None:
        self.environments_queue = []

    async def wait(self) -> None:
        wait_event = asyncio.Event()
        self.environments_queue.append(wait_event)
        await wait_event.wait()

    def signal_next(self) -> None:
        if len(self.environments_queue) > 0:
            event = self.environments_queue.pop(0)
            event.set()


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
    ) -> None:
        """
        Initialize
        :param namespace: deployment namespace
        :param max_concurrent: maximum amount of concurrent environment
        :param in_cluster: flag in cluster
        :param verify_ssl: flag verify SSL
        :param pvc_name: name of the PVC to be created / used
        :param pvc_template: template of the PVC to be created
        """
        self.in_use_environments: dict[str, Environment] = {}
        self.free_environments: list[Environment] = []
        self.environments_queue = EnvironmentsQueue()
        self.deployment_conflict_manager = DeploymentConflictManager()
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

    def _delete_environment_k8s_resources(self, k8s_name: str) -> None:
        """
        Deletes a deployment. Intended to be used for cleanup or error recovery
        param: identifier: the deployment identifier
        """
        try:
            self.manager.delete_service(k8s_name=k8s_name)
        except ApiException as e:
            if e.reason != "Not Found":
                raise e
        try:
            self.manager.delete_deployment(k8s_name=k8s_name)
        except ApiException as e:
            if e.reason != "Not Found":
                raise e

    def environment_usage(self) -> dict:

        return {"max": self.max_concurrent, "in_use": self.active_environments}

    async def wait_for_env(self) -> None:
        await self.environments_queue.wait()

    def get_environment(self, model: str, definition: str) -> Environment | None:
        """
        Get an environment for definition
        :param model: LLM model name
        :param definition: environment definition - json string containing:
                        model, image, n_gpus, gpu_type, n_cpus, memory, max_batch_tokens,
                        gpu_memory_utilization, dtype, cpu_offload, max_num_seq
        :param increment_usage: increment usage flag
        :return: environment state
        """

        # check if there's an existing free environment satisfying the request
        env = self.get_matching_free_environment(definition)
        if env is None:
            if self.active_environments >= self.max_concurrent:
                # can't create more environments now, need clean up
                if len(self.free_environments) == 0:
                    # No room for creating a new environment
                    logger.debug(
                        f"There are already {self.max_concurrent} actively in use, and I can't create a new one"
                    )
                    return None

                # There are unused environments, let's try to evict one
                environment_evicted = False
                eviction_index = 0
                # Continue looping until we find one environment that can be successfully evicted or we have gone through them all
                while not environment_evicted and eviction_index < len(
                    self.free_environments
                ):
                    environment_to_evict = self.free_environments[eviction_index]
                    try:
                        # _delete_environment_k8s_resources will not raise an error if for whatever the reason the service
                        # or the deployment we are trying to delete does not exist anymore, and we assume
                        # the deployment was properly deleted.
                        self._delete_environment_k8s_resources(
                            k8s_name=environment_to_evict.k8s_name
                        )
                    except ApiException as e:
                        # If we can't delete this environment we try with the next one, but we do not
                        # delete the current env from the free list. This is to avoid spawning more pods than the maximum configured
                        # in the case the failing ones are still running.
                        # Since the current eviction candidate environment will stay in the free ones, some other measurement might
                        # try to evict again and perhaps succeed (e.g., connection restored to the cluster).
                        logger.critical(
                            f"Error deleting deployment or service {environment_to_evict.k8s_name}: {e}"
                        )
                        eviction_index += 1
                        continue

                    logger.info(
                        f"deleted environment {environment_to_evict.k8s_name}. "
                        f"Active environments {self.active_environments}"
                    )
                    environment_evicted = True

                if environment_evicted:
                    # successfully deleted an environment
                    self.free_environments.pop(eviction_index)
                elif len(self.in_use_environments) > 0:
                    # all the free ones have failed deleting but there is one or more in use that
                    # might make room for waiting measurements. In this case we just behave as if there
                    # are no free available environments and we wait.
                    return None
                else:
                    # None of the free environments could be evicted due to errors and none are in use
                    # To avoid a deadlock of the operation we fail the measurement
                    raise K8sEnvironmentCreationError(
                        "All free environments failed deleting and none are currently in use."
                    )

            # We either made space or we had enough space already
            env = Environment(model=model, configuration=definition)
            logger.debug(f"New environment created for definition {definition}")

        # If deployments target the same model and the model is not in the HF cache, they would all try to download it.
        # This can lead to corruption of the HF cache data (shared PVC).
        # To avoid this situation, we keep track of the models downloaded by the actuator during the current operation.
        # If a deployment wants to download a model for the first time, we do not allow other deployment using the
        # same model to start in parallel.
        # Once the very first download of a model is done we let any number of deployments using the same model to start
        # in parallel as they would only read the model from the cache.
        self.deployment_conflict_manager.maybe_add_deployment(
            k8s_name=env.k8s_name, model=model
        )

        self.in_use_environments[env.k8s_name] = env

        return env

    @property
    def active_environments(self) -> int:
        return len(self.in_use_environments) + len(self.free_environments)

    def get_experiment_pvc_name(self) -> str:
        return self.manager.pvc_name

    def done_creating(self, identifier: str) -> None:
        """
        Report creation
        :param identifier: environment identifier
        :return: None
        """
        self.in_use_environments[identifier].state = EnvironmentState.READY
        model = self.in_use_environments[identifier].model

        self.deployment_conflict_manager.signal(k8s_name=identifier, model=model)

    def cleanup_failed_deployment(self, identifier: str) -> None:
        env = self.in_use_environments[identifier]
        self._delete_environment_k8s_resources(k8s_name=identifier)
        self.done_using(identifier=identifier, reclaim_on_completion=True)
        self.deployment_conflict_manager.signal(
            k8s_name=identifier, model=env.model, error=True
        )

    def get_matching_free_environment(self, configuration: str) -> Environment | None:
        """
        Find a deployment matching a deployment configuration
        :param configuration: The deployment configuration to match
        :return: An already existing deployment or None
        """
        for id, env in enumerate(self.free_environments):
            if env.configuration == configuration:
                del self.free_environments[id]
                return env
        return None

    async def wait_deployment_before_starting(
        self, env: Environment, request_id: str
    ) -> None:
        await self.deployment_conflict_manager.wait(
            request_id=request_id, k8s_name=env.k8s_name, model=env.model
        )

    def done_using(self, identifier: str, reclaim_on_completion: bool = False) -> None:
        """
        Report test completion
        :param definition: environment definition
        :param reclaim_on_completion: flag to indicate the environment is to be completely removed and not freed for later use
        :return: None
        """
        env = self.in_use_environments.pop(identifier)
        if not reclaim_on_completion:
            self.free_environments.append(env)

        # Wake up any other deployment waiting in the queue for a
        # free environment.
        self.environments_queue.signal_next()

    def cleanup(self) -> None:
        """
        Clean up environment
        :return: None
        """
        logger.info("Cleaning environments")
        all_envs = list(self.in_use_environments.values()) + self.free_environments
        for env in all_envs:
            self._delete_environment_k8s_resources(k8s_name=env.k8s_name)

        # We only delete the PVC if it was created by this actuator
        if self.manager.pvc_created:
            logger.debug("Deleting PVC")
            self.manager.delete_pvc()
        else:
            logger.debug("No PVC was created. Nothing to delete!")
