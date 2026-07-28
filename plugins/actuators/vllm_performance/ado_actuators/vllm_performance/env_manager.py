# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import asyncio
import logging
import time
from enum import Enum
from typing import Annotated

import pydantic
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
from pydantic import AfterValidator

from ado.utilities.pydantic import validate_rfc_1123

logger = logging.getLogger(__name__)


class EnvironmentState(Enum):
    """
    Environment state
    """

    NONE = "None"
    READY = "ready"


class Environment(pydantic.BaseModel):
    """
    Environment class representing a deployment environment for a model.

    The k8s_name is automatically generated from the model name and validated
    to be RFC 1123 compliant.
    """

    k8s_name: Annotated[
        str,
        AfterValidator(validate_rfc_1123),
        pydantic.Field(
            description="Kubernetes-compliant name for the deployment, automatically generated from the model name and validated to be RFC 1123 compliant"
        ),
    ] = ""
    state: Annotated[
        EnvironmentState,
        pydantic.Field(description="Current state of the environment (NONE or READY)"),
    ] = EnvironmentState.NONE
    configuration: Annotated[
        str,
        pydantic.Field(
            description="Full deployment configuration as a JSON string containing model, image, GPU/CPU settings, and VLLM parameters"
        ),
    ]
    model: Annotated[
        str,
        pydantic.Field(description="LLM model name (e.g., 'meta-llama/Llama-2-7b-hf')"),
    ]
    freed_at: Annotated[
        float | None,
        pydantic.Field(
            description="Monotonic timestamp (time.monotonic()) of when the environment entered the free pool. None until the environment is freed."
        ),
    ] = None
    delete_attempts: Annotated[
        int,
        pydantic.Field(
            description="Number of GC cycles in which this environment was found still present in K8s after a delete was issued."
        ),
    ] = 0

    @pydantic.model_validator(mode="before")
    @classmethod
    def compute_k8s_name(cls, data: dict) -> dict:
        """
        Compute k8s_name from model if not provided.

        :param data: Input data dictionary
        :return: Data dictionary with k8s_name computed
        """
        if (
            isinstance(data, dict)
            and ("k8s_name" not in data or not data["k8s_name"])
            and "model" in data
        ):
            data["k8s_name"] = ComponentsYaml.get_k8s_name(model=data["model"])
        return data


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
        otlp_traces_endpoint: pydantic.AnyUrl | None = None,
        free_environment_ttl: int = 300,
        gc_force_delete: bool = False,
        gc_force_delete_threshold: int = 3,
    ) -> None:
        """
        Initialize
        :param namespace: deployment namespace
        :param max_concurrent: maximum amount of concurrent environment
        :param in_cluster: flag in cluster
        :param verify_ssl: flag verify SSL
        :param pvc_name: name of the PVC to be created / used
        :param pvc_template: template of the PVC to be created
        :param otlp_traces_endpoint: OpenTelemetry traces endpoint URL
        :param free_environment_ttl: seconds a free environment may idle before
            being garbage collected. 0 means delete immediately on release.
        :param gc_force_delete: if True, issue a force-delete (finalizer
            removal + grace_period=0) when a stuck deployment has been present
            for ``gc_force_delete_threshold`` GC cycles after deletion. The
            environment is then dropped from the watch list. When False the GC
            keeps logging a warning every cycle until the deployment disappears.
            Disabled by default.
        :param gc_force_delete_threshold: number of GC cycles a deployment may
            remain present in K8s after deletion before a force-delete is
            issued. Only used when ``gc_force_delete`` is True.
        """
        self.in_use_environments: dict[str, Environment] = {}
        self.free_environments: list[Environment] = []
        self.deleting_environments: list[Environment] = []
        self.environments_queue = EnvironmentsQueue()
        self.deployment_conflict_manager = DeploymentConflictManager()
        self.namespace = namespace
        self.max_concurrent = max_concurrent
        self.in_cluster = in_cluster
        self.verify_ssl = verify_ssl
        self.otlp_traces_endpoint = otlp_traces_endpoint
        self.free_environment_ttl = free_environment_ttl
        self.gc_force_delete = gc_force_delete
        self.gc_force_delete_threshold = gc_force_delete_threshold
        self._stop_gc = False

        # component manager for cleanup
        self.manager = ComponentsManager(
            namespace=self.namespace,
            in_cluster=self.in_cluster,
            verify_ssl=self.verify_ssl,
            init_pvc=True,
            pvc_name=pvc_name,
            pvc_template=pvc_template,
        )

        # Always start the GC loop so that deletion monitoring and escalation
        # (force-delete) work regardless of TTL setting.
        # When TTL is 0 the free pool is disabled and _gc_free_environments is
        # a no-op, but _gc_confirm_deletions still tracks stuck environments.
        self._gc_task = asyncio.get_event_loop().create_task(self._gc_loop())

    def _delete_environment_k8s_resources(self, k8s_name: str) -> None:
        """
        Deletes a deployment. Intended to be used for cleanup or error recovery
        param: identifier: the deployment identifier
        """
        self.manager.delete_service(k8s_name=k8s_name, suppress_not_found_error=True)

        self.manager.delete_deployment(k8s_name=k8s_name, suppress_not_found_error=True)

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
                        logger.warning(
                            f"Error deleting deployment or service {environment_to_evict.k8s_name}: {e}"
                        )
                        eviction_index += 1
                        continue

                    logger.info(
                        f"deleted environment {environment_to_evict.k8s_name}. "
                        f"Active environments {self.active_environments}"
                    )
                    environment_evicted = True
                    self.deleting_environments.append(environment_to_evict)

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
        self.deleting_environments.append(env)
        self.done_using(identifier=identifier, return_to_pool=False)
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

    def get_otlp_traces_endpoint(self) -> pydantic.AnyUrl | None:
        """
        Get the OTLP traces endpoint
        :return: OTLP traces endpoint URL or None
        """
        return self.otlp_traces_endpoint

    def done_using(self, identifier: str, return_to_pool: bool = True) -> None:
        """
        Report test completion.

        :param identifier: environment identifier
        :param return_to_pool: if True (default) the environment is returned to
            the free pool (or deleted immediately when TTL=0). If False the
            environment slot is simply released without entering the free pool,
            used when K8s resources have already been cleaned up externally.
        :return: None
        """
        env = self.in_use_environments.pop(identifier)
        if return_to_pool:
            if self.free_environment_ttl == 0:
                # TTL=0 means delete immediately.
                # Still track it in deleting_environments so the GC can
                # monitor and escalate if the K8s resources get stuck.
                logger.info(
                    f"TTL=0: deleting environment {identifier} immediately on release"
                )
                self._delete_environment_k8s_resources(k8s_name=identifier)
                self.deleting_environments.append(env)
            else:
                env.freed_at = time.monotonic()
                self.free_environments.append(env)

        # Wake up any other deployment waiting in the queue for a
        # free environment.
        self.environments_queue.signal_next()

    def _gc_free_environments(self) -> None:
        """
        Delete and remove free environments that have been idle longer than
        ``free_environment_ttl`` seconds.

        Only called from ``_gc_loop`` when ``free_environment_ttl > 0``.
        """
        now = time.monotonic()
        stale = [
            (i, env, env.freed_at)
            for i, env in enumerate(self.free_environments)
            if env.freed_at is not None
            and now - env.freed_at >= self.free_environment_ttl
        ]

        if not stale:
            return

        to_remove: set[int] = set()
        for i, env, freed_at in stale:
            try:
                self._delete_environment_k8s_resources(k8s_name=env.k8s_name)
                logger.info(
                    f"GC: deleted stale free environment {env.k8s_name} "
                    f"(idle {now - freed_at:.1f}s >= TTL {self.free_environment_ttl}s)"
                )
                to_remove.add(i)
                self.deleting_environments.append(env)
            except ApiException as e:  # noqa: PERF203
                logger.warning(
                    f"GC: failed to delete K8s resources for {env.k8s_name}: {e}. "
                    "Keeping in free list to retry next GC cycle."
                )

        self.free_environments = [
            env for i, env in enumerate(self.free_environments) if i not in to_remove
        ]

        # Wake any waiter that was blocked because the pool appeared full.
        self.environments_queue.signal_next()

    def _gc_confirm_deletions(self) -> None:
        """
        Check each environment in ``deleting_environments`` and remove it from
        the list once the K8s deployment is confirmed gone.

        When ``gc_force_delete`` is False the GC logs a warning every cycle
        until the deployment disappears on its own. When ``gc_force_delete`` is
        True, a force-delete (finalizer removal + grace_period=0) is issued
        after ``gc_force_delete_threshold`` cycles and the environment is then
        dropped from the watch list.
        """
        still_deleting: list[Environment] = []
        for env in self.deleting_environments:
            if not self.manager.check_deployment_exist(k8s_name=env.k8s_name):
                logger.debug(
                    f"GC: deployment {env.k8s_name} confirmed deleted from K8s."
                )
                continue

            env.delete_attempts += 1

            if (
                self.gc_force_delete
                and env.delete_attempts >= self.gc_force_delete_threshold
            ):
                logger.warning(
                    f"GC: deployment {env.k8s_name} still present after "
                    f"{env.delete_attempts} recheck(s). Issuing force-delete."
                )
                try:
                    self.manager.force_delete_environment(k8s_name=env.k8s_name)
                except ApiException as e:
                    logger.error(f"GC: force-delete failed for {env.k8s_name}: {e}")
                    # Deletion failed so we keep it in the list for the CG to check again at the next round.
                    still_deleting.append(env)

                continue

            logger.warning(
                f"GC: deployment {env.k8s_name} was deleted but is still "
                f"present in K8s (attempt {env.delete_attempts}). "
                "Will recheck next GC cycle."
            )
            still_deleting.append(env)
        self.deleting_environments = still_deleting

    async def _gc_loop(self) -> None:
        """
        Background async task that periodically checks deletions and
        garbage-collects stale free environments.
        """
        # When TTL is 0 there is no free pool, so use 60s as the poll interval
        # for deletion monitoring. Otherwise poll at TTL frequency (capped at 60s).
        poll_interval = (
            min(self.free_environment_ttl, 60) if self.free_environment_ttl > 0 else 60
        )
        while not self._stop_gc:
            await asyncio.sleep(poll_interval)
            self._gc_confirm_deletions()
            if self.free_environment_ttl > 0:
                self._gc_free_environments()

    def cleanup(self) -> None:
        """
        Clean up environment
        :return: None
        """
        self._stop_gc = True
        # ensuring the GC task is cleanly stopped
        self._gc_task.cancel()
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
