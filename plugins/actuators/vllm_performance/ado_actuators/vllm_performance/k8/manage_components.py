# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import math
import time

from ado_actuators.vllm_performance.k8.yaml_support.build_components import (
    ComponentsYaml,
    VLLMDtype,
)
from kubernetes import client, config
from kubernetes.client import ApiException

logger = logging.getLogger(__name__)


class ComponentsManager:
    """
    This class manages k8 operations
    """

    def __init__(
        self,
        namespace: str = "discovery-dev",
        in_cluster: bool = True,
        verify_ssl: bool = True,
    ):
        """
        set up for configuration usage
        :param namespace: cluster namespace to use
        :param in_cluster: flag defining whether we are running in cluster
        :param verify_ssl: flag to verify SSL (self-signed certificate)
        """
        try:
            if in_cluster:
                config.load_incluster_config()
            else:
                config.load_kube_config()
            if not verify_ssl:
                configuration = client.configuration.Configuration.get_default_copy()
                configuration.verify_ssl = False
                client.Configuration.set_default(configuration)
            self.kube_client_V1 = client.CoreV1Api()
            self.kube_client = client.AppsV1Api()
            self.namespace = namespace
        except Exception as exception:
            logger.error(f"Exception connecting to kubernetes {exception}")
            raise

    def check_pvc_exists(self, pvc_name: str) -> bool:
        """
        Check if PVC exists
        :param pvc_name: pvc name
        :return: boolean
        """
        try:
            pvcs = self.kube_client_V1.list_namespaced_persistent_volume_claim(
                namespace=self.namespace
            )
        except ApiException as e:
            logger.error(f"error getting pvc list {e}")
            return False
        return any(pvc.metadata.name == pvc_name for pvc in pvcs.items)

    def delete_pvc(self, pvc_name: str) -> None:
        """
        Delete service for model
        :param pvc_name: pvc name
        :return: boolean
        """
        try:
            self.kube_client_V1.delete_namespaced_persistent_volume_claim(
                namespace=self.namespace, name=pvc_name
            )
        except ApiException as e:
            logger.error(f"error deleting pvc {e}")

    def create_pvc(
        self, pvc_name: str, template: str = "pvc.yaml", reuse: bool = True
    ) -> None:
        """
        create service for model
        :param pvc_name: pvc name
        :param template: yaml template name
        :param reuse: reuse if exists
        :return:
        """
        # try to reuse existing one if exists
        exists = self.check_pvc_exists(pvc_name=pvc_name)
        if exists and reuse:
            return
        if exists and not reuse:
            # delete it first
            self.delete_pvc(pvc_name=pvc_name)
            # make sure that deletion is completed
            deleting = True
            for _ in range(150):
                deleting = self.check_pvc_exists(pvc_name=pvc_name)
                if not deleting:
                    break
                time.sleep(1)
            if deleting:
                logger.error("Did not complete PVC deletion")
                raise
        # create pvc
        try:
            self.kube_client_V1.create_namespaced_persistent_volume_claim(
                namespace=self.namespace,
                body=ComponentsYaml.pvc_yaml(pvc_name=pvc_name, template=template),
            )
        except ApiException as e:
            logger.error(f"error creating pvc  {e}")
            raise

    def check_service_exists(self, k8_name: str) -> bool:
        """
        Check if service for model exists
        :param k8_name: kubernetes name
        :return: boolean
        """
        try:
            svcs = self.kube_client_V1.list_namespaced_service(namespace=self.namespace)
        except ApiException as e:
            logger.error(f"error getting service list {e}")
            return False
        return any(svc.metadata.name == k8_name for svc in svcs.items)

    def delete_service(self, k8_name: str) -> None:
        """
        Delete service for model
        :param k8_name: kubernetes name
        :return: boolean
        """
        try:
            self.kube_client_V1.delete_namespaced_service(
                namespace=self.namespace,
                name=k8_name,
            )
        except ApiException as e:
            logger.error(f"error deleting service {e}")

    def create_service(
        self, k8_name: str, template: str = "service.yaml", reuse: bool = False
    ) -> None:
        """
        create service for model
        :param k8_name: kubernetes name
        :param template service yaml template
        :param reuse: reuse if exists
        :return:
        """
        # try to reuse existing one if exists
        exists = self.check_service_exists(k8_name=k8_name)
        if exists and reuse:
            return
        if exists and not reuse:
            # delete it first
            self.delete_service(k8_name=k8_name)
            # make sure that deletion is completed
            deleting = True
            for _ in range(150):
                deleting = self.check_service_exists(k8_name=k8_name)
                if not deleting:
                    break
                time.sleep(1)
            if deleting:
                logger.error("Did not complete Service deletion")
        # create service
        try:
            self.kube_client_V1.create_namespaced_service(
                namespace=self.namespace,
                body=ComponentsYaml.service_yaml(k8_name=k8_name, template=template),
            )
        except ApiException as e:
            logger.error(f"error creating service  {e}")
            raise

    def check_deployment_exist(self, k8_name: str) -> bool:
        """
        Check if deployment for model exists
        :param k8_name: kubernetes name
        :return: boolean
        """
        try:
            deployments = self.kube_client.list_namespaced_deployment(
                namespace=self.namespace
            )
        except ApiException as e:
            logger.error(f"error getting deployment list {e}")
            return False
        for deployment in deployments.items:
            if deployment.metadata.name == k8_name:
                return True
        return False

    def delete_deployment(self, k8_name: str) -> None:
        """
        Delete service for model
        :param k8_name: kubernetes name
        :return: boolean
        """
        try:
            self.kube_client.delete_namespaced_deployment(
                namespace=self.namespace,
                name=k8_name,
                body=client.V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                ),
            )
        except ApiException as e:
            logger.error(f"error deleting deployment {e}")

    def create_deployment(
        self,
        k8_name: str,
        model: str,
        gpu_type: str = "NVIDIA-A100-80GB-PCIe",
        node_selector: dict[str, str] = {},
        image: str = "vllm/vllm-openai:v0.6.3",
        image_secret: str = "",
        n_gpus: int = 1,
        n_cpus: int = 8,
        memory: str = "128Gi",
        max_batch_tokens: int = 16384,
        gpu_memory_utilization: float = 0.9,
        dtype: VLLMDtype = VLLMDtype.AUTO,
        cpu_offload: int = 0,
        max_num_seq: int = 256,
        template: str = "deployment.yaml",
        claim_name: str | None = None,
        hf_token: str | None = None,
        reuse: bool = False,
    ) -> None:
        """
        create deployment for model
        :param k8_name: kubernetes name
        :param model: LLM model name
        :param gpu_type: gpu type, for example NVIDIA-A100-80GB-PCIe, Tesla-V100-PCIE-16GB, etc.
        :param node_selector: optional node selector
        :param image: image name to use
        :param image_secret: name of the image pull secret
        :param n_gpus: number of GPUs to use in VLLM
        :param n_cpus: number of CPUs for VLLM pod
        :param memory: memory for VLLM pod
        :param max_batch_tokens: Vllm parameter - maximum number of batched tokens per iteration
        :param gpu_memory_utilization: VLLM parameter - GPU memory utilization
        :param dtype: VLLM parameter - data type for model weights and activations
        :param cpu_offload: VLLM parameter - the space in GiB to offload to CPU, per GPU
        :param max_num_seq: VLLM parameter - Maximum number of sequences per iteration.
        :param template: template for deployment yaml
        :param claim_name: PVC name
        :param hf_token: huggingface token
        :param reuse: reuse if exists
        :return:
        """
        # try to reuse existing one if exists
        exists = self.check_deployment_exist(k8_name=k8_name)
        if exists and reuse:
            return
        if exists and not reuse:
            # delete it first
            self.delete_deployment(k8_name=k8_name)
            # make sure that deletion is completed
            deleting = True
            for _ in range(150):
                deleting = self.check_deployment_exist(k8_name=k8_name)
                if not deleting:
                    break
                time.sleep(1)
            if deleting:
                logger.error("Did not complete deployment deletion")
                raise
        # create deployment
        try:
            self.kube_client.create_namespaced_deployment(
                namespace=self.namespace,
                body=ComponentsYaml.deployment_yaml(
                    k8_name=k8_name,
                    model=model,
                    gpu_type=gpu_type,
                    node_selector=node_selector,
                    image=image,
                    image_secret=image_secret,
                    n_gpus=n_gpus,
                    n_cpus=n_cpus,
                    memory=memory,
                    max_batch_tokens=max_batch_tokens,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=dtype,
                    cpu_offload=cpu_offload,
                    max_num_seq=max_num_seq,
                    template=template,
                    claim_name=claim_name,
                    hf_token=hf_token,
                ),
            )
        except ApiException as e:
            logger.error(f"error creating deployment  {e}")
            raise

    def _deployment_ready(self, k8_name: str) -> bool:
        """
        Check whether deployment pod ready
        :param k8_name: kubernetes name
        :return: boolean
        """
        try:
            deployment = self.kube_client.read_namespaced_deployment(
                namespace=self.namespace,
                name=k8_name,
            )
        except ApiException as e:
            logger.error(f"error getting deployment  {e}")
            return False
        if deployment.status.available_replicas is None:
            return False
        return deployment.status.available_replicas == 1

    def wait_deployment_ready(
        self, k8_name: str, check_interval: int = 5, timeout: int = 1200
    ) -> None:
        """
        Wait for deployment to become ready
        :param k8_name: kubernetes name
        :param check_interval: wait interval
        :param timeout: timeout
        :return: None
        """
        n_checks = math.ceil(timeout / check_interval)
        for _ in range(n_checks):
            time.sleep(check_interval)
            if self._deployment_ready(k8_name=k8_name):
                return
        logger.error("Timed out waiting for deployment to get ready")
        raise Exception("Timed out waiting for deployment to get ready")


if __name__ == "__main__":
    # model
    t_model = "meta-llama/Llama-3.1-8B-Instruct"
    t_k8_name = ComponentsYaml.get_k8_name(model=t_model)
    # manager
    c_manager = ComponentsManager(in_cluster=False, verify_ssl=False)
    # pvc
    c_manager.check_pvc_exists(pvc_name="vllm-support")
    # service
    c_manager.create_service(k8_name=t_k8_name, reuse=False)
    # deployment
    c_manager.create_deployment(
        k8_name=t_k8_name,
        model="meta-llama/Llama-3.1-8B-Instruct",
        claim_name="vllm-support",
        hf_token="token",
        image="quay.io/dataprep1/data-prep-kit/vllm_image:0.1",
    )
    c_manager.wait_deployment_ready(k8_name=t_k8_name)
    logger.info("environment is created")
