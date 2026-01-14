# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import logging
import math
import time
import uuid

from ado_actuators.vllm_performance.k8s import (
    K8sConnectionError,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
    VLLMDtype,
)
from kubernetes import client, config
from kubernetes.client import ApiException, V1Deployment

logger = logging.getLogger(__name__)

# These are the most common reasons for a container failure.
container_waiting_error_reasons = [
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
]


class ComponentsManager:
    """
    This class manages K8s operations
    """

    def __init__(
        self,
        namespace: str = "discovery-dev",
        in_cluster: bool = True,
        verify_ssl: bool = True,
        init_pvc: bool = False,
        pvc_name: None | str = None,
        pvc_template: None | str = None,
    ) -> None:
        """
        set up for configuration usage
        :param namespace: cluster namespace to use
        :param in_cluster: flag defining whether we are running in cluster
        :param verify_ssl: flag to verify SSL (self-signed certificate)
        :param init_pvc: flag to decide whether to initialize the PVC for the experiment
        :param pvc_name: the name of the pvc to be created
        :param pvc_template: the name of the template file for creating PVCs
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
            # this is just to make sure we are authenticated to the cluster
            # and fail immediately otherwise.
            self.kube_client_V1.list_namespaced_pod(namespace=namespace)
        except ApiException as error:
            error_message = "Error connecting to the Kubernetes cluster.\n"
            if in_cluster:
                error_message += "Make sure the service account used for the Ray cluster has sufficient permissions."
            else:
                error_message += "Make sure you are authenticated with the Kubernetes cluster or that you are using a valid kubeconfig."
            logger.critical(error_message)
            raise K8sConnectionError from error
        except Exception as e:
            logger.critical(f"Exception connecting to kubernetes {e}")
            raise

        self.namespace = namespace

        # We do this only once when creating a ComponentsManager in the EnvironmentManager because
        # we want the PVC to be shared by all deployments we are testing with the same operation.
        if init_pvc:
            if pvc_name is None:
                # Make sure the PVC we create has a unique name
                self.pvc_name = f"vllm-support-{uuid.uuid4().hex!s}"
                self.create_pvc(pvc_name=self.pvc_name, template=pvc_template)
                self.pvc_created = True
                logger.debug(f"Created pvc {self.pvc_name} in namespace {namespace}")
            else:
                if not self.check_pvc_exists(pvc_name=pvc_name):
                    error_message = (
                        f"The PVC {pvc_name} does not exist in namespace {namespace}."
                    )
                    logger.error(error_message)
                    raise ValueError(error_message)
                self.pvc_name = pvc_name
                self.pvc_created = False
                logger.debug(f"Reusing pvc {pvc_name} from namespace {namespace}")

    @classmethod
    def verify_k8s_auth(cls, namespace: str) -> bool:
        try:
            version_api = client.VersionApi()
            # Fetch cluster version to verify we are connected to the cluster
            version_info = version_api.get_code()
            api = client.CoreV1Api()
            api.list_namespaced_pod(namespace)
            print("Connected to Kubernetes cluster!")
            print(f"Cluster Version: {version_info.git_version}")
            print(f"Platform: {version_info.platform}")

            return True
        except ApiException:
            print("api exception")
            return False
        except Exception as error:
            print("Failed connecting to the Kubernetes cluster")
            raise K8sConnectionError from error

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

    def delete_pvc(self) -> None:
        """
        Delete service for model
        :param pvc_name: pvc name
        :return: boolean
        """
        try:
            self.kube_client_V1.delete_namespaced_persistent_volume_claim(
                namespace=self.namespace, name=self.pvc_name
            )
        except ApiException as e:
            logger.error(f"error deleting pvc {e}")

        logger.debug(f"Deleted pvc {self.pvc_name} from namespace {self.namespace}")

    def create_pvc(self, pvc_name: str, template: None | str = None) -> None:
        """
        create service for model
        :param pvc_name: pvc name
        :param template: yaml template name
        :return:
        """
        try:
            self.kube_client_V1.create_namespaced_persistent_volume_claim(
                namespace=self.namespace,
                body=ComponentsYaml.pvc_yaml(pvc_name=pvc_name, template=template),
            )
        except ApiException as e:
            logger.error(f"error creating pvc  {e}")
            raise

    def check_service_exists(self, k8s_name: str) -> bool:
        """
        Check if service for model exists
        :param k8s_name: kubernetes name
        :return: boolean
        """
        try:
            svcs = self.kube_client_V1.list_namespaced_service(namespace=self.namespace)
        except ApiException as e:
            logger.error(f"error getting service list {e}")
            return False
        return any(svc.metadata.name == k8s_name for svc in svcs.items)

    def delete_service(self, k8s_name: str) -> None:
        """
        Delete service for model
        :param k8s_name: kubernetes name
        :return: boolean
        """
        self.kube_client_V1.delete_namespaced_service(
            namespace=self.namespace,
            name=k8s_name,
        )

    def create_service(self, k8s_name: str, template: str | None = None) -> None:
        """
        create service for model
        :param k8s_name: kubernetes name
        :param template service yaml template
        :return:
        """

        # create service
        try:
            self.kube_client_V1.create_namespaced_service(
                namespace=self.namespace,
                body=ComponentsYaml.service_yaml(k8s_name=k8s_name, template=template),
            )
        except ApiException as e:
            logger.error(f"error creating service  {e}")
            raise

    def check_deployment_exist(self, k8s_name: str) -> bool:
        """
        Check if deployment for model exists
        :param k8s_name: kubernetes name
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
            if deployment.metadata.name == k8s_name:
                return True
        return False

    def delete_deployment(self, k8s_name: str) -> None:
        """
        Delete service for model
        :param k8s_name: kubernetes name
        :return: boolean
        """
        self.kube_client.delete_namespaced_deployment(
            namespace=self.namespace,
            name=k8s_name,
            body=client.V1DeleteOptions(
                propagation_policy="Foreground", grace_period_seconds=5
            ),
        )

    def create_deployment(
        self,
        k8s_name: str,
        model: str,
        gpu_type: str = "NVIDIA-A100-80GB-PCIe",
        node_selector: dict[str, str] | None = None,
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
        template: str | None = None,
        claim_name: str | None = None,
        hf_token: str | None = None,
        enforce_eager: bool = False,
        skip_tokenizer_init: bool = False,
        io_processor_plugin: str | None = None,
    ) -> None:
        """
        create deployment for model
        :param k8s_name: kubernetes name
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
        :param enforce_eager: flag to enforce using Pytorch eager mode
        :param skip_tokenizer_init: flag to skip tokenizer initialization in vLLM
        :param io_processor_plugin: name of the IO processor plugin to be used by vLLM
        :return:
        """
        if node_selector is None:
            node_selector = {}

        # create deployment
        deployment_yaml = ComponentsYaml.deployment_yaml(
            k8s_name=k8s_name,
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
            skip_tokenizer_init=skip_tokenizer_init,
            io_processor_plugin=io_processor_plugin,
            enforce_eager=enforce_eager,
        )
        logger.debug(json.dumps(deployment_yaml, indent=2))

        try:
            self.kube_client.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment_yaml,
            )
        except ApiException as e:
            logger.error(f"error creating deployment  {e}")
            raise

    def _is_pod_failed(self, deployment: V1Deployment) -> bool:
        label_selector = ",".join(
            [f"{k}={v}" for k, v in deployment.spec.selector.match_labels.items()]
        )
        pods = self.kube_client_V1.list_namespaced_pod(
            self.namespace, label_selector=label_selector
        )
        # There's really only one pod in our deployments
        for pod in pods.items:
            pod_name = pod.metadata.name
            pod_phase = pod.status.phase

            # Check if pod phase is Failed
            if pod_phase == "Failed":
                logger.warning(f"Pod {pod_name} is Failed")
                return True

            # Check container statuses for errors
            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    if (
                        cs.state.waiting
                        and cs.state.waiting.reason in container_waiting_error_reasons
                    ):
                        logger.warning(
                            f"Container {cs.name} in pod {pod_name} is in error ({cs.state.waiting.reason})"
                        )
                        return True
        return False

    def _deployment_ready(self, k8s_name: str) -> bool:
        """
        Check whether deployment pod ready
        :param k8s_name: kubernetes name
        :return: boolean
        """
        try:
            deployment = self.kube_client.read_namespaced_deployment(
                namespace=self.namespace,
                name=k8s_name,
            )
        except ApiException as e:
            logger.error(f"error getting deployment  {e}")
            return False
        if self._is_pod_failed(deployment=deployment):
            return False
        if deployment.status.available_replicas is None:
            return False
        return deployment.status.available_replicas == 1

    def wait_deployment_ready(
        self, k8s_name: str, check_interval: int = 5, timeout: int = 1200
    ) -> None:
        """
        Wait for deployment to become ready
        :param k8s_name: kubernetes name
        :param check_interval: wait interval
        :param timeout: timeout
        :return: None
        """
        n_checks = math.ceil(timeout / check_interval)
        for _ in range(n_checks):
            time.sleep(check_interval)
            if self._deployment_ready(k8s_name=k8s_name):
                return
        logger.error("Timed out waiting for deployment to get ready")
        raise Exception("Timed out waiting for deployment to get ready")


if __name__ == "__main__":
    # model
    t_model = "meta-llama/Llama-3.1-8B-Instruct"
    t_k8s_name = ComponentsYaml.get_k8s_name(model=t_model)
    # manager
    c_manager = ComponentsManager(in_cluster=False, verify_ssl=False)
    # pvc
    c_manager.check_pvc_exists(pvc_name="vllm-support")
    # service
    c_manager.create_service(k8s_name=t_k8s_name, reuse=False)
    # deployment
    c_manager.create_deployment(
        k8s_name=t_k8s_name,
        model="meta-llama/Llama-3.1-8B-Instruct",
        claim_name="vllm-support",
        image="quay.io/dataprep1/data-prep-kit/vllm_image:0.1",
    )
    c_manager.wait_deployment_ready(k8s_name=t_k8s_name)
    logger.info("environment is created")
