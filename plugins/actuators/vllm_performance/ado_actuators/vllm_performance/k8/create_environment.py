# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

from ado_actuators.vllm_performance.k8.manage_components import (
    ComponentsManager,
)
from ado_actuators.vllm_performance.k8.yaml_support.build_components import (
    ComponentsYaml,
    VLLMDtype,
)

logger = logging.getLogger(__name__)


def create_test_environment(
    k8_name: str,
    model: str,
    in_cluster: bool = True,
    verify_ssl: bool = False,
    image: str = "vllm/vllm-openai:v0.6.3",
    image_secret: str = "",
    deployment_template: str = "deployment.yaml",
    service_template: str = "service.yaml",
    pvc_template: str = "pvc.yaml",
    n_gpus: int = 1,
    gpu_type: str = "NVIDIA-A100-80GB-PCIe",
    node_selector: dict[str, str] = {},
    n_cpus: int = 8,
    memory: str = "128Gi",
    max_batch_tokens: int = 16384,
    gpu_memory_utilization: float = 0.9,
    dtype: VLLMDtype = VLLMDtype.AUTO,
    cpu_offload: int = 0,
    max_num_seq: int = 256,
    hf_token: str | None = None,
    reuse_service: bool = True,
    reuse_deployment: bool = True,
    reuse_pvc: bool = True,
    pvc_name: str = "vllm-support",
    namespace: str = "vllm-testing",
) -> None:
    """
    Create test deployment
    :param k8_name: unique k8 name
    :param model: LLM model name
    :param namespace: namespace to use for deployment
    :param in_cluster: flag - running in cluster
    :param verify_ssl:  flag - verify ssl
    :param image: image to use in deployment
    :param image_secret: name of the image pull secret
    :param deployment_template: deployment template
    :param service_template: service template
    :param pvc_template: pvc template
    :param n_gpus: number of GPUs
    :param gpu_type: type of the GPU to use
    :param node_selector: optional node selector
    :param n_cpus: number of CPUs
    :param memory: pod memory
    :param max_batch_tokens: Vllm parameter - maximum number of batched tokens per iteration
    :param gpu_memory_utilization: VLLM parameter - GPU memory utilization
    :param dtype: VLLM parameter - data type for model weights and activations
    :param cpu_offload: VLLM parameter - the space in GiB to offload to CPU, per GPU
    :param max_num_seq: VLLM parameter - Maximum number of sequences per iteration.
    :param hf_token: huggingface token
    :param reuse_service: flag to reuse deployment
    :param reuse_deployment: flag to reuse deployment
    :param reuse_pvc: flag to reuse VPC
    :param pvc_name: PVC name
    :return:
    """
    logger.info(f"Creating environment in ns {namespace} with the parameters: ")
    logger.info(
        f"model {model}, in_cluster {in_cluster}, verify_ssl {verify_ssl}, image {image}"
    )
    logger.info(
        f"image_secret {image_secret}, deployment_template {deployment_template}, "
        f"service_template {service_template}"
    )
    logger.info(
        f"pvc_template {pvc_template}, n_gpus {n_gpus}, gpu_type {gpu_type}, n_cpus {n_cpus}"
    )
    logger.info(f"node selector {node_selector}")
    logger.info(
        f"memory {memory}, max_batch_tokens {max_batch_tokens}, gpu_memory_utilization {gpu_memory_utilization}"
    )
    logger.info(
        f"dtype {dtype}, cpu_offload {cpu_offload}, max_num_seq {max_num_seq}, pvc_name {pvc_name}"
    )
    logger.info(
        f"reuse_service {reuse_service}, reuse_deployment {reuse_deployment}, reuse_pvc {reuse_pvc}"
    )

    # manager
    c_manager = ComponentsManager(
        namespace=namespace,
        in_cluster=in_cluster,
        verify_ssl=verify_ssl,
    )
    logger.debug("component manager created")
    # create PVC
    c_manager.create_pvc(pvc_name=pvc_name, template=pvc_template, reuse=reuse_pvc)
    logger.info("pvc created")
    # deployment
    c_manager.create_deployment(
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
        template=deployment_template,
        claim_name=pvc_name,
        hf_token=hf_token,
        reuse=reuse_deployment,
    )
    logger.debug("deployment created")
    c_manager.wait_deployment_ready(k8_name=k8_name)
    logger.info("deployment ready")
    # service
    c_manager.create_service(
        k8_name=k8_name, template=service_template, reuse=reuse_service
    )
    logger.info("service created")


if __name__ == "__main__":
    t_model = "meta-llama/Llama-3.1-8B-Instruct"
    create_test_environment(
        k8_name=ComponentsYaml.get_k8_name(model=t_model),
        in_cluster=False,
        verify_ssl=False,
        model=t_model,
        pvc_name="vllm-support",
        hf_token="token",
        image="quay.io/dataprep1/data-prep-kit/vllm_image:0.1",
    )
