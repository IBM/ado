# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

from ado_actuators.vllm_performance.k8s.manage_components import (
    ComponentsManager,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    ComponentsYaml,
    VLLMDtype,
)

logger = logging.getLogger(__name__)


def create_test_environment(
    k8s_name: str,
    model: str,
    pvc_name: str,
    in_cluster: bool = True,
    verify_ssl: bool = False,
    image: str = "vllm/vllm-openai:v0.6.3",
    image_secret: str = "",
    deployment_template: None | str = None,
    service_template: None | str = None,
    n_gpus: int = 1,
    gpu_type: str = "NVIDIA-A100-80GB-PCIe",
    node_selector: dict[str, str] | None = None,
    n_cpus: int = 8,
    memory: str = "128Gi",
    max_batch_tokens: int = 16384,
    gpu_memory_utilization: float = 0.9,
    dtype: VLLMDtype = VLLMDtype.AUTO,
    cpu_offload: int = 0,
    max_num_seq: int = 256,
    hf_token: str | None = None,
    namespace: str = "vllm-testing",
    enforce_eager: bool = False,
    skip_tokenizer_init: bool = False,
    io_processor_plugin: str | None = None,
    check_interval: int = 5,
    timeout: int = 1200,
) -> None:
    """
    Create test deployment
    :param k8s_name: unique K8s name
    :param model: LLM model name
    :param namespace: namespace to use for deployment
    :param pvc_name: name of the pvc to be used
    :param in_cluster: flag - running in cluster
    :param verify_ssl:  flag - verify ssl
    :param image: image to use in deployment
    :param image_secret: name of the image pull secret
    :param deployment_template: deployment template
    :param service_template: service template
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
    :param enforce_eager: flag to enforce using Pytorch eager mode
    :param skip_tokenizer_init: flag to skip tokenizer initialization in vLLM
    :param io_processor_plugin: name of the IO processor plugin to be used by vLLM
    :param check_interval: wait interval in seconds
    :param timeout: timeout in seconds
    :return:
    """
    if node_selector is None:
        node_selector = {}

    logger.info(f"Creating environment in ns {namespace} with the parameters: ")
    logger.info(
        f"model {model}, in_cluster {in_cluster}, verify_ssl {verify_ssl}, image {image}"
    )
    logger.info(
        f"image_secret {image_secret}, deployment_template {deployment_template}, "
        f"service_template {service_template}, pvc_name {pvc_name}"
    )
    logger.info(f"n_gpus {n_gpus}, gpu_type {gpu_type}, n_cpus {n_cpus}")
    logger.info(f"node selector {node_selector}")
    logger.info(
        f"memory {memory}, max_batch_tokens {max_batch_tokens}, gpu_memory_utilization {gpu_memory_utilization}"
    )
    logger.info(f"dtype {dtype}, cpu_offload {cpu_offload}, max_num_seq {max_num_seq}")

    # manager
    c_manager = ComponentsManager(
        namespace=namespace,
        in_cluster=in_cluster,
        verify_ssl=verify_ssl,
    )
    logger.debug("component manager created")

    # deployment
    c_manager.create_deployment(
        k8s_name=k8s_name,
        model=model,
        gpu_type=gpu_type,
        node_selector=node_selector,
        image=image,
        image_secret=image_secret,
        n_gpus=n_gpus,
        n_cpus=n_cpus,
        memory=memory,
        template=deployment_template,
        claim_name=pvc_name,
        hf_token=hf_token,
        enforce_eager=enforce_eager,
        skip_tokenizer_init=skip_tokenizer_init,
        io_processor_plugin=io_processor_plugin,
    )
    logger.debug("deployment created")
    c_manager.wait_deployment_ready(
        k8s_name=k8s_name,
        check_interval=check_interval,
        timeout=timeout,
    )
    logger.info("deployment ready")
    # service
    c_manager.create_service(k8s_name=k8s_name, template=service_template)
    logger.info("service created")


if __name__ == "__main__":
    t_model = "meta-llama/Llama-3.1-8B-Instruct"
    create_test_environment(
        k8s_name=ComponentsYaml.get_k8s_name(model=t_model),
        in_cluster=False,
        verify_ssl=False,
        model=t_model,
        pvc_name="vllm-support",
        image="quay.io/dataprep1/data-prep-kit/vllm_image:0.1",
    )
