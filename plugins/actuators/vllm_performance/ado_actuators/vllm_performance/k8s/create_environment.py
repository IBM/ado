# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging

import pydantic
from ado_actuators.vllm_performance.k8s import K8sDeploymentCreationTimeoutError
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
    image_pull_secret_name: str = "",
    deployment_template: str | None = None,
    service_template: str | None = None,
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
    otlp_traces_endpoint: pydantic.AnyUrl | None = None,
    renderer_num_workers: int | None = None,
    reasoning_parser: str | None = None,
    tool_call_parser: str | None = None,
    language_model_only: bool = False,
    enable_auto_tool_choice: bool = False,
    max_model_len: int | None = None,
    kv_cache_dtype: str | None = None,
    enable_prefix_caching: bool = False,
    check_interval: int = 5,
    timeout: int = 1200,
    deletion_timeout: int = 300,
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
    :param image_pull_secret_name: name of the image pull secret
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
    :param renderer_num_workers: number of renderer workers when threadpool is enabled
        when provided
    :param otlp_traces_endpoint: OpenTelemetry traces endpoint URL
    :param reasoning_parser: vLLM reasoning parser to use (e.g. qwen3)
    :param tool_call_parser: vLLM tool call parser to use (e.g. qwen3_coder)
    :param language_model_only: flag to run vLLM in language-model-only mode
    :param enable_auto_tool_choice: flag to enable automatic tool choice in vLLM
    :param max_model_len: maximum number of tokens the model can process and remember at once
    :param kv_cache_dtype: KV cache data type (e.g. fp8, turboquant_k8v4); if None, vLLM default is used
    :param enable_prefix_caching: flag to enable prefix caching in vLLM
    :param check_interval: wait interval in seconds
    :param timeout: timeout in seconds for deployment readiness
    :param deletion_timeout: timeout in seconds to wait for a deployment to be
        fully deleted after an error before re-raising. If deletion itself
        times out the original error is still re-raised.
    :return:
    """
    if node_selector is None:
        node_selector = {}

    logger.info(f"Creating environment in ns {namespace} with the parameters: ")
    logger.info(
        f"model {model}, in_cluster {in_cluster}, verify_ssl {verify_ssl}, image {image}"
    )
    logger.info(
        f"image_pull_secret_name {image_pull_secret_name}, deployment_template {deployment_template}, "
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

    try:
        # deployment
        c_manager.create_deployment(
            k8s_name=k8s_name,
            model=model,
            gpu_type=gpu_type,
            node_selector=node_selector,
            image=image,
            image_pull_secret_name=image_pull_secret_name,
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
            enforce_eager=enforce_eager,
            skip_tokenizer_init=skip_tokenizer_init,
            io_processor_plugin=io_processor_plugin,
            otlp_traces_endpoint=otlp_traces_endpoint,
            renderer_num_workers=renderer_num_workers,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            language_model_only=language_model_only,
            enable_auto_tool_choice=enable_auto_tool_choice,
            max_model_len=max_model_len,
            kv_cache_dtype=kv_cache_dtype,
            enable_prefix_caching=enable_prefix_caching,
        )

        logger.debug(f"Deployment {k8s_name} created")

        c_manager.wait_deployment_ready(
            k8s_name=k8s_name,
            check_interval=check_interval,
            timeout=timeout,
        )

        logger.info(f"Deployment {k8s_name} ready")

        # service
        c_manager.create_service(k8s_name=k8s_name, template=service_template)
        logger.info(f"Service {k8s_name} created")

    except Exception as e:
        # If the deployment/service creation fail we try to delete the existing
        # resources if created, to make sure we don't re-create existing resources during
        # a retry.
        if isinstance(e, K8sDeploymentCreationTimeoutError):
            logger.debug(f"Creating the {k8s_name} deployment has timed out")
        c_manager.delete_deployment(k8s_name=k8s_name, suppress_not_found_error=True)
        c_manager.delete_service(k8s_name=k8s_name, suppress_not_found_error=True)
        # Wait for the deployment object to disappear before re-raising so that
        # any retry (or the next entity) does not hit a 409 AlreadyExists conflict.
        try:
            c_manager.wait_deployment_deleted(
                k8s_name=k8s_name,
                check_interval=check_interval,
                timeout=deletion_timeout,
            )
        except K8sDeploymentCreationTimeoutError:
            logger.error(
                f"Timed out waiting for deleted deployment {k8s_name} to disappear."
            )
        raise (e)


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
