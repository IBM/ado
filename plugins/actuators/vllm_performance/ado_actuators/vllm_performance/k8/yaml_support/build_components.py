# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import sys
import uuid
from enum import Enum
from typing import Any

import yaml

PVC_MOUNT_PATH = "/dev/cache"
PVC_NAME = "vllm-support"
logger = logging.getLogger(__name__)


class VLLMDtype(Enum):
    """
    Type for VLLM
    """

    AUTO = "auto"
    HALF = "half"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT = "float"
    FLOAT32 = "float32"


class ComponentsYaml:
    """
    Build components Yamls class
    """

    @staticmethod
    def get_k8_name(model: str) -> str:
        """
        convert model for kubernetes usage
        :param model: LLM model
        :return: k8 unique name for a given LLM model
        """
        m_parts = model.split("/")
        return f"vllm-{m_parts[-1].lower()}-{uuid.uuid4().hex}".replace(".", "-")

    @staticmethod
    def _adjust_file_name(f: str) -> str:
        """
        Adjust file name to local directory, if required
        :param f: file name
        :return: adjusted file name
        """
        if os.path.isfile(f):
            return f
        return os.path.abspath(os.path.join(os.path.dirname(__file__), f))

    @staticmethod
    def deployment_yaml(
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
    ) -> dict[str, Any]:
        """
        Generate deployment yaml
        :param k8_name: deployment name
        :param model: model name
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
        :return:
        """
        # read template
        ComponentsYaml._adjust_file_name(template)
        try:
            with open(ComponentsYaml._adjust_file_name(template)) as file:
                deployment_yaml = yaml.safe_load(file)
        except Exception as exception:
            logger.error(f"Exception reading deployment yaml template {exception}")
            sys.exit(1)

        # Update metadata
        metadata = deployment_yaml["metadata"]
        metadata["name"] = k8_name
        metadata["labels"]["app.kubernetes.io/instance"] = k8_name

        # update spec
        spec = deployment_yaml["spec"]
        # selector
        spec["selector"]["matchLabels"]["app.kubernetes.io/instance"] = k8_name

        # update template
        d_template = spec["template"]
        # template metadata
        d_template["metadata"]["labels"]["app.kubernetes.io/instance"] = k8_name

        # update template spec
        spec = d_template["spec"]
        # node selector
        spec["nodeSelector"]["nvidia.com/gpu.product"] = gpu_type
        if len(node_selector) > 0:
            spec["nodeSelector"].update(node_selector)
        # image pull secret
        if image_secret is not None and image_secret != "":
            spec["imagePullSecrets"] = [{"name": image_secret}]
        # volumes
        if claim_name is not None:
            spec["volumes"].extend(
                [{"name": PVC_NAME, "persistentVolumeClaim": {"claimName": claim_name}}]
            )

        # container
        container = spec["containers"][0]
        # image
        container["image"] = image
        # resources
        requests = container["resources"]["requests"]
        requests["cpu"] = str(n_cpus)
        requests["memory"] = memory
        requests["nvidia.com/gpu"] = str(n_gpus)
        limits = container["resources"]["limits"]
        limits["cpu"] = str(n_cpus)
        limits["memory"] = memory
        limits["nvidia.com/gpu"] = str(n_gpus)
        # env variables to to set parameters for docker execution
        container["env"] = [
            {"name": "MODEL", "value": model},
            {"name": "GPU_MEMORY_UTILIZATION", "value": str(gpu_memory_utilization)},
            {"name": "DTYPE", "value": dtype.value},
            {"name": "CPU_OFFLOAD_GB", "value": str(cpu_offload)},
            {"name": "MAX_NUM_BATCHED_TOKENS", "value": str(max_batch_tokens)},
            {"name": "MAX_NUM_SEQ", "value": str(max_num_seq)},
            {"name": "TENSOR_PARALLEL_SIZE", "value": str(n_gpus)},
        ]
        if hf_token is not None:
            container["env"].extend([{"name": "HF_TOKEN", "value": hf_token}])
        if claim_name is not None:
            container["env"].extend(
                [
                    {
                        "name": "HF_HUB_CACHE",
                        "value": f"{PVC_MOUNT_PATH}/transformers_cache",
                    },
                ]
            )
        # volume mounts
        if claim_name is not None:
            container["volumeMounts"].extend(
                [
                    {"name": PVC_NAME, "mountPath": PVC_MOUNT_PATH},
                ]
            )

        # return
        return deployment_yaml

    @staticmethod
    def service_yaml(k8_name: str, template: str = "service.yaml") -> dict[str, Any]:
        """
        Generate service yaml for a given model
        :param k8_name: k8 unique name
        :param template: template for service yaml
        :return: service yaml
        """
        # read template
        try:
            with open(ComponentsYaml._adjust_file_name(template)) as file:
                service_yaml = yaml.safe_load(file)
        except Exception as exception:
            logger.error(f"Exception reading service yaml template {exception}")
            sys.exit(1)

        # Update metadata
        metadata = service_yaml["metadata"]
        metadata["name"] = k8_name
        metadata["labels"]["app.kubernetes.io/instance"] = k8_name

        # update selector
        service_yaml["spec"]["selector"]["app.kubernetes.io/instance"] = k8_name

        # return
        return service_yaml

    @staticmethod
    def pvc_yaml(pvc_name: str, template: str = "pvc.yaml") -> dict[str, Any]:
        """
        Generate pvc yaml
        :param pvc_name: name of the PVC claim
        :param template: template for pvc yaml
        :return: pvc yaml
        """
        # read template
        try:
            with open(ComponentsYaml._adjust_file_name(template)) as file:
                pvc_yaml = yaml.safe_load(file)
        except Exception as exception:
            logger.error(f"Exception reading pvc yaml template {exception}")
            sys.exit(1)

        # Update metadata
        pvc_yaml["metadata"]["name"] = pvc_name

        # return
        return pvc_yaml


if __name__ == "__main__":
    t_model = "meta-llama/Llama-3.1-8B-Instruct"
    t_k8_name = ComponentsYaml.get_k8_name(model=t_model)
    deployment = ComponentsYaml.deployment_yaml(
        k8_name=t_k8_name,
        model=t_model,
        claim_name="vllm-support",
        node_selector={"kubernetes.io/hostname": "cpu16"},
        hf_token="token",
        image="quay.io/dataprep1/data-prep-kit/vllm_image:0.1",
    )
    print(f"Deployment YAML: \n{yaml.dump(deployment)}")
    service = ComponentsYaml.service_yaml(k8_name=t_k8_name)
    print(f"Service YAML: \n{yaml.dump(service)}")
    pvc = ComponentsYaml.pvc_yaml(pvc_name="vllm-pvc")
    print(f"PVC YAML: \n{yaml.dump(pvc)}")
