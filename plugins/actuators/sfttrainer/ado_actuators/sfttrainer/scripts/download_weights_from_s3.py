# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import os.path

import ray
import yaml


@ray.remote
def patch_weights() -> None:
    # VV: Last time I checked, the llama 7b gptq weights on the s3 bucket are broken
    path = "/hf-models-pvc/LLaMa/models/hf/7B-gptq/quantize_config.json"
    if not os.path.isfile(path):
        with open(path, "w") as f:
            json.dump(
                {
                    "bits": 4,
                    "group_size": 128,
                    "damp_percent": 0.01,
                    "desc_act": 0,
                    "static_groups": False,
                    "sym": True,
                    "true_sequential": True,
                    "model_name_or_path": None,
                    "model_file_base_name": None,
                    "is_marlin_format": False,
                    "quant_method": "gptq",
                },
                f,
            )


@ray.remote
def download(
    uri: str, dest: str, secret_access_key: str, access_key_id: str, endpoint: str
) -> None:
    import subprocess
    import sys

    env = os.environ.copy()

    env.update(
        {
            "AWS_SECRET_ACCESS_KEY": secret_access_key,
            "AWS_ACCESS_KEY_ID": access_key_id,
        }
    )

    command = ["aws", "s3", "--endpoint", endpoint, "sync", uri, dest]

    print(command)

    proc = subprocess.Popen(  # noqa: S603
        command, stdout=sys.stderr, stderr=sys.stderr, env=env
    )
    proc.wait()

    if proc.returncode != 0:
        raise ValueError(f"Download failed with returncode {proc.returncode}")


@ray.remote
def upload(
    src: str, uri: str, secret_access_key: str, access_key_id: str, endpoint: str
) -> None:
    import subprocess
    import sys

    env = os.environ.copy()

    env.update(
        {
            "AWS_SECRET_ACCESS_KEY": secret_access_key,
            "AWS_ACCESS_KEY_ID": access_key_id,
        }
    )

    command = ["aws", "s3", "--endpoint", endpoint, "sync", src, uri]

    print(command)

    proc = subprocess.Popen(  # noqa: S603
        command, stdout=sys.stderr, stderr=sys.stderr, env=env
    )
    proc.wait()

    if proc.returncode != 0:
        raise ValueError(f"Upload failed with returncode {proc.returncode}")


def main() -> None:
    ray.init()

    warning = """The file {path} should look like this:
bucket: fmaas-integration-tests
endpoint: https://s3.us-east.cloud-object-storage.appdomain.cloud/
access_key_id: <something>
secret_access_key: <something>

The credentials are in the `SFTTrainer` vault on 1Password, if you don't have access to it
ask Vassilis Vassiliadis or Alessandro Pomponio to grant you access"""

    try:
        with open("fm-openshift_credentials.yaml") as f:
            fm_openshift_credentials = yaml.safe_load(f)
    except Exception:
        print(warning.format(path="fm-openshift_credentials.yaml"))
        raise

    try:
        with open("fmas-integration-tests-credentials.yaml") as f:
            fmas_integration_tests_credentials = yaml.safe_load(f)
    except Exception:
        print(warning.format(path="fmas-integration-tests-credentials.yaml"))
        raise

    # VV: These are the models we download from a place other than huggingface
    fm_openshift_paths = {
        "/hf-models-pvc/LLaMa/models/hf/llama2-70b/": "models/external/llama2/70B/",
        "/hf-models-pvc/LLaMa/models/hf/llama3-70b/": "models/external/llama3/70b_pre_trained/",
        "/hf-models-pvc/granite-8b-code-base/": "models/ibm-granite-hf/granite-8b-code-base/",
        "/hf-models-pvc/granite-34b-code-base/": "models/ibm-granite-hf/granite-34b-code-base/",
        "/hf-models-pvc/Mixtral-8x7B-Instruct-v0.1/": "models/external/mistralai/Mixtral-8x7B-Instruct-v0.1/",
    }

    # VV: These are all the GPTQ weights
    # VV: format is {"destination in pvc": "source in bucket"}
    fmas_integration_tests_paths = {
        "/hf-models-pvc/granite-34b-gptq/": "models/granite-34b-gptq/",
        "/hf-models-pvc/LLaMa/models/hf/llama3-70b-gptq/": "models/llama3-70b-gptq/",
        "/hf-models-pvc/mixtral_8x7b_instruct_v0.1_gptq": "models/mixtral_8x7b_instruct_v0.1_gptq/",
        "/hf-models-pvc/mistral-7B-v0.3-gptq": "models/mistral-7B-v0.3-gptq/",
        "/hf-models-pvc/granite-20b-code-base-v2/step_280000_ckpt-gptq/": "models/granite-20b-gptq/",
        "/hf-models-pvc/LLaMa/models/hf/7B-gptq/": "models/llama-2-7b-gptq/",
        "/hf-models-pvc/granite-8b-code-instruct-gptq/": "models/granite-8b-code-instruct-gptq/",
        "/hf-models-pvc/LLaMa/models/hf/llama3.1-405b-gptq": "models/llama3-405b-gptq/",
        "/hf-models-pvc/granite-7b-base-gtpq/": "models/granite_7b_base_gptq/",
    }

    for creds, models in [
        (fmas_integration_tests_credentials, fmas_integration_tests_paths),
        (fm_openshift_credentials, fm_openshift_paths),
    ]:
        for dest, src in models.items():
            print(f"Download {dest} from {src}")
            if not dest.endswith("/"):
                dest += "/"
            if not src.endswith("/"):
                src += "/"

            ray.get(
                download.remote(
                    uri=os.path.join("s3://", creds["bucket"], src),
                    dest=dest,
                    access_key_id=creds["access_key_id"],
                    secret_access_key=creds["secret_access_key"],
                    endpoint=creds["endpoint"],
                )
            )

            if dest == "/hf-models-pvc/LLaMa/models/hf/7B-gptq/":
                ray.get(patch_weights.remote())


if __name__ == "__main__":
    main()
