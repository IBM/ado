# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os

import ado_actuators.sfttrainer.ray_env.utils as utils
import pytest
import ray

import orchestrator.utilities.ray_env.ordered_pip as ordered_pip


@pytest.fixture
def set_plugin():
    os.environ["RAY_RUNTIME_ENV_PLUGINS"] = (
        '[{"class":"' + ordered_pip.OrderedPipPlugin.ClassPath + '"}]'
    )

    yield

    del os.environ["RAY_RUNTIME_ENV_PLUGINS"]


def test_ray_runtime_env_with_ordered_pip_plugin(set_plugin):
    if not utils.is_pip_available():
        pytest.skip("pip is unavailable")

    class_path = ".".join(
        (
            ordered_pip.OrderedPipPlugin.__module__,
            ordered_pip.OrderedPipPlugin.__name__,
        )
    )

    assert class_path == ordered_pip.OrderedPipPlugin.ClassPath

    assert utils.is_ordered_pip_available()

    packages = ["torch==2.6.0", "flash_attn==2.7.4.post1", "mamba-ssm==2.2.5"]

    runtime_env = utils.get_ray_environment(
        packages=packages,
        packages_requiring_extra_phase=[["flash_attn", "mamba-ssm"]],
    )

    assert runtime_env == {
        "env_vars": {"AIM_UI_TELEMETRY_ENABLED": "0", "PIP_NO_BUILD_ISOLATION": "0"},
        "ordered_pip": {
            "phases": [
                {"packages": ["torch==2.6.0"]},
                {"packages": ["flash_attn==2.7.4.post1", "mamba-ssm==2.2.5"]},
            ]
        },
    }


def test_ray_runtime_env_with_vanilla_pip():
    if not utils.is_pip_available():
        pytest.skip("pip is unavailable")

    assert utils.is_ordered_pip_available() is False

    packages = ["torch==2.6.0", "flash_attn==2.7.4.post1", "mamba-ssm==2.2.5"]

    runtime_env = utils.get_ray_environment(
        packages=packages,
        packages_requiring_extra_phase=[["flash_attn", "mamba-ssm"]],
    )

    assert runtime_env == {
        "env_vars": {"AIM_UI_TELEMETRY_ENABLED": "0", "PIP_NO_BUILD_ISOLATION": "0"},
        "pip": {
            "packages": ["torch==2.6.0", "flash_attn==2.7.4.post1", "mamba-ssm==2.2.5"]
        },
    }


def test_ordered_pip_plugin(set_plugin):
    if not utils.is_nvidia_smi_available():
        pytest.skip("there's no NVIDIA gpu on this machine")

    @ray.remote(
        runtime_env={
            "ordered_pip": {
                "phases": [
                    ["torch==2.6.0"],
                    {
                        "packages": ["mamba-ssm==2.2.5"],
                        "pip_install_options": ["--no-build-isolation"],
                    },
                ]
            },
            "env_vars": {
                "LOG_LEVEL": "debug",
                "LOGLEVEL": "debug",
            },
        },
    )
    def try_import_torch():
        import torch

        print(torch.__file__)
        assert torch.__version__ == "2.6.0"
        return True

    assert ray.get(try_import_torch.remote())
