# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os

import ado_actuators.sfttrainer.ray_env.utils as utils
import pytest
import ray

import orchestrator.utilities.ray_env.ordered_pip as ordered_pip


@pytest.fixture
def set_plugin() -> None:
    os.environ["RAY_RUNTIME_ENV_PLUGINS"] = (
        '[{"class":"' + ordered_pip.OrderedPipPlugin.ClassPath + '"}]'
    )

    yield

    del os.environ["RAY_RUNTIME_ENV_PLUGINS"]


def test_detect_support_pip_install_options() -> None:
    import ray

    version = tuple(int(x) for x in ray.__version__.split("."))
    import ado_actuators.sfttrainer.wrapper_fms_hf_tuning.tuning_versions as tv

    supported = tv.semver_cmp(version, (2, 50, 0)) >= 0

    assert supported == utils.ray_version_supports_pip_install_options()


def test_ray_runtime_env_with_ordered_pip_plugin(set_plugin: None) -> None:
    if not utils.is_pip_available():
        pytest.skip("pip is unavailable")

    class_path = f"{ordered_pip.OrderedPipPlugin.__module__}.{ordered_pip.OrderedPipPlugin.__name__}"

    assert class_path == ordered_pip.OrderedPipPlugin.ClassPath

    assert utils.is_ordered_pip_available()

    packages = [
        "torch==2.6.0",
        "flash_attn==2.7.4.post1",
        "mamba-ssm==2.2.5",
    ]

    runtime_env = utils.get_ray_environment(
        packages=packages,
        packages_requiring_extra_phase=[utils.packages_depending_on_torch()],
        env_vars={},
    )

    if utils.ray_version_supports_pip_install_options():
        assert runtime_env == {
            "env_vars": {"AIM_UI_TELEMETRY_ENABLED": "0"},
            "ordered_pip": {
                "phases": [
                    {
                        "packages": ["torch==2.6.0"],
                        "pip_install_options": ["--no-build-isolation"],
                    },
                    {
                        "packages": ["flash_attn==2.7.4.post1", "mamba-ssm==2.2.5"],
                        "pip_install_options": ["--no-build-isolation"],
                    },
                ]
            },
        }
    else:
        assert runtime_env == {
            "env_vars": {
                "AIM_UI_TELEMETRY_ENABLED": "0",
                "PIP_NO_BUILD_ISOLATION": "0",
            },
            "ordered_pip": {
                "phases": [
                    {"packages": ["torch==2.6.0"]},
                    {
                        "packages": [
                            "flash_attn==2.7.4.post1",
                            "mamba-ssm==2.2.5",
                        ]
                    },
                ]
            },
        }


def test_pip_find_links_option() -> None:
    if not utils.is_pip_available():
        pytest.skip("pip is unavailable")

    packages = ["mamba-ssm==2.2.5"]

    wheelhouse = "file:///path/to/wheelhouse"
    runtime_env = utils.get_ray_environment(
        packages=packages,
        packages_requiring_extra_phase=[utils.packages_depending_on_torch()],
        env_vars={"PIP_FIND_LINKS": wheelhouse},
    )

    if utils.ray_version_supports_pip_install_options():
        assert runtime_env == {
            "env_vars": {
                "AIM_UI_TELEMETRY_ENABLED": "0",
                "PIP_FIND_LINKS": wheelhouse,
            },
            "pip": {
                "packages": packages,
                "pip_install_options": [
                    "--no-build-isolation",
                    "--find-links",
                    wheelhouse,
                ],
            },
        }
    else:
        assert runtime_env == {
            "env_vars": {
                "AIM_UI_TELEMETRY_ENABLED": "0",
                "PIP_NO_BUILD_ISOLATION": "0",
                "PIP_FIND_LINKS": wheelhouse,
            },
            "pip": {"packages": packages},
        }


def test_ray_runtime_env_with_vanilla_pip() -> None:
    if not utils.is_pip_available():
        pytest.skip("pip is unavailable")

    assert utils.is_ordered_pip_available() is False

    packages = [
        "torch==2.6.0",
        "flash_attn==2.7.4.post1",
        "mamba-ssm==2.2.5",
    ]

    runtime_env = utils.get_ray_environment(
        packages=packages,
        packages_requiring_extra_phase=[utils.packages_depending_on_torch()],
        env_vars={},
    )

    if utils.ray_version_supports_pip_install_options():
        assert runtime_env == {
            "env_vars": {"AIM_UI_TELEMETRY_ENABLED": "0"},
            "pip": {
                "packages": [
                    "torch==2.6.0",
                    "flash_attn==2.7.4.post1",
                    "mamba-ssm==2.2.5",
                ],
                "pip_install_options": ["--no-build-isolation"],
            },
        }
    else:
        assert runtime_env == {
            "env_vars": {
                "AIM_UI_TELEMETRY_ENABLED": "0",
                "PIP_NO_BUILD_ISOLATION": "0",
            },
            "pip": {
                "packages": [
                    "torch==2.6.0",
                    "flash_attn==2.7.4.post1",
                    "mamba-ssm==2.2.5",
                ]
            },
        }


def test_ordered_pip_plugin(set_plugin: None) -> None:
    if not utils.is_pip_available():
        pytest.skip("pip is unavailable")

    @ray.remote(
        runtime_env={
            "ordered_pip": {
                "phases": [
                    ["pyyaml"],
                    {
                        "packages": ["filelock"],
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
    def try_import_packages() -> bool:
        import yaml

        _ = dir(yaml)

        import filelock

        _ = dir(filelock)

        return True

    import importlib.metadata

    installed_packages = importlib.metadata.distributions()
    installed_packages = sorted([pkg.metadata["Name"] for pkg in installed_packages])

    if ("pyyaml" in installed_packages) and ("filelock" in installed_packages):
        pytest.skip("pyyaml and filelock are both already installed")

    assert ray.get(try_import_packages.remote())
