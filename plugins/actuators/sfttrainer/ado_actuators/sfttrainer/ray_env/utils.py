# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import contextlib
import functools
import os
import typing


@functools.cache
def is_using_arm_cpu() -> bool:
    """Returns True when device is using an ARM cpu"""
    import platform

    platform_machine = platform.machine().lower()

    return platform_machine == "aarch64" or "arm" in platform_machine


@functools.cache
def is_nvcc_available() -> bool:
    """Returns True when nvcc is in $PATH"""
    import shutil

    return shutil.which("nvcc") is not None


@functools.cache
def is_uv_available() -> bool:
    """Returns True when uv is in $PATH"""
    import shutil

    return shutil.which("uv") is not None


@functools.cache
def is_nvidia_smi_available() -> bool:
    """Returns True when nvidia-smi is in $PATH"""
    import shutil

    return shutil.which("nvidia-smi") is not None


@functools.cache
def is_pip_available() -> bool:
    """Returns True when pip is import-able"""
    import importlib.util

    spec = importlib.util.find_spec("pip")
    return spec is not None and spec.loader is not None


# VV: Need to include the ray_runtime_env_plugins_string parameter here for caching to work
@functools.cache
def _check_if_ray_will_load_ordered_pip_plugin(
    ray_runtime_env_plugins: str | None,
) -> bool:
    import json

    from orchestrator.utilities.ray_env.ordered_pip import OrderedPipPlugin

    if not is_pip_available():
        return False

    with contextlib.suppress(Exception):
        decoded = json.loads(ray_runtime_env_plugins)

        for entry in decoded:
            if (
                isinstance(entry, dict)
                and "class" in entry
                and entry["class"] == OrderedPipPlugin.ClassPath
            ):
                return True

    return False


def is_ordered_pip_available() -> bool:
    """Returns True when ray is configured to load the ordered_pip RuntimeEnvPlugin and pip is importable"""
    return _check_if_ray_will_load_ordered_pip_plugin(
        os.environ.get("RAY_RUNTIME_ENV_PLUGINS")
    )


def packages_requiring_nvidia_development_binaries():
    return [
        "fms-acceleration-foak",
        "fms-acceleration-moe",
        "triton",
        "flash_attn",
        "mamba-ssm",
        "causal-conv1d",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvtx-cu12",
    ]


def apply_exclude_package_rules(
    exclude_packages: list[str], packages: list[str]
) -> tuple[list[str], list[str]]:
    """Filters out packages based on a list of exclusion rules.

    Args:
        exclude_packages:
            A list of rules for excluding package names
        packages:
            A list of packages to filter

    Returns:
        A tuple with 2 items.
            The first item is the list containing only the packages that do not match any of the exclusion rules
            The second item is the list of packages that were removed
    """
    if not exclude_packages:
        return packages, []

    ret = []
    removed = []

    for candidate_package in packages:
        # VV: Some packages look like this: "name @ file://...."
        trimmed = candidate_package.replace(" ", "")
        for unwanted_package in exclude_packages:
            if (
                trimmed.startswith((f"{unwanted_package}=", f"{unwanted_package}@"))
                or trimmed == unwanted_package
            ):
                removed.append(candidate_package)
                break
        else:
            # VV: Keep the original package
            ret.append(candidate_package)

    return ret, removed


def get_pinned_packages(
    path_requirements: str,
    override_fms_hf_tuning: str | None = None,
    ensure_aim: bool = True,
    exclude_packages: list[str] | None = None,
) -> list[str]:
    """Extracts the pinned packages from a path_requirements file

    Args:
        path_requirements:
            Path to the requirements.txt file containing the pinned packages (one package per line, a-la pip)
        override_fms_hf_tuning:
            If set, overrides the `fms-hf-tuning` pinned package from the contents of the requirements.txt file
        ensure_aim:
            If set, ensures that the dependencies include the aim python package.
        exclude_packages:
            Packages to exclude from installation
    Returns:
        An array consisting of pinned packages a-la pip
    """

    with open(path_requirements, encoding="utf-8") as f:
        packages = [x.strip() for x in f if x.strip() and not x.startswith("#")]

    def find_matching_packages(package_name: str, packages: list[str]) -> list[str]:
        return [
            x
            for x in packages
            if x.startswith((f"{package_name}=", f"{package_name} "))
            or x == package_name
        ]

    if override_fms_hf_tuning:
        exclude_packages = [*list(exclude_packages or []), "fms-hf-tuning"]

    packages, _dropped = apply_exclude_package_rules(exclude_packages, packages)

    if override_fms_hf_tuning:
        packages.append(override_fms_hf_tuning)

    if ensure_aim and len(find_matching_packages("aim", packages)) == 0:
        packages.append("aim")

    return packages


def get_ray_environment(
    packages: list[str],
    packages_requiring_extra_phase: list[list[str]],
) -> dict[str, typing.Any]:
    """Builds a ray-environment using a Ray RuntimeEnvPlugin.

    The function picks the RuntimeEnvPlugin by inspecting the host machine and virtual environment.
    It selects the plugins using the following priority:

    1. ordered_pip
    2. pip
    3. uv

    Args:
        packages:
            The list of python packages to install
        packages_requiring_extra_phase:
            A list of lists of packages. The list with index i expects that the packages in the list with index i-1
            have already been installed in the virtual environment that ray will be building.
            This is only used when the ordered_pip RuntimeEnvPlugin is available. Otherwise, it is ignored.
    Returns:
        A dictionary representing a Ray environment
    """
    if is_ordered_pip_available():
        env_plugin_name = "ordered_pip"
    elif is_pip_available():
        env_plugin_name = "pip"
    elif is_uv_available():
        env_plugin_name = "uv"
    else:
        raise NotImplementedError(
            "No uv binary in $PATH, pip cannot be imported, and ordered_pip is not configured. "
            "Ensure your virtual environment is valid."
        )

    # VV: Do not switch on pip_check.
    plugin = {}
    ray_environment = {
        "env_vars": {"AIM_UI_TELEMETRY_ENABLED": "0"},
        env_plugin_name: plugin,
    }

    if env_plugin_name == "pip":
        ray_environment["env_vars"]["PIP_NO_BUILD_ISOLATION"] = "0"
        plugin.update({"packages": packages})
    elif env_plugin_name == "uv":
        plugin.update(
            {"uv_pip_install_options": ["--no-build-isolation"], "packages": packages}
        )
    elif env_plugin_name == "ordered_pip":
        # VV: TODO For ray 2.49+ we can also set "pip_install_options"= ["--no-build-isolation"]
        ray_environment["env_vars"]["PIP_NO_BUILD_ISOLATION"] = "0"
        base_packages = []
        phases = [{"packages": base_packages}]
        plugin["phases"] = phases

        for p in packages_requiring_extra_phase or []:
            packages, this_phase = apply_exclude_package_rules(
                exclude_packages=p, packages=packages
            )
            if this_phase:
                phases.append({"packages": this_phase})

        # VV: At this point the packages var contains all the packages that must go into the very first phase
        base_packages.extend(packages)
    else:
        raise NotImplementedError("Unknown ray environment env plugin", env_plugin_name)

    return ray_environment
