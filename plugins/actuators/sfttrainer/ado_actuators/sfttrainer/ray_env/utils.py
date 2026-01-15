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


def packages_requiring_nvidia_development_binaries() -> list[str]:
    return [
        "fms-acceleration-foak",
        "fms-acceleration-moe",
        "triton",
        "flash_attn",
        "mamba-ssm",
        "causal-conv1d",
        # VV: mamba_ssm and causal_conv1d changed their package names
        "mamba_ssm",
        "causal_conv1d",
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-cufile-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-cusparselt-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvshmem-cu12",
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


@functools.cache
def ray_version_supports_pip_install_options() -> bool | None:
    import ray

    try:
        # VV: Ray added support for pip_install_options in 2.50.0
        version = [int(x) for x in ray.__version__.split(".")]
        return version[0] > 2 or (version[0] >= 2 and version[1] >= 50)
    except Exception as e:
        print(
            f"Unable to tell whether pip_install_options is available for Ray Runtime environments due to {e!s} - "
            f"will assume that it is unavailable"
        )
        return None


def packages_depending_on_torch() -> list[str]:
    # VV: mamba_ssm and causal_conv1d changed their package names
    return ["flash_attn", "mamba-ssm", "causal-conv1d", "mamba_ssm", "causal_conv1d"]


def get_ray_environment(
    packages: list[str],
    packages_requiring_extra_phase: list[list[str]],
    env_vars: dict[str, str],
    insert_pip_install_options: bool | None = None,
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
        env_vars:
            Environment variables to inject into the RuntimeContext
        insert_pip_install_options:
            Whether to insert pip_install_options fields. If set to None the method will auto-decide based on
            the version of ray by invoking the function ray_version_supports_pip_install_options()
    Returns:
        A dictionary representing a RuntimeContext for Ray jobs
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
    env = {"AIM_UI_TELEMETRY_ENABLED": "0"}
    env.update(env_vars)

    ray_environment = {
        "env_vars": env,
        env_plugin_name: plugin,
    }

    pip_install_options = []

    if insert_pip_install_options is None:
        insert_pip_install_options = ray_version_supports_pip_install_options()

    if insert_pip_install_options or env_plugin_name == "uv":
        # VV: Ray added support for pip_install_options in 2.50.0
        pip_install_options = ["--no-build-isolation"]

        if env.get("PIP_FIND_LINKS"):
            # VV: I find that exporting PIP_FIND_LINKS does not behave the same way as using --find-links
            pip_install_options.extend(("--find-links", env["PIP_FIND_LINKS"]))

    if env_plugin_name == "pip":
        phase = {"packages": packages}

        if pip_install_options and insert_pip_install_options:
            phase["pip_install_options"] = pip_install_options
        else:
            ray_environment["env_vars"]["PIP_NO_BUILD_ISOLATION"] = "0"

        plugin.update(phase)
    elif env_plugin_name == "uv":
        pip_install_options.insert(0, "--no-build-isolation")
        plugin.update(
            {"uv_pip_install_options": pip_install_options, "packages": packages}
        )
    elif env_plugin_name == "ordered_pip":
        # VV: Keeps the linter happy
        base_packages = []
        phases = []

        plugin["phases"] = phases

        for p in packages_requiring_extra_phase or []:
            packages, this_phase = apply_exclude_package_rules(
                exclude_packages=p, packages=packages
            )
            if this_phase:
                phase = {"packages": this_phase}
                if insert_pip_install_options and pip_install_options:
                    phase["pip_install_options"] = list(pip_install_options)
                phases.append(phase)

        # VV: At this point the packages var contains all the packages that must go into the very first phase
        base_packages.extend(packages)

        base_phase = {"packages": base_packages}
        if insert_pip_install_options and pip_install_options:
            base_phase["pip_install_options"] = list(pip_install_options)
        phases.insert(0, base_phase)
    else:
        raise NotImplementedError("Unknown ray environment env plugin", env_plugin_name)

    return ray_environment
