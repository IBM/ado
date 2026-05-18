# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import logging
import os
from importlib.metadata import requires

import ray

logger = logging.getLogger(__name__)


def _get_dependency_from_package_metadata(
    package_name: str, extra_name: str
) -> str | None:
    """
    Extract dependency specification from installed package metadata.

    Args:
        package_name: Name of the package to query (e.g., 'ado-vllm-performance')
        extra_name: Name of the optional dependency group (e.g., 'vllm', 'guidellm')

    Returns:
        Dependency specification from the package's optional-dependencies, or None if not found
    """
    try:
        # Get the package requirements
        # The requires() function returns a list of requirement strings
        # Format: "package>=version; extra == 'extra_name'"
        package_requires = requires(package_name) or []
        for req in package_requires:
            # Parse requirements that match our extra
            if f'extra == "{extra_name}"' in req or f"extra == '{extra_name}'" in req:
                # Extract just the package specification (before the semicolon)
                dep_spec = req.split(";")[0].strip()
                logger.debug(
                    f"Extracted dependency for '{extra_name}' from {package_name} metadata: {dep_spec}"
                )
                return dep_spec
    except Exception as e:
        logger.debug(f"Could not read metadata for package '{package_name}': {e}")

    return None


def inherit_ray_job_env_and_add_extra(base_package_name: str, extra: str) -> dict:
    """
    Build a Ray runtime environment that inherits packages from the job's runtime
    environment and adds an extra dependency by extracting its version from the
    base package's metadata.

    This is useful for actuators that need to add experiment-specific dependencies
    (e.g., 'vllm', 'guidellm') that are defined as extras in the base package.

    The function handles two cases:
    1. Base package in job uv: Reinstalls base package + ado-core + extra
    2. Base package in base env only: Just installs the extra

    Args:
        base_package_name: Name of the base package that defines the extra
                          (e.g., 'ado-vllm-performance')
        extra: Name of the extra package to install (e.g., 'vllm', 'guidellm')

    Returns:
        Runtime environment dict with uv packages list

    Raises:
        RuntimeError: If base package is not found in either job uv or base environment

    Example:
        >>> # If job uv has: ["numpy", "pandas", "ado-vllm-performance==1.2.3"]
        >>> env = inherit_ray_job_env_and_add_extra('ado-vllm-performance', 'vllm')
        >>> # Returns: {"uv": ["numpy", "pandas", "ado-vllm-performance==1.2.3", "vllm>=0.6.0"]}
        >>>
        >>> # If job uv has: ["numpy", "pandas"] (base package in base image)
        >>> env = inherit_ray_job_env_and_add_extra('ado-vllm-performance', 'vllm')
        >>> # Returns: {"uv": ["numpy", "pandas", "vllm>=0.6.0"]}
    """
    # Get the job's runtime environment to inherit its packages
    job_runtime_env = ray.get_runtime_context().runtime_env
    job_uv_config = job_runtime_env.get("uv", {}) if job_runtime_env else {}

    # Extract packages from the uv config
    # Ray normalizes the uv config to a dict with a "packages" key:
    # {"packages": ["pkg1", "pkg2"], "uv_check": false, "uv_pip_install_options": [...]}
    job_uv_packages = job_uv_config.get("packages", []) if job_uv_config else []

    logger.debug(f"Job runtime environment uv packages: {job_uv_packages}")

    # Check if base package is in the job environment
    base_package_in_job = None
    other_packages = []

    # Normalize base package name for comparison (handle both - and _)
    normalized_base = base_package_name.lower().replace("-", "_")

    for pkg in job_uv_packages:
        # Check if this is the base package
        pkg_lower = pkg.lower().replace("-", "_")
        if normalized_base in pkg_lower:
            # Keep the package exactly as specified (with version and extras if any)
            base_package_in_job = pkg
        else:
            other_packages.append(pkg)

    # Try to extract the dependency specification from the base package
    extra_dependency = _get_dependency_from_package_metadata(base_package_name, extra)

    if not extra_dependency:
        # If we can't find the dependency in metadata, the base package might not be installed
        # or the extra doesn't exist
        raise RuntimeError(
            f"Base package '{base_package_name}' does not define extra '{extra}'. "
            f"Ensure the package is installed and the extra is defined in its metadata."
        )

    logger.debug(
        f"Extracted dependency for extra '{extra}' from {base_package_name}: {extra_dependency}"
    )

    # Build the final package list based on whether base package is in job uv
    if base_package_in_job:
        # Case 1: Base package is in job uv
        # Reinstall: other packages + base package (exactly as specified) + extra
        logger.debug(
            f"Base package '{base_package_name}' found in job uv as '{base_package_in_job}', "
            f"reinstalling exactly as specified with extra"
        )
        final_packages = [
            *other_packages,
            base_package_in_job,  # Keep exact specification (version, extras, etc.)
            extra_dependency,
        ]
    else:
        # Case 2: Base package is in base env only
        # Just install: existing packages + extra
        logger.debug(
            f"Base package '{base_package_name}' not in job uv (assumed in base env), "
            f"installing only the extra"
        )
        final_packages = [*job_uv_packages, extra_dependency]

    logger.debug(f"Final package list: {final_packages}")

    return {"uv": final_packages}


def enable_ray_actor_coverage(identifier: str) -> None:
    """For coverage to work with distributed ray actors they need to call this function in their init

    If coverage module is not installed or COVERAGE_PROCESS_START is not defined in the environment
    this function does nothing"""

    if "COVERAGE_PROCESS_START" in os.environ:
        # Don't start multiple times in the same process
        if not globals().get("_coverage_started", False):
            try:
                import coverage
            except ImportError:
                logging.warning(
                    f"{identifier}: COVERAGE_PROCESS_START is defined in the environment but the coverage module is not installed"
                )
            else:
                logging.debug(f"Starting coverage for {identifier}")
                coverage.process_startup()
                globals()["_coverage_started"] = True
        else:
            logging.debug(
                f"Requested to start coverage for {identifier} but _coverage_started is already set"
            )
