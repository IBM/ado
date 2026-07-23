# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import logging
import os

import ray
from packaging.requirements import Requirement

logger = logging.getLogger(__name__)


def extract_package_specs_from_job_env(
    package_names: list[str],
) -> dict[str, dict[str, str | None]]:
    """
    Parse package specifications from the Ray job's uv environment.

    Given a list of package names, returns information about those packages
    if they are present in the worker job's uv venv.

    Args:
        package_names: List of package names to look for (e.g., ['ado-vllm-performance', 'numpy'])

    Returns:
        Dictionary mapping package names to their specifications:
        {
            "package-name": {
                "source": "package-name" or "/path/to/wheel.whl" (no version or extras),
                "version": "==1.2.3" or None,
                "extras": "extra1,extra2" or None
            }
        }
        Only includes packages that are found in the job uv environment.

    Example:
        >>> # If job uv has: ["numpy>=1.20", "ado-vllm-performance[vllm]==1.2.3", "/path/to/custom.whl"]
        >>> result = parse_job_uv_packages(['numpy', 'ado-vllm-performance', 'custom'])
        >>> # Returns: {
        >>> #     "numpy": {"source": "numpy", "version": ">=1.20", "extras": None},
        >>> #     "ado-vllm-performance": {"source": "ado-vllm-performance", "version": "==1.2.3", "extras": "vllm"},
        >>> #     "custom": {"source": "/path/to/custom.whl", "version": None, "extras": None}
        >>> # }
    """
    # Get the job's runtime environment
    job_runtime_env = ray.get_runtime_context().runtime_env
    job_uv_config = job_runtime_env.get("uv", {}) if job_runtime_env else {}
    job_uv_packages = job_uv_config.get("packages", []) if job_uv_config else []

    logger.debug(f"Parsing job uv packages: {job_uv_packages}")

    result = {}

    # Iterate over requested package names
    for requested_name in package_names:
        # Search for this package in the job uv packages
        for pkg_spec in job_uv_packages:
            # Quick substring pre-filter: skip specs that clearly don't contain the name.
            # Check the name as-is and with all dashes replaced by underscores.
            pkg_spec_lower = pkg_spec.lower()
            name_lower = requested_name.lower()
            name_with_underscores = requested_name.replace("-", "_").lower()
            if (
                name_lower not in pkg_spec_lower
                and name_with_underscores not in pkg_spec_lower
            ):
                continue  # Not a match, skip to next package

            # Check if it's a wheel file path
            if pkg_spec.endswith(".whl") or "/" in pkg_spec:
                # It's a wheel file path - can't use Requirement parser.
                # Extract name by stripping extras/version from the path basename.
                source = pkg_spec.split("[")[0] if "[" in pkg_spec else pkg_spec
                parsed_name = source.rstrip("/").rsplit("/", 1)[-1]
                extras = None
                if "[" in pkg_spec:
                    extras = pkg_spec.split("[", 1)[1].split("]")[0]
                version = None
            else:
                # It's a PyPI package — use Requirement for an exact name match.
                req = Requirement(pkg_spec)
                parsed_name = req.name
                source = req.name
                extras = ",".join(req.extras) if req.extras else None
                # Convert specifier to string (e.g., "==1.2.3")
                version = str(req.specifier) if req.specifier else None

            # Exact name check (normalise dashes/underscores per PEP 503).
            normalised_parsed = parsed_name.lower().replace("-", "_")
            normalised_requested = name_lower.replace("-", "_")
            if normalised_parsed != normalised_requested:
                continue  # Substring matched but names differ (e.g. "vllm" vs "ado-vllm-performance")

            result[requested_name] = {
                "source": source,
                "version": version,
                "extras": extras,
            }
            break  # Found the package, move to next requested name

    logger.debug(f"Parsed package info: {result}")
    return result


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
