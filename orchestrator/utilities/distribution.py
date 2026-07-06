# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""
Module for resolving Python distribution packages from module names.

This utility helps identify which installed package (distribution) a given
Python module belongs to, supporting both standard and editable installations.
"""

import importlib
import importlib.metadata as im
import json
from pathlib import Path


def distribution_from_module(module_name: str) -> str | None:
    """
    Find the distribution package name that contains the specified module.

    This function resolves which installed Python package (distribution) provides
    a given module. It handles both regular (non-editable) and editable installations
    by checking file paths and metadata.

    Args:
        module_name: The fully qualified module name (e.g., 'package.submodule').

    Returns:
        The distribution package name if found, None otherwise.

    Example:
        >>> distribution_from_module('numpy.core')
        'numpy'
    """
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    top_level = module_name.split(".")[0]

    # Get all distributions that provide this top-level package
    for dist_name in im.packages_distributions().get(top_level, []):
        dist = im.distribution(dist_name)

        # Non-editable installs: check if module path matches any file in distribution
        for file in dist.files or []:
            file_path = Path(dist.locate_file(file)).resolve()
            if file_path == module_path:
                return dist.metadata["Name"]

        # Editable installs: check if module is within the source directory
        try:
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                data = json.loads(direct_url)
                if "dir_info" in data:
                    source_root = Path(data["url"].replace("file://", "")).resolve()
                    if module_path.is_relative_to(source_root):
                        return dist.metadata["Name"]
        except Exception:  # noqa: S110
            pass

    # Fallback for src-layout editable installs: packages_distributions() has no
    # entry for the top-level package because hatchling only adds a .pth file
    # (no source files appear in RECORD). Scan all editable distributions and
    # check whether the module lives under their source root.
    # Use the longest (most specific) matching source root to avoid false
    # positives from parent-directory editable installs (e.g. ado-core installed
    # from the repo root would match every plugin module as well).
    most_specific_dist_name: str | None = None
    most_specific_source_root: Path | None = None
    for dist in im.distributions():
        try:
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                data = json.loads(direct_url)
                if "dir_info" in data:
                    source_root = Path(data["url"].replace("file://", "")).resolve()
                    if module_path.is_relative_to(source_root) and (
                        most_specific_source_root is None
                        or len(source_root.parts) > len(most_specific_source_root.parts)
                    ):
                        most_specific_source_root = source_root
                        most_specific_dist_name = dist.metadata["Name"]
        except Exception:  # noqa: PERF203, S110
            pass

    return most_specific_dist_name
