# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import functools
import importlib.util
import os
import sys
import types


def semver_cmp(v1: tuple[int, ...], v2: tuple[int, ...]) -> int:
    """Compares 2 semver versions.

    They must consist of the same number of integers

    Args:
        v1:
            the first version
        v2:
            the second version

    Returns:
        * +1 if v1 >  v2
        * -1 if v1 <  v2
        *  0 if v1 == v2

    Raises:
        NotImplementedError:
            If the two semver versions do not consist of the same number of integers
    """
    max_len = max(len(v1), len(v2))

    if max_len != len(v1):
        v1 = tuple(list(v1) + [0] * (max_len - len(v1)))

    if max_len != len(v2):
        v2 = tuple(list(v2) + [0] * (max_len - len(v2)))

    if v1 < v2:
        return -1
    if v2 < v1:
        return +1

    return 0


def semver_parse(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in version.split("."))


def import_from_path(module_name: str, file_path: str) -> types.ModuleType:

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _select_compatible_version(
    candidates: list[tuple[int, ...]], version: tuple[int, ...]
) -> tuple[int, ...]:
    """Picks the thin wrapper which is compatible with the desired version

    The idea is to select the most recent @candidate which supports the desired version - works similar to
    insertion sort.

    Args:
        candidates:
            A list of thin wrapper versions - they must all have the same length.
            Each entry in the candidate maps to the MINIMUM version it supports. You can think of these entries,
            as the times that the fms-hf-tuning API broke backwards compatibility.
        version:
            The desired version

    Returns:
        The candidate which best matches the desired version
    """
    candidates = sorted(candidates, key=functools.cmp_to_key(semver_cmp), reverse=True)

    # VV: Pick the file with the highest compatible version
    # Each file handles codes that have at least version X.Y.Z. We iterate an array of files
    # which is sorted in descending order (i.e. highest versions are earlier in the array).
    # We stop when we find a file whose version is less than or equal to the requested version

    for c in candidates:
        comparison = semver_cmp(c, version)
        if comparison == 0 or comparison == -1:
            return c

    min_supported = ".".join(map(str, candidates[-1]))
    version = ".".join(map(str, version))

    raise NotImplementedError(
        f"Requested fms-hf-tuning version {version} is too old. Supported versions are >={min_supported}"
    )


def get_wrapper_name_for_version(
    version: str,
    path_to_thin_wrappers_directory: str,
) -> str:
    """Returns the name of the wrapper script for a fms-hf-tuning version

    Args:
        version:
            The version string, must be in the form %d(.%d)*
        path_to_thin_wrappers_directory:
            The path to the directory containing the thin wrappers

    Returns:
        A string with the name of the python wrapper script (e.g. "at_least_3_0_0_1.py")
    """
    version = semver_parse(version)

    # VV: Make a list of all the candidate files, they are in the form of "at_least_$semver.py"
    candidates: list[tuple[int, ...]] = []

    for name in os.listdir(path_to_thin_wrappers_directory):
        path = os.path.join(path_to_thin_wrappers_directory, name)
        if not os.path.isfile(path) or not (
            name.startswith("at_least_") and name.endswith(".py")
        ):
            continue

        candidates.append(
            tuple(
                int(x)
                for x in name.removeprefix("at_least_").removesuffix(".py").split("_")
            )
        )

    compatible_version = _select_compatible_version(
        candidates=candidates, version=version
    )

    return "".join(("at_least_", "_".join(map(str, compatible_version))))


def import_tuning_version(
    version: str,
    path_to_thin_wrappers_directory: str | None = None,
) -> types.ModuleType:
    """Loads the appropriate thin wrapper to fms-hf-tuning based on the desired version

    Args:
        version:
            The version string, must be in the form %d(.%d)*
        path_to_thin_wrappers_directory:
            The path to the directory containing the thin wrappers. When None uses the directory of this
            file.

    Returns:
        The python module
    """
    if path_to_thin_wrappers_directory is None:
        path_to_thin_wrappers_directory = os.path.dirname(__file__)

    module_name = get_wrapper_name_for_version(
        version,
        path_to_thin_wrappers_directory=path_to_thin_wrappers_directory,
    )

    filename = f"{module_name}.py"
    wrapper_file = os.path.join(
        path_to_thin_wrappers_directory,
        filename,
    )

    print("Loading fms-hf-tuning thin wrapper", module_name, file=sys.stderr)

    return import_from_path(module_name, wrapper_file)
