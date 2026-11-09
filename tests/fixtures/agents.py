# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Fixtures providing locations within the repository's ``.agents`` directory."""

import pathlib

import pytest


@pytest.fixture(scope="session")
def agents_dir() -> pathlib.Path:
    """Absolute path to the repository's ``.agents`` directory."""
    return pathlib.Path(__file__).resolve().parent.parent.parent / ".agents"
