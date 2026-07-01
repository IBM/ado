# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib

import pytest

from tests.fixtures.ado_cli_isolation import apply_isolated_ado_app_dir


@pytest.fixture(autouse=True)
def isolated_ado_app_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """Isolate ado CLI config for every test in this package."""
    apply_isolated_ado_app_dir(tmp_path, monkeypatch)
    return tmp_path
