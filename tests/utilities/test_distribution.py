# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for resolving editable distribution metadata."""

import json
import pathlib
import sys

import pytest

from ado.utilities.distribution import distribution_from_module


@pytest.mark.parametrize("listed_top_level", [True, False])
def test_distribution_from_editable_file_uri(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    listed_top_level: bool,
) -> None:
    """Resolve an editable distribution from a file URI containing spaces."""
    module_name = "ado_editable_distribution_fixture"
    source_root = tmp_path / "editable project"
    module_root = source_root / "src" / module_name
    module_root.mkdir(parents=True)
    (module_root / "__init__.py").write_text("", encoding="utf-8")

    site_packages = tmp_path / "site-packages"
    metadata = site_packages / "editable_distribution_fixture-1.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: editable-distribution-fixture\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata / "direct_url.json").write_text(
        json.dumps({"url": source_root.as_uri(), "dir_info": {"editable": True}}),
        encoding="utf-8",
    )
    if listed_top_level:
        (metadata / "top_level.txt").write_text(module_name, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site_packages))
    monkeypatch.syspath_prepend(str(source_root / "src"))

    try:
        assert distribution_from_module(module_name) == "editable-distribution-fixture"
    finally:
        sys.modules.pop(module_name, None)
