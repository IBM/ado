# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.cli.utils.generic.wrappers import get_sql_store
from orchestrator.cli.utils.pydantic.metadata_merge import (
    merge_configuration_metadata_dicts,
)
from orchestrator.core import SampleStoreResource
from orchestrator.core.metadata import ConfigurationMetadata
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext


def test_merge_configuration_metadata_dicts_preserves_name_merges_labels() -> None:
    base = ConfigurationMetadata(
        name="keep-me", description="d", labels={"a": "1"}
    ).model_dump()
    patch = {"labels": {"b": "2"}}
    merged = merge_configuration_metadata_dicts(base, patch)
    assert merged["name"] == "keep-me"
    assert merged["labels"] == {"a": "1", "b": "2"}


def test_merge_configuration_metadata_dicts_labels_from_none() -> None:
    base = ConfigurationMetadata(name=None, labels=None).model_dump()
    patch = {"labels": {"x": "y"}}
    merged = merge_configuration_metadata_dicts(base, patch)
    assert merged["labels"] == {"x": "y"}


def test_ado_edit_metadata_mutex(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    patch_file = tmp_path / "m.yaml"
    patch_file.write_text("labels:\n  k: v\n")

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "edit",
            "samplestore",
            "dummy",
            "--metadata",
            str(patch_file),
            "--editor",
            "vim",
        ],
    )
    assert result.exit_code == 1
    assert "may not be used together" in result.output


def test_ado_edit_metadata_merges_into_store(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    store = random_sample_store_resource_from_file()
    store.config.metadata = ConfigurationMetadata(name="orig", labels={"team": "ado"})

    sql = get_sql_store(project_context=valid_ado_project_context)
    sql.addResource(store)

    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("labels:\n  run: 'ci'\n")

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "edit",
            "samplestore",
            store.identifier,
            "--metadata",
            str(patch_file),
        ],
    )
    assert result.exit_code == 0, result.output

    updated = sql.getResource(
        identifier=store.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert updated is not None
    assert updated.config.metadata.name == "orig"
    assert updated.config.metadata.labels == {"team": "ado", "run": "ci"}


def test_ado_edit_metadata_rejects_non_mapping_yaml(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    store = random_sample_store_resource_from_file()
    sql = get_sql_store(project_context=valid_ado_project_context)
    sql.addResource(store)

    patch_file = tmp_path / "bad.yaml"
    patch_file.write_text("- not\n- a\n- mapping\n")

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "edit",
            "samplestore",
            store.identifier,
            "--metadata",
            str(patch_file),
        ],
    )
    assert result.exit_code == 1
    assert "must contain a YAML mapping" in result.output
