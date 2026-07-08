# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.resources.handlers import (
    strategic_merge_configuration_metadata,
)
from ado.core import SampleStoreResource
from ado.core.metadata import ConfigurationMetadata
from ado.core.resources import CoreResourceKinds
from ado.metastore.project import ProjectContext


def test_strategic_merge_preserves_name_merges_labels() -> None:
    base = ConfigurationMetadata(
        name="keep-me", description="d", labels={"a": "1"}
    ).model_dump()
    patch = {"labels": {"b": "2"}}
    merged = strategic_merge_configuration_metadata(base, patch)
    assert merged["name"] == "keep-me"
    assert merged["labels"] == {"a": "1", "b": "2"}


def test_strategic_merge_labels_from_none() -> None:
    base = ConfigurationMetadata(name=None, labels=None).model_dump()
    patch = {"labels": {"x": "y"}}
    merged = strategic_merge_configuration_metadata(base, patch)
    assert merged["labels"] == {"x": "y"}


def test_ado_edit_mutex_patch_and_patch_file(
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
    f = tmp_path / "m.yaml"
    f.write_text("labels:\n  k: v\n")

    result = runner.invoke(
        ado,
        [
            "edit",
            "samplestore",
            "dummy",
            "-p",
            "labels: {a: b}",
            "--patch-file",
            str(f),
        ],
    )
    assert result.exit_code == 1
    assert "only one of" in result.output.lower()


def test_ado_edit_editor_ignored_with_patch_file(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """Test that --editor flag is ignored (not rejected) when --patch-file is used."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Create a sample store to edit
    store = random_sample_store_resource_from_file()
    store.config.metadata = ConfigurationMetadata(labels={"original": "value"})
    sql = get_sql_store(project_context=valid_ado_project_context)
    sql.addResource(store)

    # Create patch file
    patch_file = tmp_path / "m.yaml"
    patch_file.write_text("labels:\n  patched: 'yes'\n")

    # Run edit with both --patch-file and --editor
    result = runner.invoke(
        ado,
        [
            "edit",
            "samplestore",
            store.identifier,
            "--patch-file",
            str(patch_file),
            "--editor",
            "vim",
        ],
    )
    # Should succeed (exit code 0) - editor is ignored, not rejected
    assert result.exit_code == 0, result.output

    # Verify the patch was applied (editor was ignored)
    updated = sql.getResource(
        identifier=store.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert updated is not None
    assert updated.config.metadata.labels == {"original": "value", "patched": "yes"}


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
            "edit",
            "samplestore",
            store.identifier,
            "--patch-file",
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


def test_ado_edit_inline_patch(
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
    store.config.metadata = ConfigurationMetadata(labels={"a": "1"})
    sql = get_sql_store(project_context=valid_ado_project_context)
    sql.addResource(store)

    result = runner.invoke(
        ado,
        [
            "edit",
            "samplestore",
            store.identifier,
            "-p",
            "labels: {b: '2'}",
        ],
    )
    assert result.exit_code == 0, result.output
    updated = sql.getResource(
        identifier=store.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert updated is not None
    assert updated.config.metadata.labels == {"a": "1", "b": "2"}


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
            "edit",
            "samplestore",
            store.identifier,
            "--patch-file",
            str(patch_file),
        ],
    )
    assert result.exit_code == 1
    assert "YAML/JSON object" in result.output
