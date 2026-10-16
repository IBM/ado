# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import pytest
import typer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.cli.utils.generic.wrappers import get_sql_store
from ado.cli.utils.resources.handlers import (
    apply_patch_to_resources,
    strategic_merge_configuration_metadata,
)
from ado.core import SampleStoreResource
from ado.core.metadata import ConfigurationMetadata
from ado.core.resources import CoreResourceKinds
from ado.metastore.base import ResourcesDoNotExistError
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

    store = random_sample_store_resource_from_file()
    store.config.metadata = ConfigurationMetadata(labels={"original": "value"})
    sql = get_sql_store(project_context=valid_ado_project_context)
    sql.addResource(store)

    patch_file = tmp_path / "m.yaml"
    patch_file.write_text("labels:\n  patched: 'yes'\n")

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
    assert result.exit_code == 0, result.output

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


# Tests for apply_patch_to_resources


def test_apply_patch_to_resources_updates_all(
    valid_ado_project_context: ProjectContext,
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """Valid patch on two existing resources — both are updated in the store."""
    sql = get_sql_store(project_context=valid_ado_project_context)

    store1 = random_sample_store_resource_from_file()
    store1.config.metadata = ConfigurationMetadata(labels={"a": "1"})
    store2 = random_sample_store_resource_from_file()
    store2.config.metadata = ConfigurationMetadata(labels={"b": "2"})
    sql.addResource(store1)
    sql.addResource(store2)

    apply_patch_to_resources(
        resource_ids=[store1.identifier, store2.identifier],
        resource_type=CoreResourceKinds.SAMPLESTORE,
        project_context=valid_ado_project_context,
        metadata_patch="labels: {patched: 'yes'}",
        metadata_path=None,
    )

    updated1 = sql.getResource(
        identifier=store1.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    updated2 = sql.getResource(
        identifier=store2.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert updated1 is not None
    assert updated2 is not None
    assert updated1.config.metadata.labels == {"a": "1", "patched": "yes"}
    assert updated2.config.metadata.labels == {"b": "2", "patched": "yes"}


def test_apply_patch_to_resources_missing_id_raises_and_does_not_modify(
    valid_ado_project_context: ProjectContext,
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """Patch referencing one nonexistent ID — raises and existing resource is not modified."""
    sql = get_sql_store(project_context=valid_ado_project_context)

    store = random_sample_store_resource_from_file()
    store.config.metadata = ConfigurationMetadata(labels={"original": "value"})
    sql.addResource(store)

    with pytest.raises(ResourcesDoNotExistError):
        apply_patch_to_resources(
            resource_ids=[store.identifier, "nonexistent-id-xyz"],
            resource_type=CoreResourceKinds.SAMPLESTORE,
            project_context=valid_ado_project_context,
            metadata_patch="labels: {patched: 'yes'}",
            metadata_path=None,
        )

    unchanged = sql.getResource(
        identifier=store.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert unchanged is not None
    assert unchanged.config.metadata.labels == {"original": "value"}


def test_apply_patch_to_resources_file_patch_updates_resource(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """Patch loaded from a file — resource is updated correctly."""
    sql = get_sql_store(project_context=valid_ado_project_context)

    store = random_sample_store_resource_from_file()
    store.config.metadata = ConfigurationMetadata(labels={"original": "value"})
    sql.addResource(store)

    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text("labels: {from_file: 'yes'}")

    apply_patch_to_resources(
        resource_ids=[store.identifier],
        resource_type=CoreResourceKinds.SAMPLESTORE,
        project_context=valid_ado_project_context,
        metadata_patch=None,
        metadata_path=patch_file,
    )

    updated = sql.getResource(
        identifier=store.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert updated is not None
    assert updated.config.metadata.labels == {"original": "value", "from_file": "yes"}


def test_apply_patch_to_resources_invalid_yaml_raises_exit(
    valid_ado_project_context: ProjectContext,
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """Invalid (non-mapping) YAML patch — raises typer.Exit and no resource is modified."""
    sql = get_sql_store(project_context=valid_ado_project_context)

    store = random_sample_store_resource_from_file()
    store.config.metadata = ConfigurationMetadata(labels={"original": "value"})
    sql.addResource(store)

    with pytest.raises(typer.Exit):
        apply_patch_to_resources(
            resource_ids=[store.identifier],
            resource_type=CoreResourceKinds.SAMPLESTORE,
            project_context=valid_ado_project_context,
            metadata_patch="- not\n- a\n- mapping",
            metadata_path=None,
        )

    unchanged = sql.getResource(
        identifier=store.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert unchanged is not None
    assert unchanged.config.metadata.labels == {"original": "value"}


# CLI-level bulk edit tests


def test_ado_edit_bulk_patch_applies_to_all_ids(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """Two resources, inline patch via CLI — both are updated."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sql = get_sql_store(project_context=valid_ado_project_context)
    store1 = random_sample_store_resource_from_file()
    store1.config.metadata = ConfigurationMetadata(labels={"x": "1"})
    store2 = random_sample_store_resource_from_file()
    store2.config.metadata = ConfigurationMetadata(labels={"y": "2"})
    sql.addResource(store1)
    sql.addResource(store2)

    result = runner.invoke(
        ado,
        [
            "edit",
            "samplestore",
            store1.identifier,
            store2.identifier,
            "-p",
            "labels: {bulk: 'yes'}",
        ],
    )
    assert result.exit_code == 0, result.output

    updated1 = sql.getResource(
        identifier=store1.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    updated2 = sql.getResource(
        identifier=store2.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert updated1 is not None
    assert updated2 is not None
    assert updated1.config.metadata.labels == {"x": "1", "bulk": "yes"}
    assert updated2.config.metadata.labels == {"y": "2", "bulk": "yes"}


def test_ado_edit_bulk_all_or_nothing_on_missing_id(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """One valid resource + one nonexistent ID — valid resource is NOT modified, exit 1."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sql = get_sql_store(project_context=valid_ado_project_context)
    store = random_sample_store_resource_from_file()
    store.config.metadata = ConfigurationMetadata(labels={"original": "value"})
    sql.addResource(store)

    result = runner.invoke(
        ado,
        [
            "edit",
            "samplestore",
            store.identifier,
            "nonexistent-id-xyz",
            "-p",
            "labels: {bulk: 'yes'}",
        ],
    )
    assert result.exit_code == 1
    assert "No changes were made." in result.output

    unchanged = sql.getResource(
        identifier=store.identifier, kind=CoreResourceKinds.SAMPLESTORE
    )
    assert unchanged is not None
    assert unchanged.config.metadata.labels == {"original": "value"}


def test_ado_edit_bulk_requires_patch_when_multiple_ids(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    random_sample_store_resource_from_file: Callable[[], SampleStoreResource],
) -> None:
    """Two IDs, no patch — exit 1 with informative error message."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    sql = get_sql_store(project_context=valid_ado_project_context)
    store1 = random_sample_store_resource_from_file()
    store2 = random_sample_store_resource_from_file()
    sql.addResource(store1)
    sql.addResource(store2)

    result = runner.invoke(
        ado,
        [
            "edit",
            "samplestore",
            store1.identifier,
            store2.identifier,
        ],
    )
    assert result.exit_code == 1
    assert "--patch" in result.output or "patch" in result.output.lower()
