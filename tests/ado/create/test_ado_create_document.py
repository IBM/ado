# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import yaml
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.document.config import DocumentConfiguration
from ado.core.document.resource import DocumentResource
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore


def test_create_document_dry_run_success(tmp_path: pathlib.Path) -> None:
    """Dry-run validates a document configuration without persisting it."""
    document_file = "tests/fixtures/document.yaml"
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "document",
            "-f",
            document_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "The configuration passed is valid!" in result.output


def test_create_document_dry_run_failure(tmp_path: pathlib.Path) -> None:
    """Dry-run rejects an invalid document configuration."""
    document_file = pathlib.Path("tests/fixtures/document.yaml")
    invalid_document_file = tmp_path / "invalid.yaml"
    document_configuration = yaml.safe_load(document_file.read_text())
    del document_configuration["content"]
    invalid_document_file.write_text(yaml.safe_dump(document_configuration))

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "document",
            "-f",
            invalid_document_file,
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "The document provided was not valid:" in result.output


def test_create_document(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Create persists a document resource."""
    document_file = "tests/fixtures/document.yaml"
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "create",
            "document",
            "-f",
            document_file,
        ],
    )

    assert result.exit_code == 0
    assert result.output.startswith("Success! Created document with identifier")


def test_get_document_config_round_trip(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Get document -o config returns the stored configuration."""
    config = DocumentConfiguration.model_validate(
        yaml.safe_load(pathlib.Path("tests/fixtures/document.yaml").read_text())
    )
    resource = DocumentResource(config=config)
    sql_store.addResource(resource)

    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "get",
            "document",
            resource.identifier,
            "-o",
            "config",
        ],
    )

    assert result.exit_code == 0
    restored = DocumentConfiguration.model_validate(yaml.safe_load(result.output))
    assert restored.content == config.content
    assert restored.relatedResources == config.relatedResources
