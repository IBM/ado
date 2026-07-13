# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.document.config import DocumentConfiguration
from ado.core.document.resource import DocumentResource
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore


def test_delete_document(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Delete removes a document resource."""
    config = DocumentConfiguration(content="Example report")
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
            "delete",
            "document",
            resource.identifier,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Success!" in result.output.strip()


def test_delete_nonexistent_document(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Delete reports an error for a missing document."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "delete",
            "document",
            "does-not-exist",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "Failed to delete does-not-exist" in result.output
    assert "Resource does not exist" in result.output
