# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import yaml
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core import DataContainerResource
from ado.core.datacontainer.resource import DataContainer
from ado.core.document.config import DocumentConfiguration, RelatedResource
from ado.core.document.resource import DocumentResource
from ado.core.resources import CoreResourceKinds
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
    """Create persists a document resource with no related resources."""
    document_file = tmp_path / "document.yaml"
    document_file.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "Create test report"},
                "content": "# Report\n\nBody.\n",
            }
        )
    )
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
            str(document_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("Success! Created document with identifier")


def test_create_document_with_parent_relationship(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Create writes a parent→document edge for role parent."""
    parent = DataContainerResource(config=DataContainer(data={"key": "parent"}))
    parent.identifier = "dc-document-parent"
    sql_store.addResource(parent)

    document_file = tmp_path / "document_parent.yaml"
    document_file.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "Parent link report"},
                "content": "# Report\n",
                "relatedResources": [
                    {"id": parent.identifier, "role": "parent"},
                ],
            }
        )
    )

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
            str(document_file),
        ],
    )
    assert result.exit_code == 0, result.output

    document_id = result.output.strip().rsplit(maxsplit=1)[-1]
    parents = sql_store.getRelatedSubjectResourceIdentifiers(identifier=document_id)
    assert parent.identifier in set(parents["IDENTIFIER"])


def test_create_document_with_child_relationship(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Create writes a document→child edge for role child."""
    child = DataContainerResource(config=DataContainer(data={"key": "child"}))
    child.identifier = "dc-document-child"
    sql_store.addResource(child)

    document_file = tmp_path / "document_child.yaml"
    document_file.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "Child link report"},
                "content": "# Report\n",
                "relatedResources": [
                    {"id": child.identifier, "role": "child"},
                ],
            }
        )
    )

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
            str(document_file),
        ],
    )
    assert result.exit_code == 0, result.output

    document_id = result.output.strip().rsplit(maxsplit=1)[-1]
    children = sql_store.getRelatedObjectResourceIdentifiers(identifier=document_id)
    assert child.identifier in set(children["IDENTIFIER"])


def test_create_document_unknown_related_resource(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    sql_store: SQLStore,
) -> None:
    """Create fails when a related resource id does not exist."""
    document_file = tmp_path / "document_missing.yaml"
    document_file.write_text(
        yaml.safe_dump(
            {
                "metadata": {"name": "Missing related"},
                "content": "# Report\n",
                "relatedResources": [
                    {"id": "i-do-not-exist", "role": "parent"},
                ],
            }
        )
    )

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
            str(document_file),
        ],
    )
    assert result.exit_code == 1
    assert "Unknown related resource identifier" in result.output
    documents = sql_store.getResourceIdentifiersOfKind(kind=CoreResourceKinds.DOCUMENT)
    assert documents.empty


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
    assert restored.relatedResources == [
        RelatedResource(id="operation-test-12345678", role="parent"),
    ]
