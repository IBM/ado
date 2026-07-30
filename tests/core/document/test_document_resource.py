# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.core.document.config import DocumentConfiguration, RelatedResource
from ado.core.document.resource import DocumentResource


def test_document_resource_lifecycle() -> None:
    """DocumentResource round-trips through model_dump and model_validate."""
    config = DocumentConfiguration(
        content="# Report\n\nExample body",
        relatedResources=[
            RelatedResource(id="operation-test-12345678", role="parent"),
        ],
    )
    resource = DocumentResource(config=config)

    dumped = resource.model_dump()
    restored = DocumentResource.model_validate(dumped)
    assert restored == resource


def test_document_resource_rich_print() -> None:
    """DocumentResource __rich__ includes identifier and config fields."""
    from rich.console import Console

    resource = DocumentResource(
        config=DocumentConfiguration(
            content="# Report\n\nBody",
            metadata={"name": "Rich print report"},
        )
    )
    assert hasattr(resource, "__rich__")
    console = Console()
    with console.capture() as capture:
        console.print(resource)
    output = capture.get()
    assert resource.identifier in output
    assert "Rich print report" in output
    assert "Report" in output
