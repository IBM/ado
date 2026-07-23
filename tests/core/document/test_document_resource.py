# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from ado.core.document.config import DocumentConfiguration
from ado.core.document.resource import DocumentResource


def test_document_resource_lifecycle() -> None:
    """DocumentResource round-trips through model_dump and model_validate."""
    config = DocumentConfiguration(
        content="# Report\n\nExample body",
        relatedResources=["operation-test-12345678"],
    )
    resource = DocumentResource(config=config)

    dumped = resource.model_dump()
    restored = DocumentResource.model_validate(dumped)
    assert restored == resource
