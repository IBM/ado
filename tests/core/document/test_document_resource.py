# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import re

import pydantic
import pytest

from ado.core.document.config import DocumentConfiguration
from ado.core.document.resource import DocumentResource
from ado.core.resources import CoreResourceKinds


def test_document_resource_lifecycle() -> None:
    """DocumentResource round-trips through model_dump and model_validate."""
    config = DocumentConfiguration(
        content="# Report\n\nExample body",
        relatedResources=["operation-test-12345678"],
        attachments={"chart.png": "cGV4"},
    )
    resource = DocumentResource(config=config)

    dumped = resource.model_dump()
    restored = DocumentResource.model_validate(dumped)
    assert restored == resource


def test_document_resource_identifier_auto_generation() -> None:
    """DocumentResource auto-generates a document identifier."""
    config = DocumentConfiguration(content="Example report")
    resource = DocumentResource(config=config)

    assert re.fullmatch(r"document-[0-9a-f]{8}", resource.identifier)


def test_document_resource_kind_is_pinned() -> None:
    """DocumentResource kind is pinned to document."""
    config = DocumentConfiguration(content="Example report")
    resource = DocumentResource(config=config)

    assert resource.kind == CoreResourceKinds.DOCUMENT


def test_document_resource_requires_content() -> None:
    """DocumentResource requires config content."""
    with pytest.raises(pydantic.ValidationError, match="content"):
        DocumentResource.model_validate(
            {
                "kind": CoreResourceKinds.DOCUMENT,
                "config": {},
            }
        )
