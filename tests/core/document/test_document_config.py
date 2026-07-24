# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib

import pydantic
import pytest
import yaml

from ado.core.document.config import DocumentConfiguration
from ado.utilities.output import pydantic_model_as_yaml


def test_document_configuration_requires_content() -> None:
    """DocumentConfiguration requires content."""
    with pytest.raises(pydantic.ValidationError):
        DocumentConfiguration.model_validate({})


def test_document_configuration_default_content_type() -> None:
    """DocumentConfiguration defaults contentType to markdown."""
    config = DocumentConfiguration(content="Example report")
    assert config.contentType == "markdown"


def test_document_configuration_round_trip_yaml() -> None:
    """DocumentConfiguration round-trips through YAML."""
    config = DocumentConfiguration(
        content="# Report\n\nBody text",
        contentType="markdown",
        relatedResources=["operation-abc-12345678"],
        metadata={"name": "Test report"},
    )
    yaml_text = pydantic_model_as_yaml(config)
    restored = DocumentConfiguration.model_validate(yaml.safe_load(yaml_text))
    assert restored == config


def test_document_configuration_html_round_trip_yaml() -> None:
    """DocumentConfiguration with HTML contentType round-trips through YAML."""
    config = DocumentConfiguration(
        content="<html><body><h1>Report</h1></body></html>",
        contentType="html",
        relatedResources=["operation-abc-12345678"],
        metadata={"name": "HTML report"},
    )
    yaml_text = pydantic_model_as_yaml(config)
    restored = DocumentConfiguration.model_validate(yaml.safe_load(yaml_text))
    assert restored == config
    assert restored.contentType == "html"


def test_document_configuration_from_fixture(
    document_configuration_file: pathlib.Path,
) -> None:
    """Document fixture validates as DocumentConfiguration."""
    config = DocumentConfiguration.model_validate(
        yaml.safe_load(document_configuration_file.read_text())
    )
    assert config.content.startswith("# Operation report")
    assert config.relatedResources == ["operation-test-12345678"]
    assert config.contentType == "markdown"


def test_document_configuration_from_html_fixture(
    document_html_configuration_file: pathlib.Path,
) -> None:
    """HTML document fixture validates as DocumentConfiguration."""
    config = DocumentConfiguration.model_validate(
        yaml.safe_load(document_html_configuration_file.read_text())
    )
    assert config.contentType == "html"
    assert "<h1>Operation report</h1>" in config.content
