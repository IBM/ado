# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib

import pytest


@pytest.fixture
def document_configuration_file() -> pathlib.Path:
    """Return path to a valid document configuration fixture."""
    return pathlib.Path("tests/fixtures/document.yaml")


@pytest.fixture
def document_with_attachment_configuration_file() -> pathlib.Path:
    """Return path to a document configuration fixture with attachments."""
    return pathlib.Path("tests/fixtures/document_with_attachment.yaml")
