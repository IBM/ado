# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import uuid
from typing import Annotated, Any

import pydantic

from orchestrator.core.document.config import DocumentConfiguration
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.utilities.pydantic import Defaultable


class DocumentResource(ADOResource):
    """A resource that stores markdown documents and optional attachments."""

    @staticmethod
    def _identifier_from_data(data: dict[str, Any]) -> str:
        return f"document-{str(uuid.uuid4())[:8]}"

    version: Annotated[str, pydantic.Field()] = "v1"
    kind: Annotated[CoreResourceKinds, pydantic.Field()] = CoreResourceKinds.DOCUMENT
    config: DocumentConfiguration
    identifier: Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=_identifier_from_data,
        ),
    ]
