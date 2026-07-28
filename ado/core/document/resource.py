# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import uuid
from typing import Annotated

import pydantic

from ado.core.document.config import DocumentConfiguration
from ado.core.resources import ADOResource, CoreResourceKinds
from ado.utilities.pydantic import Defaultable


class DocumentResource(ADOResource):
    """A resource that stores markdown or HTML documents."""

    version: Annotated[str, pydantic.Field()] = "v1"
    kind: Annotated[CoreResourceKinds, pydantic.Field()] = CoreResourceKinds.DOCUMENT
    config: DocumentConfiguration
    identifier: Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=lambda: f"document-{str(uuid.uuid4())[:8]}",
        ),
    ]
