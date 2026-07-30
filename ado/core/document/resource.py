# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import typing
import uuid
from typing import Annotated, Literal

import pydantic

from ado.core.document.config import DocumentConfiguration
from ado.core.resources import ADOResource, CoreResourceKinds
from ado.utilities.pydantic import Defaultable

if typing.TYPE_CHECKING:
    from rich.console import RenderableType


class DocumentResource(ADOResource):
    """A resource that stores markdown or HTML documents."""

    version: Annotated[str, pydantic.Field()] = "v1"
    kind: Annotated[Literal[CoreResourceKinds.DOCUMENT], pydantic.Field()] = (
        CoreResourceKinds.DOCUMENT
    )
    config: DocumentConfiguration
    identifier: Annotated[
        Defaultable[str],
        pydantic.Field(
            default_factory=lambda: f"document-{str(uuid.uuid4())[:8]}",
        ),
    ]

    def __rich__(self) -> "RenderableType":
        """Render this document resource using rich."""
        from rich.console import Group
        from rich.padding import Padding
        from rich.text import Text

        return Group(
            Text.assemble(("Identifier: ", "bold"), (self.identifier, "bold green")),
            Padding(self.config, (1, 0, 0, 0)),
        )
