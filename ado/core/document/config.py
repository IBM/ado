# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal

import pydantic
from pydantic import ConfigDict

from ado.core.metadata import ConfigurationMetadata


class DocumentConfiguration(pydantic.BaseModel):
    """Configuration for a document resource."""

    model_config = ConfigDict(extra="forbid")

    content: Annotated[
        str,
        pydantic.Field(description="Body of the document (markdown or HTML)"),
    ]
    contentType: Annotated[
        Literal["markdown", "html"],
        pydantic.Field(description="Format of the content field"),
    ] = "markdown"
    relatedResources: Annotated[
        list[str],
        pydantic.Field(
            default_factory=list,
            description="Identifiers of related ado resources, if any",
        ),
    ]
    metadata: Annotated[
        ConfigurationMetadata,
        pydantic.Field(
            description="Metadata about the document including optional name, "
            "description, labels for filtering, and any additional custom fields"
        ),
    ] = ConfigurationMetadata()
