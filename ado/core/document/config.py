# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal

import pydantic
from pydantic import ConfigDict

from ado.core.metadata import ConfigurationMetadata


class RelatedResource(pydantic.BaseModel):
    """A resource linked to a document as a parent or child."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, pydantic.Field(description="Related resource identifier")]
    role: Annotated[
        Literal["parent", "child"],
        pydantic.Field(
            description=(
                "parent: report is about this resource; "
                "child: resource created in response to this document"
            ),
        ),
    ]


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
        list[RelatedResource],
        pydantic.Field(
            default_factory=list,
            description=(
                "Related ado resources with role parent (report is about them) "
                "or child (created in response to this document)"
            ),
        ),
    ]
    metadata: Annotated[
        ConfigurationMetadata,
        pydantic.Field(
            description="Metadata about the document including optional name, "
            "description, labels for filtering, and any additional custom fields"
        ),
    ] = ConfigurationMetadata()
