# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.core.metadata import ConfigurationMetadata


class DocumentConfiguration(pydantic.BaseModel):
    """Configuration for a document resource."""

    model_config = ConfigDict(extra="forbid")

    content: Annotated[
        str,
        pydantic.Field(description="Markdown body of the document"),
    ]
    relatedResources: Annotated[
        list[str],
        pydantic.Field(
            default_factory=list,
            description="Identifiers of related ado resources, if any",
        ),
    ]
    attachments: Annotated[
        dict[str, str],
        pydantic.Field(
            default_factory=dict,
            description=(
                "Mapping of filename to base64-encoded content referenced "
                "from the markdown content"
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
