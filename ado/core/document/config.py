# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated, Literal

import pydantic
from pydantic import ConfigDict

from ado.core.metadata import ConfigurationMetadata

if typing.TYPE_CHECKING:
    from rich.console import RenderableType


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


class StudyDocumentMetadata(ConfigurationMetadata):
    """Metadata for a document that tracks a study."""

    todo: Annotated[
        list[str] | None,
        pydantic.Field(description="Optional list of outstanding todo items"),
    ] = None


def _document_metadata_discriminator(value: typing.Any) -> str:  # noqa: ANN401
    """Select the metadata model: study metadata if and only if todo is present.

    Args:
        value: Raw metadata (mapping or model instance) being validated.

    Returns:
        The tag of the metadata model to validate against.
    """
    if isinstance(value, dict):
        return "study" if "todo" in value else "default"
    return "study" if isinstance(value, StudyDocumentMetadata) else "default"


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
        Annotated[ConfigurationMetadata, pydantic.Tag("default")]
        | Annotated[StudyDocumentMetadata, pydantic.Tag("study")],
        pydantic.Discriminator(_document_metadata_discriminator),
        pydantic.Field(
            description="Metadata about the document including optional name, "
            "description, labels for filtering, and any additional custom fields. "
            "Study documents additionally support a todo list"
        ),
    ] = ConfigurationMetadata()

    def __rich__(self) -> "RenderableType":
        """Render metadata, related resources, and content for rich."""
        from rich.console import Group
        from rich.markdown import Markdown
        from rich.text import Text

        from ado.utilities.rich import ConsoleWidthRenderable, RstrippedRenderable

        parts: list[typing.Any] = []
        if self.metadata.name:
            parts.append(Text.assemble(("Name: ", "bold"), (self.metadata.name,)))
        if self.metadata.description:
            parts.append(
                Text.assemble(
                    ("Description: ", "bold"), (self.metadata.description.rstrip(),)
                )
            )
        if self.relatedResources:
            related_summary = ", ".join(
                f"{related.id} ({related.role})" for related in self.relatedResources
            )
            parts.append(
                Text.assemble(("Related resources: ", "bold"), (related_summary,))
            )
        if isinstance(self.metadata, StudyDocumentMetadata) and self.metadata.todo:
            todo_lines = "\n".join(f"  - {item}" for item in self.metadata.todo)
            parts.append(Text.assemble(("Todo:", "bold"), (f"\n{todo_lines}", "")))
        if parts:
            parts.append("")

        if self.contentType == "html":
            parts.append(self.content)
        else:
            parts.append(Markdown(self.content))

        return RstrippedRenderable(ConsoleWidthRenderable(Group(*parts)))
