# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Validate and normalize the structured output of the ado-review-pr skill.

The field names are snake_case and match the names used by the GitHub API
(``commit_id``, ``start_line``, ``in_reply_to``), so the JSON maps directly to
the API calls that post it.

Usage:
    uv run python pr_review_model.py [FILE]
        Read the review JSON from FILE (default: stdin). On success print the
        normalized compact JSON and exit 0; when the review is not valid print
        every problem to stderr and exit 1; when FILE does not exist, is a
        directory, or is not readable exit 2.
    uv run python pr_review_model.py --schema
        Print the JSON schema of the review.
"""

import json
import pathlib
import sys
from typing import Annotated, Literal

import pydantic
import typer

from ado.utilities.pydantic import Defaultable

app = typer.Typer(add_completion=False)


class ReviewComment(pydantic.BaseModel):
    """A new comment on a line, or a range of lines, of the PR diff."""

    model_config = pydantic.ConfigDict(extra="forbid")

    path: Annotated[
        str,
        pydantic.Field(
            min_length=1, description="File path relative to the repository root"
        ),
    ]
    line: Annotated[
        int,
        pydantic.Field(
            gt=0,
            description="Last line of the commented range in the new version of the file",
        ),
    ]
    start_line: Annotated[
        int | None,
        pydantic.Field(
            gt=0,
            description="First line of the range, only if the comment spans several lines",
        ),
    ] = None
    body: Annotated[
        str,
        pydantic.Field(
            min_length=1,
            description="Explanation of the issue, in markdown, not hard-wrapped",
        ),
    ]
    suggestion: Annotated[
        Defaultable[str],
        pydantic.Field(
            description=(
                "Exact replacement text for the commented line(s), only when a "
                "concrete code change is proposed. Empty otherwise"
            ),
        ),
    ] = ""

    @pydantic.field_validator("path")
    @classmethod
    def strip_leading_dot_slash(cls, value: str) -> str:
        """Remove leading ``./`` segments from a path.

        Args:
            value: The path.

        Returns:
            The path without leading ``./`` segments.

        Raises:
            ValueError: If no path is left after removing the segments.
        """
        while value.startswith("./"):
            value = value[2:]
        if not value:
            raise ValueError("path must not be empty")
        return value

    @pydantic.model_validator(mode="after")
    def check_range(self) -> "ReviewComment":
        """Ensure ``start_line`` does not come after ``line``.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If ``start_line`` is greater than ``line``.
        """
        if self.start_line is not None and self.start_line > self.line:
            raise ValueError("start_line must not be greater than line")
        return self


class CommentReply(pydantic.BaseModel):
    """A reply to a comment that already exists on the PR."""

    model_config = pydantic.ConfigDict(extra="forbid")

    in_reply_to: Annotated[
        int,
        pydantic.Field(
            gt=0,
            description=(
                "Id of the existing review comment to reply to, as returned "
                "by pulls/{number}/comments"
            ),
        ),
    ]
    body: Annotated[
        str,
        pydantic.Field(
            min_length=1,
            description="The reply, in markdown, not hard-wrapped",
        ),
    ]
    thread_state: Annotated[
        Literal["unchanged", "resolved", "unresolved"],
        pydantic.Field(
            description=(
                "State to leave the thread in after posting the reply: "
                "'resolved' when the finding is addressed, 'unresolved' to "
                "reopen a resolved thread whose finding still holds, "
                "'unchanged' otherwise"
            ),
        ),
    ] = "unchanged"


class Review(pydantic.BaseModel):
    """A review: a summary, new line comments, and replies to existing comments."""

    model_config = pydantic.ConfigDict(extra="forbid")

    commit_id: Annotated[
        str,
        pydantic.Field(
            pattern=r"^[0-9a-f]{40}$",
            description=(
                "The full sha of the PR head commit that was reviewed; "
                "the line numbers of the comments refer to it"
            ),
        ),
    ]
    summary: Annotated[
        str,
        pydantic.Field(
            min_length=1,
            description="Overall summary of the review, in markdown, not hard-wrapped",
        ),
    ]
    comments: Annotated[
        list[ReviewComment],
        pydantic.Field(
            default_factory=list,
            description=(
                "Only new comments that refer to a specific file and line. "
                "Anything else belongs in the summary"
            ),
        ),
    ]
    replies: Annotated[
        list[CommentReply],
        pydantic.Field(
            default_factory=list,
            description=(
                "Replies to existing comments, for example to say that a "
                "finding was addressed or to answer a question"
            ),
        ),
    ]


def describe_errors(error: pydantic.ValidationError) -> list[str]:
    """List the problems of a validation error, one per line.

    Args:
        error: The validation error.

    Returns:
        One ``location: message (got value)`` string per problem.
    """
    problems = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        received = repr(item.get("input"))
        if len(received) > 80:
            received = received[:77] + "..."
        problems.append(f"{location}: {item['msg']} (got {received})")
    return problems


@app.command(help="Validate a review and print it in normalized form.")
def main(
    file: Annotated[
        pathlib.Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Review JSON file (default: stdin)",
        ),
    ] = None,
    schema: Annotated[
        bool, typer.Option("--schema", help="Print the JSON schema and exit")
    ] = False,
) -> None:
    """Validate a review and print it in normalized form.

    Args:
        file: File to read the review from; stdin when omitted.
        schema: Print the JSON schema of the review instead of validating.

    Raises:
        typer.Exit: With status 1 when the review is not valid.
    """
    if schema:
        typer.echo(json.dumps(Review.model_json_schema(), indent=2))
        return

    content = file.read_bytes() if file else sys.stdin.buffer.read()

    try:
        review = Review.model_validate_json(content)
    except pydantic.ValidationError as error:
        typer.echo("Invalid review:", err=True)
        for problem in describe_errors(error):
            typer.echo(f"  - {problem}", err=True)
        raise typer.Exit(1) from error

    typer.echo(review.model_dump_json())


if __name__ == "__main__":
    app()
