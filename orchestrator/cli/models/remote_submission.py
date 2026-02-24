# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Pydantic models for remote submission preparation.

This module contains specifications and parsing results for CLI flags
that require special handling when preparing commands for remote execution.
"""

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Flag Definition Models
# ============================================================================


class RemoteSubmissionFlagSpec(BaseModel):
    """Specification for a CLI flag requiring special handling during remote submission.

    This defines the characteristics of flags that need processing when preparing
    a CLI command for remote execution (e.g., stripping, file path rewriting).
    This is NOT part of the remote execution mechanism itself, but rather describes
    flags that the submission preparation logic needs to recognize and handle.

    Attributes:
        names: All forms of the flag (e.g., {"-f", "--file"}).
        hasValue: Whether the flag expects a value.
        valueType: Type of value: "string", "file_path", or "key_value".
        stripFromRemote: Whether to remove this flag before remote execution.
        description: Human-readable description for documentation.
    """

    model_config = {"frozen": True}

    names: Annotated[frozenset[str], Field(description="All forms of the flag")]
    hasValue: Annotated[bool, Field(description="Whether the flag expects a value")]
    valueType: Annotated[
        Literal["string", "file_path", "key_value"],
        Field(description="Type of value expected"),
    ] = "string"
    stripFromRemote: Annotated[
        bool,
        Field(description="Whether to remove before remote execution"),
    ] = False
    description: Annotated[str, Field(description="Human-readable description")] = ""

    def matches(self, arg: str) -> bool:
        """Check if arg matches this flag (including --flag=value form)."""
        if arg in self.names:
            return True
        return any(arg.startswith(f"{name}=") for name in self.names)

    def extract_value_from_equals_form(self, arg: str) -> str | None:
        """Extract value from --flag=value form, or None if not equals form."""
        for name in self.names:
            if arg.startswith(f"{name}="):
                return arg[len(name) + 1 :]
        return None

    def get_canonical_name(self) -> str:
        """Return the canonical (longest) flag name for display."""
        return max(self.names, key=len)


# ============================================================================
# Parsing Result Models
# ============================================================================


class RemoteSubmissionFlagMatch(BaseModel):
    """A matched instance of a flag spec in actual CLI arguments.

    This represents a matched occurrence of a flag explicitly defined in this module,
    not just any arbitrary command-line flag.

    Attributes:
        position: Index in original argv where the flag appears.
        name: The actual flag string used (e.g., "-f" or "--file").
        value: The value associated with the flag, if any.
        value_position: Position of value in argv if separate from flag, else None.
        is_equals_form: True if flag was in --flag=value form.
    """

    model_config = {"frozen": True}

    position: Annotated[int, Field(description="Index in original argv")]
    name: Annotated[str, Field(description="The actual flag string used")]
    value: Annotated[
        str | None, Field(description="Value associated with the flag")
    ] = None
    value_position: Annotated[
        int | None,
        Field(description="Position of value in argv if separate"),
    ] = None
    is_equals_form: Annotated[
        bool, Field(description="True if flag was in --flag=value form")
    ]


class ParsedRemoteSubmissionFlags(BaseModel):
    """Result of parsing argv for remote submission flags with position tracking.

    This class contains all CLI flags and arguments, but is primarily used to handle
    flags defined as RemoteSubmissionFlagSpec that require special processing during
    remote submission preparation (stripping, file copying, value rewriting).

    The handled_flags field contains only those flags matched against RemoteSubmissionFlagSpec
    definitions, while passthrough_args contains all other arguments that don't require
    special handling.

    Attributes:
        handled_flags: Flags matched against RemoteSubmissionFlagSpec requiring special processing.
        passthrough_args: Arguments not requiring special processing, passed through unchanged.
    """

    handled_flags: Annotated[
        list[RemoteSubmissionFlagMatch],
        Field(
            description="Flags requiring special processing during submission preparation"
        ),
    ]
    passthrough_args: Annotated[
        list[tuple[int, str]],
        Field(description="Arguments passed through unchanged to remote command"),
    ]

    def reconstruct_argv(
        self,
        exclude_flags: set[str] | None = None,
        value_transformer: (
            Callable[[RemoteSubmissionFlagMatch], str | None] | None
        ) = None,
    ) -> list[str]:
        """Reconstruct argv with optional filtering and value transformation.

        Args:
            exclude_flags: Set of flag names to exclude from output.
            value_transformer: Optional function to transform flag values. If it returns None,
                the original value is used.

        Returns:
            Reconstructed argument list maintaining original order.
        """
        exclude_flags = exclude_flags or set()

        items: list[tuple[int, str]] = []

        for flag in self.handled_flags:
            if flag.name in exclude_flags:
                continue

            # Transform value if transformer provided
            value = flag.value
            if value_transformer is not None:
                transformed = value_transformer(flag)
                if transformed is not None:
                    value = transformed

            # Add flag and value to items
            if flag.is_equals_form:
                items.append((flag.position, f"{flag.name}={value}"))
            else:
                items.append((flag.position, flag.name))
                if value is not None and flag.value_position is not None:
                    items.append((flag.value_position, value))

        # Add non-flag arguments
        items.extend(self.passthrough_args)

        # Sort by position and extract args
        items.sort(key=lambda x: x[0])
        return [arg for _, arg in items]


# ============================================================================
# Flag Registry
# ============================================================================

REMOTE_FLAG = RemoteSubmissionFlagSpec(
    names=frozenset({"--remote"}),
    hasValue=True,
    stripFromRemote=True,
    description="Path to RemoteExecutionContext YAML for remote submission",
)

OVERRIDE_ADO_APP_DIR_FLAG = RemoteSubmissionFlagSpec(
    names=frozenset({"--override-ado-app-dir"}),
    hasValue=True,
    stripFromRemote=True,
    description="Override ado app directory (testing only)",
)

CONTEXT_FLAG = RemoteSubmissionFlagSpec(
    names=frozenset({"-c", "--context"}),
    hasValue=True,
    valueType="file_path",
    stripFromRemote=True,
    description="Project context file path",
)

FILE_FLAG = RemoteSubmissionFlagSpec(
    names=frozenset({"-f", "--file"}),
    hasValue=True,
    valueType="file_path",
    description="Input file path",
)

WITH_FLAG = RemoteSubmissionFlagSpec(
    names=frozenset({"--with"}),
    hasValue=True,
    valueType="key_value",
    description="Resource reference (KEY=VALUE or KEY=path/to/file)",
)

# ============================================================================
# Flag Groups (for common operations)
# ============================================================================

# All flags that should be stripped before remote execution
SUBMISSION_STRIP_FLAGS = [
    REMOTE_FLAG,
    OVERRIDE_ADO_APP_DIR_FLAG,
    CONTEXT_FLAG,
]

# All flags whose values are file paths that need copying
SUBMISSION_FILE_COPY_FLAGS = [FILE_FLAG, WITH_FLAG]

# All flags recognized by the remote submission parser
# (Not all CLI flags - only those relevant for remote submission preparation)
REMOTE_SUBMISSION_FLAGS = [
    REMOTE_FLAG,
    OVERRIDE_ADO_APP_DIR_FLAG,
    CONTEXT_FLAG,
    FILE_FLAG,
    WITH_FLAG,
]

# Made with Bob
