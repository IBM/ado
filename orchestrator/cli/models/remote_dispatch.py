# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Pydantic models for remote dispatch argument parsing.

This module contains all models and flag definitions for remote dispatch operations,
centralizing flag metadata and parsing structures.
"""

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Flag Definition Models
# ============================================================================


class FlagDefinition(BaseModel):
    """Metadata for a command-line flag used in remote dispatch.

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
        Field(default="string", description="Type of value expected"),
    ] = "string"
    stripFromRemote: Annotated[
        bool,
        Field(default=False, description="Whether to remove before remote execution"),
    ] = False
    description: Annotated[
        str, Field(default="", description="Human-readable description")
    ] = ""

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


class RemoteDispatchFlagOccurrence(BaseModel):
    """A single occurrence of a remote dispatch flag in argv.

    This represents an occurrence of a flag explicitly defined in this module,
    not just any arbitrary command-line flag.

    Attributes:
        position: Index in original argv where the flag appears.
        flag_name: The actual flag string used (e.g., "-f" or "--file").
        value: The value associated with the flag, if any.
        value_position: Position of value in argv if separate from flag, else None.
        is_equals_form: True if flag was in --flag=value form.
    """

    model_config = {"frozen": True}

    position: Annotated[int, Field(description="Index in original argv")]
    flag_name: Annotated[str, Field(description="The actual flag string used")]
    value: Annotated[
        str | None, Field(default=None, description="Value associated with the flag")
    ]
    value_position: Annotated[
        int | None,
        Field(default=None, description="Position of value in argv if separate"),
    ]
    is_equals_form: Annotated[
        bool, Field(description="True if flag was in --flag=value form")
    ]


class ParsedRemoteDispatchFlags(BaseModel):
    """Result of parsing argv for remote dispatch flags with position tracking.

    This contains only the flags explicitly defined in this module,
    not all command-line arguments.

    Attributes:
        flag_occurrences: All recognized remote dispatch flag occurrences with their positions.
        other_args: All non-flag arguments with their positions.
    """

    flag_occurrences: Annotated[
        list[RemoteDispatchFlagOccurrence],
        Field(description="All recognized flag occurrences"),
    ]
    other_args: Annotated[
        list[tuple[int, str]],
        Field(description="All non-flag arguments with positions"),
    ]

    def reconstruct_argv(
        self,
        exclude_flags: set[str] | None = None,
        value_transformer: (
            Callable[[RemoteDispatchFlagOccurrence], str | None] | None
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

        for occ in self.flag_occurrences:
            if occ.flag_name in exclude_flags:
                continue

            # Transform value if transformer provided
            value = occ.value
            if value_transformer is not None:
                transformed = value_transformer(occ)
                if transformed is not None:
                    value = transformed

            # Add flag and value to items
            if occ.is_equals_form:
                items.append((occ.position, f"{occ.flag_name}={value}"))
            else:
                items.append((occ.position, occ.flag_name))
                if value is not None and occ.value_position is not None:
                    items.append((occ.value_position, value))

        # Add non-flag arguments
        items.extend(self.other_args)

        # Sort by position and extract args
        items.sort(key=lambda x: x[0])
        return [arg for _, arg in items]


# ============================================================================
# Flag Registry
# ============================================================================

EXECUTION_CONTEXT = FlagDefinition(
    names=frozenset({"--execution-context"}),
    hasValue=True,
    stripFromRemote=True,
    description="Path to ExecutionContext YAML for remote dispatch",
)

OVERRIDE_ADO_APP_DIR = FlagDefinition(
    names=frozenset({"--override-ado-app-dir"}),
    hasValue=True,
    stripFromRemote=True,
    description="Override ado app directory (testing only)",
)

CONTEXT = FlagDefinition(
    names=frozenset({"-c", "--context"}),
    hasValue=True,
    valueType="file_path",
    description="Project context file path",
)

FILE = FlagDefinition(
    names=frozenset({"-f", "--file"}),
    hasValue=True,
    valueType="file_path",
    description="Input file path",
)

WITH = FlagDefinition(
    names=frozenset({"--with"}),
    hasValue=True,
    valueType="key_value",
    description="Resource reference (KEY=VALUE or KEY=path/to/file)",
)

# ============================================================================
# Flag Groups (for common operations)
# ============================================================================

# All flags that should be stripped before remote execution
REMOTE_STRIP_FLAGS = [EXECUTION_CONTEXT, OVERRIDE_ADO_APP_DIR]

# All flags whose values are file paths that need copying
FILE_COPY_FLAGS = [FILE, WITH]

# Context flags (for special handling)
CONTEXT_FLAGS = [CONTEXT]

# All flags recognized by the remote dispatch parser
# (Not all CLI flags - only those relevant for remote submission)
REMOTE_DISPATCH_FLAGS = [
    EXECUTION_CONTEXT,
    OVERRIDE_ADO_APP_DIR,
    CONTEXT,
    FILE,
    WITH,
]

# Made with Bob
