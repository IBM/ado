# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Flag definitions for remote dispatch argument parsing.

This module centralizes all flag metadata, making it easy to add new flags
without modifying parsing logic.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class FlagDefinition(BaseModel):
    """Metadata for a command-line flag.

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
