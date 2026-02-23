# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Remote dispatch utilities for executing ado commands on Ray clusters."""

from orchestrator.cli.utils.remote.arg_parser import (
    FlagOccurrence,
    ParsedArguments,
    filter_and_rewrite,
    parse_argv_with_positions,
    rewrite_flag_values,
    strip_flags,
)
from orchestrator.cli.utils.remote.dispatch import dispatch
from orchestrator.cli.utils.remote.flag_definitions import (
    CONTEXT,
    CONTEXT_FLAGS,
    EXECUTION_CONTEXT,
    FILE,
    FILE_COPY_FLAGS,
    OVERRIDE_ADO_APP_DIR,
    REMOTE_DISPATCH_FLAGS,
    REMOTE_STRIP_FLAGS,
    WITH,
    FlagDefinition,
)

__all__ = [
    "CONTEXT",
    "CONTEXT_FLAGS",
    "EXECUTION_CONTEXT",
    "FILE",
    "FILE_COPY_FLAGS",
    "OVERRIDE_ADO_APP_DIR",
    "REMOTE_DISPATCH_FLAGS",
    "REMOTE_STRIP_FLAGS",
    "WITH",
    "FlagDefinition",
    "FlagOccurrence",
    "ParsedArguments",
    "dispatch",
    "filter_and_rewrite",
    "parse_argv_with_positions",
    "rewrite_flag_values",
    "strip_flags",
]

# Made with Bob
