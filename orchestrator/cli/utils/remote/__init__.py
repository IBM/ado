# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Remote dispatch utilities for executing ado commands on Ray clusters."""

from orchestrator.cli.models.remote_dispatch import (
    CONTEXT,
    CONTEXT_FLAGS,
    EXECUTION_CONTEXT,
    FILE,
    FILE_COPY_FLAGS,
    OVERRIDE_ADO_APP_DIR,
    REMOTE_DISPATCH_FLAGS,
    REMOTE_STRIP_FLAGS,
    WITH,
    ParsedRemoteDispatchFlags,
    RemoteDispatchFlagDefinition,
    RemoteDispatchFlagOccurrence,
)
from orchestrator.cli.utils.remote.arg_parser import (
    filter_and_rewrite,
    parse_argv_with_positions,
    rewrite_flag_values,
    strip_flags,
)
from orchestrator.cli.utils.remote.dispatch import dispatch

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
    "ParsedRemoteDispatchFlags",
    "RemoteDispatchFlagDefinition",
    "RemoteDispatchFlagOccurrence",
    "dispatch",
    "filter_and_rewrite",
    "parse_argv_with_positions",
    "rewrite_flag_values",
    "strip_flags",
]

# Made with Bob
