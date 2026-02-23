# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Remote dispatch utilities for executing ado commands on Ray clusters."""

from orchestrator.cli.models.remote_dispatch import (
    CONTEXT_SPEC,
    EXECUTION_CONTEXT_SPEC,
    FILE_SPEC,
    OVERRIDE_ADO_APP_DIR_SPEC,
    REMOTE_SUBMISSION_FLAGS,
    SUBMISSION_CONTEXT_FLAGS,
    SUBMISSION_FILE_COPY_FLAGS,
    SUBMISSION_STRIP_FLAGS,
    WITH_SPEC,
    ParsedRemoteSubmissionFlags,
    RemoteSubmissionFlagMatch,
    RemoteSubmissionFlagSpec,
)
from orchestrator.cli.utils.remote.arg_parser import (
    filter_and_rewrite,
    parse_argv_with_positions,
    rewrite_flag_values,
    strip_flags,
)
from orchestrator.cli.utils.remote.dispatch import dispatch

__all__ = [
    "CONTEXT_SPEC",
    "EXECUTION_CONTEXT_SPEC",
    "FILE_SPEC",
    "OVERRIDE_ADO_APP_DIR_SPEC",
    "REMOTE_SUBMISSION_FLAGS",
    "SUBMISSION_CONTEXT_FLAGS",
    "SUBMISSION_FILE_COPY_FLAGS",
    "SUBMISSION_STRIP_FLAGS",
    "WITH_SPEC",
    "ParsedRemoteSubmissionFlags",
    "RemoteSubmissionFlagMatch",
    "RemoteSubmissionFlagSpec",
    "dispatch",
    "filter_and_rewrite",
    "parse_argv_with_positions",
    "rewrite_flag_values",
    "strip_flags",
]

# Made with Bob
