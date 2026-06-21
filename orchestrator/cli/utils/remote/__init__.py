# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Remote dispatch utilities for executing ado commands on Ray clusters."""

from orchestrator.cli.models.remote_submission import (
    CONTEXT_FLAG,
    FILE_FLAG,
    OVERRIDE_ADO_APP_DIR_FLAG,
    REMOTE_FLAG,
    REMOTE_SUBMISSION_FLAGS,
    SUBMISSION_FILE_COPY_FLAGS,
    SUBMISSION_STRIP_FLAGS,
    WITH_FLAG,
    ParsedRemoteSubmissionFlags,
    RemoteSubmissionFlagMatch,
    RemoteSubmissionFlagSpec,
)
from orchestrator.cli.utils.remote.arg_parser import (
    parse_argv_with_positions,
    rewrite_flag_values,
    strip_flags,
)
from orchestrator.cli.utils.remote.dispatch import dispatch

__all__ = [
    "CONTEXT_FLAG",
    "FILE_FLAG",
    "OVERRIDE_ADO_APP_DIR_FLAG",
    "REMOTE_FLAG",
    "REMOTE_SUBMISSION_FLAGS",
    "SUBMISSION_FILE_COPY_FLAGS",
    "SUBMISSION_STRIP_FLAGS",
    "WITH_FLAG",
    "ParsedRemoteSubmissionFlags",
    "RemoteSubmissionFlagMatch",
    "RemoteSubmissionFlagSpec",
    "dispatch",
    "parse_argv_with_positions",
    "rewrite_flag_values",
    "strip_flags",
]

# Made with Bob
