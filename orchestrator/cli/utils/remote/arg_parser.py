# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Position-aware argument parsing utilities for remote submission preparation.

This module provides parsing functions for flags explicitly defined in
orchestrator.cli.models.remote_submission. These are used to prepare commands
for remote execution by handling flags that need special processing (stripping,
file copying, value rewriting).
"""

from collections.abc import Callable

from orchestrator.cli.models.remote_submission import (
    ParsedRemoteSubmissionFlags,
    RemoteSubmissionFlagMatch,
    RemoteSubmissionFlagSpec,
)


def parse_argv_with_positions(
    argv: list[str],
    flag_definitions: list[RemoteSubmissionFlagSpec],
) -> ParsedRemoteSubmissionFlags:
    """Parse argv tracking positions of remote submission flags and arguments.

    This is the core parsing function that recognizes only flags explicitly
    defined in remote_submission.py for remote submission preparation.

    Args:
        argv: The argument list to parse.
        flag_definitions: List of flag specs to recognize.

    Returns:
        Parsed remote submission flags with position information.

    Raises:
        ValueError: If a flag expecting a value is at the end of argv without a value.

    Examples:
        >>> from orchestrator.cli.models.remote_submission import FILE_FLAG, CONTEXT_FLAG
        >>> argv = ["-c", "ctx.yaml", "create", "op", "-f", "op.yaml"]
        >>> parsed = parse_argv_with_positions(argv, [FILE_FLAG, CONTEXT_FLAG])
        >>> len(parsed.handled_flags)
        2
        >>> parsed.passthrough_args
        [(2, 'create'), (3, 'op')]
    """
    handled_flags: list[RemoteSubmissionFlagMatch] = []
    passthrough_args: list[tuple[int, str]] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        matched = False

        for flag_def in flag_definitions:
            # Check for --flag=value form
            value_from_equals = flag_def.extract_value_from_equals_form(arg)
            if value_from_equals is not None:
                handled_flags.append(
                    RemoteSubmissionFlagMatch(
                        position=i,
                        name=arg.split("=", 1)[0],
                        value=value_from_equals,
                        value_position=None,
                        is_equals_form=True,
                    )
                )
                matched = True
                i += 1
                break

            # Check for --flag value form
            if arg in flag_def.names:
                if flag_def.hasValue:
                    if i + 1 >= len(argv):
                        raise ValueError(
                            f"Flag {arg} expects a value but is at end of arguments"
                        )
                    value = argv[i + 1]
                    handled_flags.append(
                        RemoteSubmissionFlagMatch(
                            position=i,
                            name=arg,
                            value=value,
                            value_position=i + 1,
                            is_equals_form=False,
                        )
                    )
                    i += 2
                else:
                    handled_flags.append(
                        RemoteSubmissionFlagMatch(
                            position=i,
                            name=arg,
                            value=None,
                            value_position=None,
                            is_equals_form=False,
                        )
                    )
                    i += 1
                matched = True
                break

        if not matched:
            passthrough_args.append((i, arg))
            i += 1

    return ParsedRemoteSubmissionFlags(
        handled_flags=handled_flags, passthrough_args=passthrough_args
    )


# ============================================================================
# High-Level Generic Operations
# ============================================================================


def strip_flags(
    argv: list[str],
    flags_to_strip: list[RemoteSubmissionFlagSpec],
) -> list[str]:
    """Remove specified flags and their values from argv.

    This is a generic function that replaces both remove_execution_context_from_argv
    and _strip_context_flag with a single implementation.

    Args:
        argv: The argument list to process.
        flags_to_strip: List of flag specs to remove.

    Returns:
        New argument list without the specified flags.

    Examples:
        >>> from orchestrator.cli.models.remote_submission import REMOTE_FLAG
        >>> argv = ["-c", "ctx.yaml", "--remote", "exec.yaml", "create", "op"]
        >>> strip_flags(argv, [REMOTE_FLAG])
        ["-c", "ctx.yaml", "create", "op"]
    """
    parsed = parse_argv_with_positions(argv, flags_to_strip)

    # Collect all flag names that appeared
    exclude_flags = {flag.name for flag in parsed.handled_flags}

    return parsed.reconstruct_argv(exclude_flags=exclude_flags)


def rewrite_flag_values(
    argv: list[str],
    flags_to_rewrite: list[RemoteSubmissionFlagSpec],
    value_rewriter: Callable[
        [RemoteSubmissionFlagMatch, RemoteSubmissionFlagSpec], str
    ],
) -> list[str]:
    """Rewrite values of specified flags using a custom function.

    This is a generic function that can handle file path rewriting,
    basename extraction, or any other value transformation.

    Args:
        argv: The argument list to process.
        flags_to_rewrite: List of flag specs whose values should be rewritten.
        value_rewriter: Function that takes (RemoteSubmissionFlagMatch, RemoteSubmissionFlagSpec) and returns
            the new value string.

    Returns:
        New argument list with rewritten values.

    Examples:
        >>> from pathlib import Path
        >>> from orchestrator.cli.models.remote_submission import FILE_FLAG
        >>> def to_basename(flag, flag_def):
        ...     return Path(flag.value).name if flag.value else flag.value
        >>> argv = ["-f", "/path/to/file.yaml"]
        >>> rewrite_flag_values(argv, [FILE_FLAG], to_basename)
        ["-f", "file.yaml"]
    """
    parsed = parse_argv_with_positions(argv, flags_to_rewrite)

    # Create a mapping from flag name to definition for lookup
    flag_def_map = {
        name: flag_def for flag_def in flags_to_rewrite for name in flag_def.names
    }

    def transformer(flag: RemoteSubmissionFlagMatch) -> str | None:
        """Transform a single flag match."""
        if flag.value is None:
            return None

        flag_def = flag_def_map.get(flag.name)
        if flag_def is None:
            return None

        return value_rewriter(flag, flag_def)

    return parsed.reconstruct_argv(value_transformer=transformer)


# Made with Bob
