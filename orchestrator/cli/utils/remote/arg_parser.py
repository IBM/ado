# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Position-aware argument parsing utilities for remote dispatch.

This module provides parsing functions for flags explicitly defined in
orchestrator.cli.models.remote_dispatch. These are used for remote dispatch
operations to handle flags that need special processing (stripping, file copying,
value rewriting).
"""

from collections.abc import Callable

from orchestrator.cli.models.remote_dispatch import (
    ParsedRemoteDispatchFlags,
    RemoteDispatchFlagDefinition,
    RemoteDispatchFlagOccurrence,
)


def parse_argv_with_positions(
    argv: list[str],
    flag_definitions: list[RemoteDispatchFlagDefinition],
) -> ParsedRemoteDispatchFlags:
    """Parse argv tracking positions of remote dispatch flags and arguments.

    This is the core parsing function that recognizes only flags explicitly
    defined in flag_definitions.py for remote dispatch operations.

    Args:
        argv: The argument list to parse.
        flag_definitions: List of flag definitions to recognize.

    Returns:
        Parsed remote dispatch flags with position information.

    Raises:
        ValueError: If a flag expecting a value is at the end of argv without a value.

    Examples:
        >>> from orchestrator.cli.utils.remote.flag_definitions import FILE, CONTEXT
        >>> argv = ["-c", "ctx.yaml", "create", "op", "-f", "op.yaml"]
        >>> parsed = parse_argv_with_positions(argv, [FILE, CONTEXT])
        >>> len(parsed.flag_occurrences)
        2
        >>> parsed.other_args
        [(2, 'create'), (3, 'op')]
    """
    flag_occurrences: list[RemoteDispatchFlagOccurrence] = []
    other_args: list[tuple[int, str]] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        matched = False

        for flag_def in flag_definitions:
            # Check for --flag=value form
            value_from_equals = flag_def.extract_value_from_equals_form(arg)
            if value_from_equals is not None:
                flag_occurrences.append(
                    RemoteDispatchFlagOccurrence(
                        position=i,
                        flag_name=arg.split("=", 1)[0],
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
                    flag_occurrences.append(
                        RemoteDispatchFlagOccurrence(
                            position=i,
                            flag_name=arg,
                            value=value,
                            value_position=i + 1,
                            is_equals_form=False,
                        )
                    )
                    i += 2
                else:
                    flag_occurrences.append(
                        RemoteDispatchFlagOccurrence(
                            position=i,
                            flag_name=arg,
                            value=None,
                            value_position=None,
                            is_equals_form=False,
                        )
                    )
                    i += 1
                matched = True
                break

        if not matched:
            other_args.append((i, arg))
            i += 1

    return ParsedRemoteDispatchFlags(
        flag_occurrences=flag_occurrences, other_args=other_args
    )


# ============================================================================
# High-Level Generic Operations
# ============================================================================


def strip_flags(
    argv: list[str],
    flags_to_strip: list[RemoteDispatchFlagDefinition],
) -> list[str]:
    """Remove specified flags and their values from argv.

    This is a generic function that replaces both remove_execution_context_from_argv
    and _strip_context_flag with a single implementation.

    Args:
        argv: The argument list to process.
        flags_to_strip: List of flag definitions to remove.

    Returns:
        New argument list without the specified flags.

    Examples:
        >>> from orchestrator.cli.utils.remote.flag_definitions import EXECUTION_CONTEXT
        >>> argv = ["-c", "ctx.yaml", "--execution-context", "exec.yaml", "create", "op"]
        >>> strip_flags(argv, [EXECUTION_CONTEXT])
        ["-c", "ctx.yaml", "create", "op"]
    """
    parsed = parse_argv_with_positions(argv, flags_to_strip)

    # Collect all flag names that appeared
    exclude_flags = {occ.flag_name for occ in parsed.flag_occurrences}

    return parsed.reconstruct_argv(exclude_flags=exclude_flags)


def rewrite_flag_values(
    argv: list[str],
    flags_to_rewrite: list[RemoteDispatchFlagDefinition],
    value_rewriter: Callable[
        [RemoteDispatchFlagOccurrence, RemoteDispatchFlagDefinition], str
    ],
) -> list[str]:
    """Rewrite values of specified flags using a custom function.

    This is a generic function that can handle file path rewriting,
    basename extraction, or any other value transformation.

    Args:
        argv: The argument list to process.
        flags_to_rewrite: List of flag definitions whose values should be rewritten.
        value_rewriter: Function that takes (RemoteDispatchFlagOccurrence, RemoteDispatchFlagDefinition) and returns
            the new value string.

    Returns:
        New argument list with rewritten values.

    Examples:
        >>> from pathlib import Path
        >>> from orchestrator.cli.utils.remote.flag_definitions import FILE
        >>> def to_basename(occ, flag_def):
        ...     return Path(occ.value).name if occ.value else occ.value
        >>> argv = ["-f", "/path/to/file.yaml"]
        >>> rewrite_flag_values(argv, [FILE], to_basename)
        ["-f", "file.yaml"]
    """
    parsed = parse_argv_with_positions(argv, flags_to_rewrite)

    # Create a mapping from flag name to definition for lookup
    flag_def_map = {
        name: flag_def for flag_def in flags_to_rewrite for name in flag_def.names
    }

    def transformer(occ: RemoteDispatchFlagOccurrence) -> str | None:
        """Transform a single flag occurrence."""
        if occ.value is None:
            return None

        flag_def = flag_def_map.get(occ.flag_name)
        if flag_def is None:
            return None

        return value_rewriter(occ, flag_def)

    return parsed.reconstruct_argv(value_transformer=transformer)


def filter_and_rewrite(
    argv: list[str],
    flags_to_strip: list[RemoteDispatchFlagDefinition],
    flags_to_rewrite: list[RemoteDispatchFlagDefinition],
    value_rewriter: Callable[
        [RemoteDispatchFlagOccurrence, RemoteDispatchFlagDefinition], str
    ],
) -> list[str]:
    """Combined operation: strip some flags and rewrite others.

    This is more efficient than calling strip_flags() and rewrite_flag_values()
    separately because it only parses argv once.

    Args:
        argv: The argument list to process.
        flags_to_strip: Flags to remove completely.
        flags_to_rewrite: Flags whose values should be transformed.
        value_rewriter: Function to transform flag values.

    Returns:
        Processed argument list.
    """
    all_flags = flags_to_strip + flags_to_rewrite
    parsed = parse_argv_with_positions(argv, all_flags)

    # Collect flags to exclude
    strip_flag_names = {name for flag in flags_to_strip for name in flag.names}
    exclude_flags = {
        occ.flag_name
        for occ in parsed.flag_occurrences
        if occ.flag_name in strip_flag_names
    }

    # Create rewrite mapping
    rewrite_flag_map = {
        name: flag_def for flag_def in flags_to_rewrite for name in flag_def.names
    }

    def transformer(occ: RemoteDispatchFlagOccurrence) -> str | None:
        if occ.value is None or occ.flag_name in exclude_flags:
            return None

        flag_def = rewrite_flag_map.get(occ.flag_name)
        if flag_def is None:
            return None

        return value_rewriter(occ, flag_def)

    return parsed.reconstruct_argv(
        exclude_flags=exclude_flags,
        value_transformer=transformer,
    )


# Made with Bob
