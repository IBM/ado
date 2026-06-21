# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import typing

import rich.box
import rich.style
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

if typing.TYPE_CHECKING:
    import pandas as pd
    from rich.console import OverflowMethod, RenderableType


def get_rich_repr(obj: typing.Any) -> "RenderableType":  # noqa: ANN401
    """Get a rich representation of an object.

    If the object has a __rich__() method, use it.
    Otherwise, fall back to rich.pretty.Pretty for automatic formatting.

    Args:
        obj: Any object to get a rich representation for

    Returns:
        A RenderableType that can be displayed by rich Console
    """
    if hasattr(obj, "__rich__"):
        return obj.__rich__()
    return Pretty(obj)


def dataframe_to_rich_table(
    df: "pd.DataFrame",
    title: str | None = None,
    show_header: bool = True,
    show_lines: bool = False,
    show_edge: bool = False,
    box: rich.box.Box = rich.box.HEAVY,
    show_index: bool = False,
    overflow: "OverflowMethod" = "ellipsis",
    no_wrap: bool = False,
    do_not_truncate_columns: bool | list[str] = False,
) -> Table:
    """Convert a pandas DataFrame to a rich Table.

    Args:
        df: A pandas DataFrame to convert
        title: Optional title for the table
        show_header: Whether to show column headers
        show_lines: Whether to show lines between rows
        show_edge: Whether to show the table border
        box: Box style for the table
        show_index: Whether to include the DataFrame's index as the first column.
            If True, the index will be displayed with a header label using the
            DataFrame's index name if available, or "INDEX" as a default.
            Default is False for backward compatibility.
        overflow: How to handle content that exceeds column width. Options include
            "ellipsis" (default), "ignore", "fold", "crop". When do_not_truncate_columns
            is active, this is automatically set to "ignore" for affected columns.
        no_wrap: Whether to disable text wrapping in cells. When do_not_truncate_columns
            is active, this is automatically set to True for affected columns.
        do_not_truncate_columns: Controls which columns should not be truncated:
            - False (default): All columns use default truncation behavior
            - True: No columns are truncated (all widths calculated, table width set)
            - list[str]: Only specified column names are not truncated (table width NOT set)
            When active, automatically sets overflow="ignore" and no_wrap=True
            for affected columns.

    Returns:
        A rich Table object ready for rendering
    """
    index_name = str(df.index.name) if df.index.name is not None else "INDEX"

    # Initialize variables for column width control
    table_width = None
    column_width = {}
    console_width = Console().width

    # Determine which columns need width calculation based on input type
    columns_with_no_truncation = []
    disable_truncation_for_all_columns = (
        False  # Track if truncation is disabled for ALL columns
    )

    if do_not_truncate_columns is True:
        # Disable truncation for ALL columns - we'll set table width
        disable_truncation_for_all_columns = True
        columns_with_no_truncation = list(df.columns)
        if show_index:
            columns_with_no_truncation.insert(0, index_name)

    elif isinstance(do_not_truncate_columns, list):
        # Disable truncation for ONLY specified columns - do NOT set table width
        disable_truncation_for_all_columns = False
        columns_with_no_truncation = [
            col for col in do_not_truncate_columns if col in df.columns
        ]
        if show_index and index_name in do_not_truncate_columns:
            columns_with_no_truncation.insert(0, index_name)

    # Calculate widths for columns that should not be truncated
    if columns_with_no_truncation:
        for col in columns_with_no_truncation:
            if col == index_name:
                # Handle index column
                name_len = len(index_name)
                content_len = len(str(df.index.max()))
            else:
                # Handle data columns
                name_len = len(col)
                content_len = df[col].astype(str).str.len().max()

            column_width[col] = max(name_len, content_len)

        # The minimum required table width for us is given by:
        # the width of the non truncated columns
        # + 1 character per truncated column (ellipsis …)
        # + 3 characters per column (2 padding whitespaces and one separator)
        # + 1 additional separator
        # If it's less than that, the table will be rendered with empty columns
        # or might not have all columns.
        total_column_count = len(df.columns) + (1 if show_index else 0)
        non_truncated_width = sum(column_width.values())
        truncated_column_count = total_column_count - len(column_width)
        minimum_truncated_width = truncated_column_count
        minimum_required_table_width = (
            non_truncated_width + minimum_truncated_width + total_column_count * 3 + 1
        )

        # Set table width when truncation is disabled for all columns, or when
        # the console is too narrow to show even the minimum content for the
        # current non-truncated/truncated column mix.
        if (
            disable_truncation_for_all_columns
            or minimum_required_table_width > console_width
        ):
            if disable_truncation_for_all_columns:
                # Formula: sum of widths + padding (2 per col) + separators (1 per col + 1)
                table_width = non_truncated_width + total_column_count * 3 + 1
            else:
                table_width = minimum_required_table_width
        # else: table_width remains None, let rich handle it

    table = Table(
        title=title,
        show_header=show_header,
        show_lines=show_lines,
        show_edge=show_edge,
        box=box,
        width=table_width,
    )

    # Add index column if requested
    if show_index:
        truncation_is_disabled = index_name in column_width
        table.add_column(
            index_name,
            overflow="ignore" if truncation_is_disabled else overflow,
            no_wrap=True if truncation_is_disabled else no_wrap,
            min_width=column_width.get(index_name),
            width=column_width.get(index_name),
        )

    # Add data columns with selective truncation disabling
    for column in df.columns:
        truncation_is_disabled = column in column_width
        table.add_column(
            column,
            overflow="ignore" if truncation_is_disabled else overflow,
            no_wrap=True if truncation_is_disabled else no_wrap,
            min_width=column_width.get(column),
            width=column_width.get(column),
        )

    # Add rows
    for idx, row in df.iterrows():
        # Using pretty ensures we get highlighting
        formatted_row = [
            (
                Pretty(cell)
                if cell is None
                or isinstance(cell, (list, dict, tuple, bool, float, int))
                else str(cell)
            )
            for cell in row
        ]

        # Prepend index value if show_index is True
        if show_index:
            index_value = (
                Pretty(idx)
                if idx is None or isinstance(idx, (list, dict, tuple, bool, float, int))
                else str(idx)
            )
            formatted_row.insert(0, index_value)

        table.add_row(*formatted_row)

    return table


def render_to_string(
    renderable: "RenderableType",
    width: int | None = None,
    auto_width: bool = False,
) -> str:
    """Render a rich renderable to a string.

    Args:
        renderable: A RenderableType object (e.g., Table, Panel, Text, etc.)
        width: Optional width for the console. If None, uses default width.
        auto_width: If True, automatically set console width to match the renderable's
            width (if available). This prevents truncation when writing to files.
            Takes precedence over the width parameter.

    Returns:
        A string representation of the rendered output
    """
    # If auto_width is requested and the renderable has a width attribute, use it
    if auto_width and hasattr(renderable, "width") and renderable.width is not None:
        width = renderable.width

    console = Console(width=width, force_terminal=False, legacy_windows=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()
