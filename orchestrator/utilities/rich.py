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
    do_not_truncate_column_content: bool = False,
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
            "ellipsis" (default), "ignore", "fold", "crop". When do_not_truncate_column_content
            is True, this is automatically set to "ignore".
        no_wrap: Whether to disable text wrapping in cells. When do_not_truncate_column_content
            is True, this is automatically set to True.
        do_not_truncate_column_content: If True, automatically calculates column widths
            to fit all content without truncation. This sets overflow="ignore" and no_wrap=True,
            and calculates the minimum table width needed to display all content.

    Returns:
        A rich Table object ready for rendering
    """
    index_name = str(df.index.name) if df.index.name is not None else "INDEX"

    # Initialize variables for column width control
    table_width = None
    column_width = {}

    if do_not_truncate_column_content:
        # Override overflow and no_wrap settings to prevent truncation
        overflow = "ignore"
        no_wrap = True

        # Calculate column name lengths
        column_names_length = {col: len(col) for col in df.columns}
        column_names_length[index_name] = len(index_name)

        # Calculate longest string in each column
        longest_string_in_column = df.apply(
            lambda col: col.astype(str).str.len().max()
        ).to_dict()
        longest_string_in_column[index_name] = len(str(df.index.max()))

        # Determine column widths (max of column name length and longest content)
        column_width = {
            col: max(column_names_length[col], longest_string_in_column[col])
            for col in column_names_length
        }

        # Calculate total table width:
        #   sum of column widths
        # + padding (2 per column)
        # + separators (1 per column + 1 for borders)
        table_width = sum(column_width.values()) + len(column_width) * 3 + 1

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
        table.add_column(
            index_name,
            overflow=overflow,
            no_wrap=no_wrap,
            min_width=column_width.get(index_name),
            width=column_width.get(index_name),
        )

    # Add columns
    for column in df.columns:
        table.add_column(
            column,
            overflow=overflow,
            no_wrap=no_wrap,
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


def render_to_string(renderable: "RenderableType", width: int | None = None) -> str:
    """Render a rich renderable to a string.

    Args:
        renderable: A RenderableType object (e.g., Table, Panel, Text, etc.)
        width: Optional width for the console. If None, uses default width.

    Returns:
        A string representation of the rendered output
    """
    console = Console(width=width, force_terminal=False, legacy_windows=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()
