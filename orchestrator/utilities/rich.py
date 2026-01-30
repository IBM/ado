# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import typing

import rich.box
import rich.style
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

if typing.TYPE_CHECKING:
    import pandas as pd
    from rich.console import RenderableType


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
) -> Table:
    """Convert a pandas DataFrame to a rich Table.

    Args:
        df: A pandas DataFrame to convert
        title: Optional title for the table
        show_header: Whether to show column headers
        show_lines: Whether to show lines between rows
        show_edge: Whether to show the table border

    Returns:
        A rich Table object ready for rendering
    """
    table = Table(
        title=title,
        show_header=show_header,
        show_lines=show_lines,
        show_edge=show_edge,
        box=rich.box.HEAVY,
    )

    # Add columns
    for column in df.columns:
        table.add_column(str(column))

    # Add rows
    for _, row in df.iterrows():
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
