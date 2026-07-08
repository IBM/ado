# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from pathlib import Path
from typing import Literal

import rich.box

from ado.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    HINT,
    SUCCESS,
    console_print,
    magenta,
)
from ado.utilities.rich import dataframe_to_rich_table

if typing.TYPE_CHECKING:
    import pandas as pd

DATAFRAME_ROWS_THRESHOLD = 50
DATAFRAME_COLS_THRESHOLD = 20


def df_to_output(
    df: "pd.DataFrame",
    output_format: Literal["table", "json", "csv"],
    output_file: Path | None = None,
    no_trunc: bool = False,
) -> None:
    """Output a dataframe to stdout or file.

    Args:
        df: The dataframe to output
        output_format: The format to use (table, json, or csv)
        output_file: Optional file path. If None, output goes to stdout.
            When writing table format to a file, columns are not truncated by default.
        no_trunc: Whether to avoid truncating columns in table output (console only).
            Ignored when output_file is provided as truncation is automatically disabled.
    """
    if df.empty:
        console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
        return

    match output_format:
        case "table":
            # When writing to file, avoid truncating columns by default
            do_not_truncate = True if output_file else no_trunc

            table = dataframe_to_rich_table(
                df,
                show_edge=True,
                show_index=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=do_not_truncate,
            )
            if output_file:
                # Convert table to string for file output
                from ado.utilities.rich import render_to_string

                output_str = render_to_string(table, auto_width=True)
                output_file.write_text(output_str)
                console_print(
                    f"{SUCCESS} Output saved as {magenta(str(output_file))}",
                    stderr=True,
                )
            else:
                console_print(table)
                if (
                    df.shape[0] >= DATAFRAME_ROWS_THRESHOLD
                    or df.shape[1] >= DATAFRAME_COLS_THRESHOLD
                ):
                    console_print(
                        f"{HINT}The output is very large. Consider redirecting it to a file",
                        stderr=True,
                    )
            return
        case "csv":
            output = df.to_csv()
        case "json":
            output = df.to_json() or ""
        case _:
            raise ValueError(f"Unsupported output format: {output_format!r}")

    if not output_file:
        console_print(output)
        if (
            df.shape[0] >= DATAFRAME_ROWS_THRESHOLD
            or df.shape[1] >= DATAFRAME_COLS_THRESHOLD
        ):
            console_print(
                f"{HINT}The output is very large. Consider redirecting it to a file",
                stderr=True,
            )
    else:
        output_file.write_text(output)
        console_print(
            f"{SUCCESS} Output saved as {magenta(str(output_file))}", stderr=True
        )
