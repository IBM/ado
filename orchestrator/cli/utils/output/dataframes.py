# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typing
from pathlib import Path
from typing import Literal

import rich.box

from orchestrator.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    HINT,
    SUCCESS,
    console_print,
    magenta,
)
from orchestrator.utilities.rich import dataframe_to_rich_table

if typing.TYPE_CHECKING:
    import pandas as pd

DATAFRAME_ROWS_THRESHOLD = 50
DATAFRAME_COLS_THRESHOLD = 20


def df_to_output(
    df: "pd.DataFrame",
    output_format: Literal["console", "json", "csv"],
    output_file: Path | str | None = None,
    no_trunc: bool = False,
) -> None:
    """Output a dataframe to stdout or file.

    Args:
        df: The dataframe to output
        output_format: The format to use (console, json, or csv)
        output_file: Optional file path. If None, output goes to stdout (except for console format)
        no_trunc: Whether to avoid truncating columns in console output
    """
    if df.empty:
        console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
        return

    # For csv and json formats
    match output_format:
        case "console":
            output = dataframe_to_rich_table(
                df,
                show_edge=True,
                show_index=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=no_trunc,
            )
        case "csv":
            output = df.to_csv()
        case "json":
            output = df.to_json()

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
