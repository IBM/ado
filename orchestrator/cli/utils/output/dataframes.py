# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Literal

import typer

from orchestrator.cli.utils.output.prints import (
    ADO_INFO_EMPTY_DATAFRAME,
    HINT,
    SUCCESS,
    console_print,
    cyan,
    magenta,
)

if typing.TYPE_CHECKING:
    import pandas as pd

DATAFRAME_ROWS_THRESHOLD = 50
DATAFRAME_COLS_THRESHOLD = 20


def df_to_output(
    df: "pd.DataFrame",
    output_format: Literal["console", "json", "csv"],
    file_name: str | None = None,
) -> None:
    if output_format != "console" and not file_name:
        console_print(
            f"Outputting a dataframe as {output_format} requires specifying a file name",
            stderr=True,
        )
        raise typer.Exit(1)

    if df.empty:
        console_print(ADO_INFO_EMPTY_DATAFRAME, stderr=True)
        return

    if output_format == "console":
        console_print(df, has_pandas_content=True)
        if (
            df.shape[0] >= DATAFRAME_ROWS_THRESHOLD
            or df.shape[1] >= DATAFRAME_COLS_THRESHOLD
        ):
            console_print(
                f"{HINT}The output is very large. "
                f"Consider using {cyan('--output-format csv')}",
                stderr=True,
            )
        return
    if output_format == "csv":
        df.to_csv(file_name)
    elif output_format == "json":
        df.to_json(file_name)

    console_print(f"{SUCCESS} Output saved as {magenta(file_name)}", stderr=True)
