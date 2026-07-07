# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Shared rendering helper for `ado show stats` output."""

import json
import pathlib
import typing

import yaml

from orchestrator.cli.models.types import AdoShowStatsSupportedOutputFormats
from orchestrator.cli.utils.output.prints import (
    SUCCESS,
    console_print,
    magenta,
)

if typing.TYPE_CHECKING:
    import pandas as pd


def render_stats_dataframe(
    df: "pd.DataFrame",
    output_format: AdoShowStatsSupportedOutputFormats,
    output_file: pathlib.Path | None,
    render_output: bool,
    index_column: str = "IDENTIFIER",
) -> None:
    """Render a stats DataFrame to the requested output format.

    Writes to *output_file* when provided, otherwise prints to stdout.

    Args:
        df: DataFrame to render. Must contain *index_column* when
            using json or yaml output formats.
        output_format: One of TABLE, MARKDOWN_TABLE, CSV, JSON, YAML.
        output_file: Optional path to write output to instead of stdout.
        render_output: When True and output_format is MARKDOWN_TABLE,
            render the markdown via rich in the console.
        index_column: Column to use as the outer key in json/yaml output.
            Defaults to "IDENTIFIER".
    """
    Fmt = AdoShowStatsSupportedOutputFormats
    match output_format:
        case Fmt.JSON | Fmt.YAML:
            # Convert to a plain dict so both branches share the same structure.
            # We avoid df.to_json() because it lacks a default= coercion for
            # non-serialisable types (e.g. datetime) and has no YAML equivalent;
            # going through a dict keeps JSON and YAML output structurally identical.
            data = df.set_index(index_column).to_dict(orient="index")
            if output_format == Fmt.JSON:
                result: str = json.dumps(data, indent=2, default=str)
            else:
                result = yaml.dump(data, default_flow_style=False)
        case Fmt.TABLE:
            import rich.box

            from orchestrator.utilities.rich import (
                dataframe_to_rich_table,
                render_to_string,
            )

            table = dataframe_to_rich_table(
                df,
                show_edge=True,
                show_index=True,
                box=rich.box.SQUARE,
                do_not_truncate_columns=True,
            )
            result = render_to_string(table, auto_width=True)
        case Fmt.MARKDOWN_TABLE:
            result = df.to_markdown()
        case _:
            # CSV
            result = df.to_csv(index=False)

    if output_file:
        output_file.write_text(result)
        console_print(
            f"{SUCCESS} Output saved as {magenta(str(output_file))}",
            stderr=True,
        )
        return

    if (
        render_output
        and output_format == AdoShowStatsSupportedOutputFormats.MARKDOWN_TABLE
    ):
        from rich.markdown import Markdown

        console_print(Markdown(result))
        return

    console_print(result)
