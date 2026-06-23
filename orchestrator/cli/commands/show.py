# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.commands.show_details import (
    register_show_details_command,
)
from orchestrator.cli.commands.show_measurements import (
    register_show_measurements_command,
)
from orchestrator.cli.commands.show_related import (
    register_show_related_command,
)
from orchestrator.cli.commands.show_summary import (
    register_show_summary_command,
)
from orchestrator.cli.commands.show_trace import (
    register_show_trace_command,
)

show_command = typer.Typer(
    no_args_is_help=True,
    help="""
    Display content related to one or more resources.

    See https://ibm.github.io/ado/getting-started/ado/#ado-show for detailed
    documentation and examples.
    """,
    rich_markup_mode="rich",
)

register_show_details_command(show_command)
register_show_measurements_command(show_command)
register_show_related_command(show_command)
register_show_summary_command(show_command)
register_show_trace_command(show_command)


def register_show_command(app: typer.Typer) -> None:
    app.add_typer(
        show_command,
        name="show",
        options_metavar="",
        no_args_is_help=True,
    )
