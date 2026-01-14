# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.commands.show_details import (
    register_show_details_command,
)
from orchestrator.cli.commands.show_entities import (
    register_show_entities_command,
)
from orchestrator.cli.commands.show_related import (
    register_show_related_command,
)
from orchestrator.cli.commands.show_requests import (
    register_show_requests_command,
)
from orchestrator.cli.commands.show_results import (
    register_show_results_command,
)
from orchestrator.cli.commands.show_summary import (
    register_show_summary_command,
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
register_show_entities_command(show_command)
register_show_related_command(show_command)
register_show_requests_command(show_command)
register_show_results_command(show_command)
register_show_summary_command(show_command)


def register_show_command(app: typer.Typer) -> None:
    app.add_typer(
        show_command,
        name="show",
        options_metavar="",
        no_args_is_help=True,
    )
