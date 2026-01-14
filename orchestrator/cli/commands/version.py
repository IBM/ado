# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typer

from orchestrator.cli.utils.output.prints import console_print


def print_version() -> None:
    from importlib.metadata import version

    console_print(version("ado-core"))


def register_version_command(app: typer.Typer) -> None:
    app.command(
        name="version",
        help="Display ado's version.",
    )(print_version)
