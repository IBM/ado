# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import pathlib
import sys
from typing import Annotated

import pydantic
import typer
import yaml

from orchestrator.cli.commands.context import (
    register_context_command,
    register_contexts_command,
)
from orchestrator.cli.commands.create import register_create_command
from orchestrator.cli.commands.delete import register_delete_command
from orchestrator.cli.commands.describe import register_describe_command
from orchestrator.cli.commands.edit import register_edit_command
from orchestrator.cli.commands.get import register_get_command
from orchestrator.cli.commands.show import register_show_command
from orchestrator.cli.commands.template import register_template_command
from orchestrator.cli.commands.upgrade import register_upgrade_command
from orchestrator.cli.commands.version import register_version_command
from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.cli.models.types import AdoLoggingLevel
from orchestrator.cli.utils.output.prints import ERROR, console_print
from orchestrator.cli.utils.remote.dispatch import (
    dispatch as remote_dispatch,
)
from orchestrator.core.remotecontext.config import RemoteExecutionContext
from orchestrator.utilities.location import SQLiteStoreConfiguration
from orchestrator.utilities.logging import configure_logging

# Logging conf
FORMAT = "%(levelname)-9s %(threadName)-30s %(name)-30s: %(funcName)-20s %(asctime)-15s: %(message)s"
logging.basicConfig(format=FORMAT)

SHARED_OPTIONS_PANEL_NAME = "Shared options"

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    options_metavar="[-c | --context <context_file>] [--remote <remote_context_file>] [-l | --log-level <value>]",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    short_help="",
    help="""
    ado is a unified platform for executing computational experiments at scale and analysing their results.

    See https://ibm.github.io/ado/getting-started/ado/ for detailed
    documentation and examples.
    """,
)

configure_logging()


register_context_command(app)
register_contexts_command(app)
register_create_command(app)
register_delete_command(app)
register_describe_command(app)
register_edit_command(app)
register_get_command(app)
register_show_command(app)
register_template_command(app)
register_upgrade_command(app)
register_version_command(app)


# Load CLI plugins via entrypoints
def load_cli_plugins() -> None:
    """Load and register CLI plugins from entry points."""
    from importlib.metadata import entry_points

    cli_logger = logging.getLogger(__name__)
    for cli_plugin in entry_points(group="ado.cli"):
        # We don't want one plugin failing to load to prevent others loading
        try:
            plugin_module = cli_plugin.load()
            if hasattr(plugin_module, "register"):
                plugin_module.register(app)
                cli_logger.debug(
                    f"Loaded CLI plugin: {cli_plugin.name} from {cli_plugin.value}"
                )
            else:
                cli_logger.error(
                    f"CLI plugin {cli_plugin.name} does not have a 'register' function"
                )
        except Exception as e:  # noqa: PERF203
            cli_logger.warning(
                f"Failed to load CLI plugin {cli_plugin.name}: {e}", exc_info=True
            )


load_cli_plugins()


@app.callback()
def common_options(
    ctx: typer.Context,
    project_context_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--context",
            "-c",
            help="""Override the active context for ado's current invocation by loading a ProjectContext from a file.
            No permanent configuration changes will take place.

            Typically used when running on remote Ray clusters.""",
            show_default=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            rich_help_panel=SHARED_OPTIONS_PANEL_NAME,
        ),
    ] = None,
    remote_context_file: Annotated[
        pathlib.Path | None,
        typer.Option(
            "--remote",
            help="""Dispatch this ado command to a remote Ray cluster.

            Provide a path to a RemoteExecutionContext YAML file that describes the
            remote Ray cluster (URL, optional port-forward configuration,
            packages to install, and environment variables).

            When provided, ado will package the necessary files, optionally
            build plugin wheels, and submit the command to the cluster via
            ``ray job submit`` instead of running it locally.

            The active project context must use a non-SQLite (remote) metastore.

            See https://ibm.github.io/ado/getting-started/remote_run/ for details.""",
            show_default=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            rich_help_panel=SHARED_OPTIONS_PANEL_NAME,
        ),
    ] = None,
    log_level: Annotated[
        AdoLoggingLevel,
        typer.Option(
            "-l",
            "--log-level",
            envvar="LOGLEVEL",
            help="""Sets the level of logging for ado's current invocation.

            Refer to https://docs.python.org/3/library/logging.html#logging-levels for additional information.
            """,
            rich_help_panel=SHARED_OPTIONS_PANEL_NAME,
        ),
    ] = AdoLoggingLevel.WARNING,
    override_ado_app_dir: Annotated[
        pathlib.Path | None, typer.Option(help="For testing only", hidden=True)
    ] = None,
) -> None:
    logging.getLogger().setLevel(log_level.value)

    # AP: 25/06/2025
    # Since project_context_file is Optional, there's no real validation on it,
    # we must check that the file exists and is a file on our own.
    if project_context_file:
        if not project_context_file.exists():
            console_print(
                f"{ERROR}The provided path {project_context_file.resolve()} does not exist.",
                stderr=True,
            )
            raise typer.Exit(1)

        if not project_context_file.is_file():
            console_print(
                f"{ERROR}The provided path {project_context_file.resolve()} is not a file.",
                stderr=True,
            )
            raise typer.Exit(1)

    if remote_context_file:
        if not remote_context_file.exists():
            console_print(
                f"{ERROR}The provided path {remote_context_file.resolve()} does not exist.",
                stderr=True,
            )
            raise typer.Exit(1)

        if not remote_context_file.is_file():
            console_print(
                f"{ERROR}The provided path {remote_context_file.resolve()} is not a file.",
                stderr=True,
            )
            raise typer.Exit(1)

    # AP 27/05/2025:
    # If the user is running ado context/contexts we must not fail
    # as it would create a situation where the user is softlocked,
    # unable to activate a context which is required everywhere else.
    # We also disable it for ado create, delegating stricter validation
    # to ado create itself.
    do_not_fail_on_available_contexts = ctx.invoked_subcommand in (
        "context",
        "contexts",
        "create",
    )
    ado_config = AdoConfiguration.load(
        from_project_context=project_context_file,
        do_not_fail_on_available_contexts=do_not_fail_on_available_contexts,
        _override_config_dir=override_ado_app_dir,
    )

    ctx.obj = ado_config

    if remote_context_file:
        _handle_remote_dispatch(
            remote_context_file=remote_context_file,
            ado_config=ado_config,
        )


def _handle_remote_dispatch(
    remote_context_file: pathlib.Path,
    ado_config: AdoConfiguration,
) -> None:
    """Validate, build, and dispatch the current ado invocation to a remote Ray cluster.

    Called from ``common_options`` when ``--remote`` is present.
    Exits the process via ``typer.Exit`` after dispatching (or on error).

    Args:
        remote_context_file: Path to the RemoteExecutionContext YAML file.
        ado_config: The loaded AdoConfiguration for this invocation. Must have a valid
            project context loaded (either from ``-c`` or the active context).
    """
    # Guard: remote execution requires a non-SQLite project context
    # Note: AdoConfiguration.load() ensures project_context is set when not using
    # do_not_fail_on_available_contexts, which is not the case here
    project_context = ado_config.project_context
    if project_context is None:
        # This should never happen given AdoConfiguration.load() behavior,
        # but we check for type safety
        console_print(
            f"{ERROR}Cannot use --remote: no project context is active.\n"
            "Activate a context with 'ado context set' or provide one with -c.",
            stderr=True,
        )
        raise typer.Exit(1)

    if isinstance(project_context.metadataStore, SQLiteStoreConfiguration):
        console_print(
            f"{ERROR}Cannot use --remote with a SQLite project context.\n"
            "Remote execution requires a non-SQLite (e.g. MySQL) metastore so the "
            "remote Ray cluster can connect to the same database.",
            stderr=True,
        )
        raise typer.Exit(1)

    # Load the remote execution context
    try:
        remote_context = RemoteExecutionContext.model_validate(
            yaml.safe_load(remote_context_file.read_text())
        )
    except pydantic.ValidationError as e:
        console_print(
            f"{ERROR}The remote execution context file is not valid:\n{e}",
            stderr=True,
        )
        raise typer.Exit(1) from e

    try:
        exit_code = remote_dispatch(
            remote_context=remote_context,
            project_context=project_context,
            argv=sys.argv[1:],
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console_print(
            f"{ERROR}Remote dispatch failed: {exc}",
            stderr=True,
        )
        raise typer.Exit(1) from exc

    if exit_code != 0:
        console_print(
            f"{ERROR}Remote dispatch failed (exit code {exit_code}).",
            stderr=True,
        )
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
