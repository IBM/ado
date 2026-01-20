# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.cli.core.config import AdoConfiguration
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.output import pydantic_model_as_yaml


# ado context
def test_ado_context_print_active_context(tmp_path: Path) -> None:
    """
    We expect ado to create the local context and have it as default.
    """
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "context",
        ],
    )

    assert result.exit_code == 0
    expected_output = (
        "INFO:   Initializing contexts - local is now your default context.\nlocal\n"
    )
    assert expected_output in result.output


def test_ado_context_override_sets_active_context(
    valid_ado_project_context: ProjectContext,
    tmp_path: Path,
    random_identifier: Callable[[], str],
) -> None:
    """
    We expect ado to have the context from valid_ado_project_context
    as active context.
    """
    context_location = tmp_path / f"{random_identifier()}.yaml"
    context_location.write_text(pydantic_model_as_yaml(valid_ado_project_context))

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "-c",
            str(context_location),
            "context",
        ],
    )

    context_location.unlink()

    assert result.exit_code == 0
    assert result.output.strip() == valid_ado_project_context.project


def test_ado_context_cannot_set_nonexisting_context(
    tmp_path: Path,
) -> None:
    """
    We expect ado to disallow setting a context that does not exist.
    """

    # By initializing AdoConfiguration in the tmp_path, the "local"
    # context will be automatically created, along with the folder
    # structure.
    ado_configuration = AdoConfiguration.load(
        do_not_fail_on_available_contexts=True, _override_config_dir=tmp_path
    )

    # We create empty contexts where ado expects them, so they appear
    # in the list of available contexts
    ado_configuration.project_context_path_for_context("first-context").touch()
    ado_configuration.project_context_path_for_context("second-context").touch()

    runner = CliRunner()
    available_contexts = sorted(["local", "first-context", "second-context"])

    # We try to activate a "third-context", which does not exist
    activate_nonexistent_context_result = runner.invoke(
        ado,
        ["--override-ado-app-dir", str(tmp_path), "context", "third-context"],
    )
    assert activate_nonexistent_context_result.exit_code == 1
    activate_nonexistent_context_expected_output = (
        "ERROR:  third-context is not in the available contexts.\n"
        f"HINT:   The available contexts are {available_contexts}\n"
    )
    assert (
        activate_nonexistent_context_result.output
        == activate_nonexistent_context_expected_output
    )


def test_ado_context_set_context(
    tmp_path: Path,
) -> None:
    """
    We expect ado to allow activating contexts
    """

    # By initializing AdoConfiguration in the tmp_path, the "local"
    # context will be automatically created, along with the folder
    # structure.
    ado_configuration = AdoConfiguration.load(
        do_not_fail_on_available_contexts=True, _override_config_dir=tmp_path
    )

    # We create empty contexts where ado expects them, so they appear
    # in the list of available contexts
    ado_configuration.project_context_path_for_context("first-context").touch()
    ado_configuration.project_context_path_for_context("second-context").touch()

    runner = CliRunner()
    activate_context_result = runner.invoke(
        ado,
        ["--override-ado-app-dir", str(tmp_path), "context", "second-context"],
    )
    assert activate_context_result.exit_code == 0
    activate_context_expected_output = "Success! Now using context second-context\n"
    assert activate_context_result.output == activate_context_expected_output


# ado contexts
def test_ado_contexts_list_contexts(tmp_path: Path) -> None:
    """
    We expect ado to list three contexts
    """

    # By initializing AdoConfiguration in the tmp_path, the "local"
    # context will be automatically created, along with the folder
    # structure.
    ado_configuration = AdoConfiguration.load(
        do_not_fail_on_available_contexts=True, _override_config_dir=tmp_path
    )

    # We create empty contexts where ado expects them, so they appear
    # in the list of available contexts
    ado_configuration.project_context_path_for_context("first-context").touch()
    ado_configuration.project_context_path_for_context("second-context").touch()

    runner = CliRunner()

    # Test with the default (rich) output
    ado_contexts_default_output_result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "contexts",
        ],
    )
    assert ado_contexts_default_output_result.exit_code == 0
    ado_contexts_default_output_expected_output = (
        "          CONTEXT DEFAULT\n"
        "0   first-context        \n"
        "1           local       *\n"
        "2  second-context        \n"
        "\n"
        "The active context is: local\n"
    )
    assert (
        ado_contexts_default_output_result.output
        == ado_contexts_default_output_expected_output
    )

    # Test with the simple output
    ado_contexts_simple_output_result = runner.invoke(
        ado,
        ["--override-ado-app-dir", str(tmp_path), "contexts", "--simple"],
    )
    assert ado_contexts_simple_output_result.exit_code == 0
    ado_contexts_simple_output_expected_output = (
        "first-context\nlocal\nsecond-context\n"
    )
    assert (
        ado_contexts_simple_output_result.output
        == ado_contexts_simple_output_expected_output
    )


def test_ado_contexts_list_contexts_with_context_and_empty_dir_override(
    valid_ado_project_context: ProjectContext,
    random_identifier: Callable[[], str],
    tmp_path: Path,
) -> None:
    """
    We expect ado to fail as there are no contexts available
    in the tmp_path directory.
    """
    context_location = tmp_path / f"{random_identifier()}.yaml"
    context_location.write_text(pydantic_model_as_yaml(valid_ado_project_context))

    runner = CliRunner()
    ado_contexts_result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "-c",
            context_location,
            "contexts",
        ],
    )
    assert ado_contexts_result.exit_code == 1
    ado_contexts_expected_output = (
        "WARN:   There are no contexts available.\n"
        "HINT:   You can create a context with ado create context\n"
    )
    assert ado_contexts_result.output == ado_contexts_expected_output

    # Test with the simple output
    ado_contexts_simple_output_result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "-c",
            context_location,
            "contexts",
            "--simple",
        ],
    )
    assert ado_contexts_simple_output_result.exit_code == 1
    ado_contexts_simple_output_expected_output = (
        "WARN:   There are no contexts available.\n"
        "HINT:   You can create a context with ado create context\n"
    )
    assert (
        ado_contexts_simple_output_result.output
        == ado_contexts_simple_output_expected_output
    )


def test_ado_contexts_list_contexts_with_context_and_valid_dir_override(
    valid_ado_project_context: ProjectContext,
    random_identifier: Callable[[], str],
    tmp_path: Path,
) -> None:
    """
    We expect ado to list the available contexts.

    The overridden context will not be in the output, but will be printed
    as the active context. The simple output will not have it.

    As the overriding context will be the default, the rich print will not
    have a star to mark the active context.
    """

    # By initializing AdoConfiguration in the tmp_path, the "local"
    # context will be automatically created, along with the folder
    # structure.
    ado_configuration = AdoConfiguration.load(
        do_not_fail_on_available_contexts=True, _override_config_dir=tmp_path
    )

    # We create empty contexts where ado expects them, so they appear
    # in the list of available contexts
    ado_configuration.project_context_path_for_context("first-context").touch()
    ado_configuration.project_context_path_for_context("second-context").touch()

    # We prepare our override context
    context_location = tmp_path / f"{random_identifier()}.yaml"
    context_location.write_text(pydantic_model_as_yaml(valid_ado_project_context))

    runner = CliRunner()
    # Test with the default (rich) output
    ado_contexts_default_output_result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "-c",
            context_location,
            "contexts",
        ],
    )
    assert ado_contexts_default_output_result.exit_code == 0
    ado_contexts_default_output_expected_output = (
        "          CONTEXT DEFAULT\n"
        "0   first-context        \n"
        "1           local        \n"
        "2  second-context        \n"
        "\n"
        f"The active context is: {valid_ado_project_context.project}\n"
    )
    assert (
        ado_contexts_default_output_result.output
        == ado_contexts_default_output_expected_output
    )

    # Test with the simple output
    ado_contexts_simple_output_result = runner.invoke(
        ado,
        ["--override-ado-app-dir", str(tmp_path), "contexts", "--simple"],
    )
    assert ado_contexts_simple_output_result.exit_code == 0
    ado_contexts_simple_output_expected_output = (
        "first-context\nlocal\nsecond-context\n"
    )
    assert (
        ado_contexts_simple_output_result.output
        == ado_contexts_simple_output_expected_output
    )
