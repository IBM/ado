# Copyright IBM Corporation 2025, 2026
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
        ["context", "third-context"],
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
    tmp_path: Path, valid_ado_mysql_context_yaml: str
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
    ado_configuration.project_context_path_for_context("first-context").write_text(
        valid_ado_mysql_context_yaml
    )
    ado_configuration.project_context_path_for_context("second-context").write_text(
        valid_ado_mysql_context_yaml
    )

    runner = CliRunner()
    activate_context_result = runner.invoke(
        ado,
        ["context", "second-context"],
    )
    assert activate_context_result.exit_code == 0
    activate_context_expected_output = "Success! Now using context second-context\n"
    assert activate_context_result.output == activate_context_expected_output


def test_ado_context_cannot_set_invalid_context(tmp_path: Path) -> None:
    """
    We expect ado to disallow setting a context that is invalid (exists but has invalid content).
    """

    # By initializing AdoConfiguration in the tmp_path, the "local"
    # context will be automatically created, along with the folder
    # structure.
    ado_configuration = AdoConfiguration.load(
        do_not_fail_on_available_contexts=True, _override_config_dir=tmp_path
    )

    # We create an invalid context (just touch the file, making it empty/invalid)
    ado_configuration.project_context_path_for_context("invalid-context").touch()

    runner = CliRunner()
    # Try to activate the invalid context
    activate_invalid_context_result = runner.invoke(
        ado,
        ["context", "invalid-context"],
    )
    assert activate_invalid_context_result.exit_code == 1
    # Check that the error message indicates the context is not valid
    assert "ERROR" in activate_invalid_context_result.output
    assert (
        "Context invalid-context is not valid:"
        in activate_invalid_context_result.output
    )
    assert "WARN" in activate_invalid_context_result.output
    assert (
        "You must fix the context manually:" in activate_invalid_context_result.output
    )


def test_ado_switches_to_local_when_active_context_becomes_invalid(
    tmp_path: Path, valid_ado_mysql_context_yaml: str
) -> None:
    """
    We expect ado to switch back to local context when the active context becomes invalid.
    """

    # By initializing AdoConfiguration in the tmp_path, the "local"
    # context will be automatically created, along with the folder
    # structure.
    ado_configuration = AdoConfiguration.load(
        do_not_fail_on_available_contexts=True, _override_config_dir=tmp_path
    )

    # We create a valid context and activate it
    ado_configuration.project_context_path_for_context("valid-context").write_text(
        valid_ado_mysql_context_yaml
    )

    runner = CliRunner()
    activate_context_result = runner.invoke(
        ado,
        ["context", "valid-context"],
    )
    assert activate_context_result.exit_code == 0

    # Now we corrupt the active context by overwriting it with invalid content
    ado_configuration.project_context_path_for_context("valid-context").write_text(
        "invalid yaml content"
    )

    # Try to run any ado command (e.g., ado get spaces)
    get_spaces_result = runner.invoke(
        ado,
        ["get", "spaces"],
    )
    assert get_spaces_result.exit_code == 1
    # Check that the error message indicates the context is not valid
    assert "ERROR" in get_spaces_result.output
    assert "The provided project context was not valid:" in get_spaces_result.output
    assert "WARN" in get_spaces_result.output
    assert "You must fix the context manually:" in get_spaces_result.output
    assert "INFO" in get_spaces_result.output
    assert "Your default context will be switched to local" in get_spaces_result.output

    # Verify that the active context has been switched back to local
    check_context_result = runner.invoke(
        ado,
        ["context"],
    )
    assert check_context_result.exit_code == 0
    assert check_context_result.output.strip() == "local"


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
            "contexts",
        ],
    )
    assert ado_contexts_default_output_result.exit_code == 0
    ado_contexts_default_output_expected_output = (
        "┌───────┬────────────────┬────────┐\n"
        "│ INDEX │ CONTEXT        │ ACTIVE │\n"
        "├───────┼────────────────┼────────┤\n"
        "│ 0     │ first-context  │        │\n"
        "│ 1     │ local          │ ✅     │\n"
        "│ 2     │ second-context │        │\n"
        "└───────┴────────────────┴────────┘\n"
    )
    assert (
        ado_contexts_default_output_result.output
        == ado_contexts_default_output_expected_output
    )

    # Test with the name output format
    ado_contexts_name_output_result = runner.invoke(
        ado,
        ["contexts", "-o", "name"],
    )
    assert ado_contexts_name_output_result.exit_code == 0
    ado_contexts_name_output_expected_output = "first-context\nlocal\nsecond-context\n"
    assert (
        ado_contexts_name_output_result.output
        == ado_contexts_name_output_expected_output
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

    # Test with the name output format
    ado_contexts_name_output_result = runner.invoke(
        ado,
        [
            "-c",
            context_location,
            "contexts",
            "-o",
            "name",
        ],
    )
    assert ado_contexts_name_output_result.exit_code == 1
    ado_contexts_name_output_expected_output = (
        "WARN:   There are no contexts available.\n"
        "HINT:   You can create a context with ado create context\n"
    )
    assert (
        ado_contexts_name_output_result.output
        == ado_contexts_name_output_expected_output
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

    # We create contexts where ado expects them, so they appear
    # in the list of available contexts
    valid_context_yaml = pydantic_model_as_yaml(valid_ado_project_context)
    ado_configuration.project_context_path_for_context("first-context").write_text(
        valid_context_yaml
    )
    ado_configuration.project_context_path_for_context("second-context").write_text(
        valid_context_yaml
    )

    # We prepare our override context
    context_location = tmp_path / f"{random_identifier()}.yaml"
    context_location.write_text(valid_context_yaml)

    runner = CliRunner()
    # Test with the default (rich) output
    ado_contexts_default_output_result = runner.invoke(
        ado,
        [
            "-c",
            context_location,
            "contexts",
        ],
    )
    assert ado_contexts_default_output_result.exit_code == 0
    ado_contexts_default_output_expected_output = (
        "┌───────┬────────────────┬────────┐\n"
        "│ INDEX │ CONTEXT        │ ACTIVE │\n"
        "├───────┼────────────────┼────────┤\n"
        "│ 0     │ first-context  │        │\n"
        "│ 1     │ local          │        │\n"
        "│ 2     │ second-context │        │\n"
        "└───────┴────────────────┴────────┘\n"
    )
    assert (
        ado_contexts_default_output_result.output
        == ado_contexts_default_output_expected_output
    )

    # Test with the name output format
    ado_contexts_name_output_result = runner.invoke(
        ado,
        [
            "-c",
            context_location,
            "contexts",
            "-o",
            "name",
        ],
    )
    assert ado_contexts_name_output_result.exit_code == 0
    ado_contexts_name_output_expected_output = "first-context\nlocal\nsecond-context\n"
    assert (
        ado_contexts_name_output_result.output
        == ado_contexts_name_output_expected_output
    )
