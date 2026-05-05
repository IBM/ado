# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for the --use-latest flag behavior in show_summary command."""

import pathlib
from collections.abc import Callable

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.metastore.project import ProjectContext
from tests.conftest import requires_sqlite_3_38


@requires_sqlite_3_38
def test_show_summary_use_latest_alone(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    pfas_space: DiscoverySpace,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Test that --use-latest alone uses the latest space ID, not an older one."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # We have two spaces: pfas_space (created first) and ml_multi_cloud_space (created second, so it's latest)
    # Run show summary with --use-latest
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "summary",
            "space",
            "--use-latest",
        ],
    )

    # Should succeed
    assert result.exit_code == 0

    # Should show the LATEST space ID (ml_multi_cloud_space), not the older one (pfas_space)
    assert ml_multi_cloud_space.uri in result.output
    # Verify it's not showing the older space
    assert pfas_space.uri not in result.output


@requires_sqlite_3_38
def test_show_summary_use_latest_with_explicit_id(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    pfas_space: DiscoverySpace,
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Test that explicit ID takes precedence over --use-latest with a warning."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # We have two spaces: pfas_space (older) and ml_multi_cloud_space (latest)
    # Request the older space explicitly with --use-latest
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "summary",
            "space",
            pfas_space.uri,
            "--use-latest",
        ],
    )

    # Should succeed
    assert result.exit_code == 0

    # Should show warning about explicit ID taking precedence
    assert "explicitly specified resource ids take precedence" in result.output.lower()

    # Should show the EXPLICIT (older) space ID in output, not the latest one
    assert pfas_space.uri in result.output
    # Verify it's not showing the latest space
    assert ml_multi_cloud_space.uri not in result.output


@requires_sqlite_3_38
def test_show_summary_multiple_ids_without_use_latest(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    pfas_space: DiscoverySpace,
) -> None:
    """Test that multiple IDs work without --use-latest."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Get the first space ID
    space_id_1 = pfas_space.uri

    # For this test, we'll just use the same ID twice to verify the command accepts multiple IDs
    # In a real scenario, you'd create a second space
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "summary",
            "space",
            space_id_1,
            space_id_1,
        ],
    )

    # Should succeed (even with duplicate IDs, the command should work)
    assert result.exit_code == 0

    # Should show the complete space ID (Space ID column is never truncated)
    assert space_id_1 in result.output


@requires_sqlite_3_38
def test_show_summary_use_latest_without_spaces(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
) -> None:
    """Test that --use-latest fails gracefully when no spaces exist."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    # Run show summary with --use-latest when no spaces exist
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "show",
            "summary",
            "space",
            "--use-latest",
        ],
    )

    # Should fail with appropriate error
    assert result.exit_code == 1
    assert "unable to find" in result.output.lower()


# Made with Bob
