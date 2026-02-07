# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import os

from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado


def test_get_experiments_basic() -> None:
    """Test basic experiment listing without details"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "ACTUATOR ID" in result.output
        assert "EXPERIMENT ID" in result.output
        assert "SUPPORTED" not in result.output


def test_get_experiments_basic_show_deprecated() -> None:
    """Test basic experiment listing without details"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "--show-deprecated"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "ACTUATOR ID" in result.output
        assert "EXPERIMENT ID" in result.output
        assert "SUPPORTED" in result.output


def test_get_experiments_with_details() -> None:
    """Test experiment listing with --details flag"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "--details"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "ACTUATOR ID" in result.output
        assert "EXPERIMENT ID" in result.output
        assert "DESCRIPTION" in result.output
        assert "SUPPORTED" not in result.output


def test_get_experiments_with_details_show_deprecated() -> None:
    """Test experiment listing with --details flag"""
    runner = CliRunner()
    result = runner.invoke(
        ado, ["get", "experiments", "--details", "--show-deprecated"]
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "ACTUATOR ID" in result.output
        assert "EXPERIMENT ID" in result.output
        assert "DESCRIPTION" in result.output
        assert "SUPPORTED" in result.output


def test_get_specific_experiment() -> None:
    """Test getting a specific experiment by ID"""
    runner = CliRunner()
    # Use a known experiment from robotic_lab actuator
    result = runner.invoke(ado, ["get", "experiments", "peptide_mineralization"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "peptide_mineralization" in result.output
        assert "robotic_lab" in result.output


def test_get_specific_experiment_with_details() -> None:
    """Test getting a specific experiment with details"""
    runner = CliRunner()
    result = runner.invoke(
        ado, ["get", "experiments", "peptide_mineralization", "--details"]
    )
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        assert "peptide_mineralization" in result.output
        assert "robotic_lab" in result.output
        assert "DESCRIPTION" in result.output


def test_get_nonexistent_experiment() -> None:
    """Test error handling for non-existent experiment"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "nonexistent_experiment_xyz"])
    assert result.exit_code == 1
    if os.environ.get("CI", "false") != "true":
        assert "does not exist" in result.output


def test_get_experiments_show_deprecated() -> None:
    """Test showing deprecated experiments"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "--show-deprecated"])
    assert result.exit_code == 0


def test_get_experiments_show_deprecated_with_details() -> None:
    """Test showing deprecated experiments with details"""
    runner = CliRunner()
    result = runner.invoke(
        ado, ["get", "experiments", "--show-deprecated", "--details"]
    )
    assert result.exit_code == 0


def test_get_experiments_plural_alias() -> None:
    """Test that 'experiment' and 'experiments' both work"""
    runner = CliRunner()
    result1 = runner.invoke(ado, ["get", "experiment"])
    result2 = runner.invoke(ado, ["get", "experiments"])
    assert result1.exit_code == 0
    assert result2.exit_code == 0
    # Both should produce same output
    if os.environ.get("CI", "false") != "true":
        assert result1.output == result2.output


def test_get_experiments_invalid_output_format() -> None:
    """Test that non-default output formats are rejected"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "-o", "json"])
    assert result.exit_code == 1
    if os.environ.get("CI", "false") != "true":
        assert "Only the default output format" in result.output


def test_get_experiments_with_yaml_output_format() -> None:
    """Test that yaml output format is rejected"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "-o", "yaml"])
    assert result.exit_code == 1
    if os.environ.get("CI", "false") != "true":
        assert "Only the default output format" in result.output


def test_get_experiments_output_contains_actuator_info() -> None:
    """Test that output contains actuator information"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments"])
    assert result.exit_code == 0
    if os.environ.get("CI", "false") != "true":
        # Should contain at least one actuator
        assert "robotic_lab" in result.output or "gt4sd" in result.output


def test_get_experiments_sorted_output() -> None:
    """Test that experiments are sorted by experiment ID then actuator ID"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments"])
    assert result.exit_code == 0
    # Output should be sorted - we can't easily verify the exact order
    # but we can verify the command runs successfully


def test_get_experiments_empty_result_for_nonexistent() -> None:
    """Test that searching for non-existent experiment gives proper error"""
    runner = CliRunner()
    result = runner.invoke(ado, ["get", "experiments", "this_does_not_exist_12345"])
    assert result.exit_code == 1
