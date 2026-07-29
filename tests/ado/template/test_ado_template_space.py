# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import yaml
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration


def test_template_space(
    tmp_path: pathlib.Path, random_identifier: Callable[[], str]
) -> None:
    runner = CliRunner()
    file_name = tmp_path / random_identifier()
    result = runner.invoke(
        ado,
        [
            "template",
            "space",
            "--output-file",
            file_name,
        ],
    )
    assert result.exit_code == 0
    assert f"Success! File saved as {file_name}" in result.output


def test_template_space_from_experiment(
    tmp_path: pathlib.Path, random_identifier: Callable[[], str]
) -> None:
    runner = CliRunner()
    file_name = tmp_path / random_identifier()
    result = runner.invoke(
        ado,
        [
            "template",
            "space",
            "--from-experiment",
            "peptide_mineralization",
            "--output-file",
            file_name,
        ],
    )
    assert result.exit_code == 0
    assert f"Success! File saved as {file_name}" in result.output

    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(file_name.read_text())
    )
    assert (
        space_configuration.experiments[0].experimentIdentifier
        == "peptide_mineralization"
    )
    assert space_configuration.experiments[0].actuatorIdentifier == "robotic_lab"
    assert len(space_configuration.entitySpace) == 3


def test_template_space_from_vllm_experiment(
    tmp_path: pathlib.Path, random_identifier: Callable[[], str]
) -> None:
    runner = CliRunner()
    file_name = tmp_path / random_identifier()
    result = runner.invoke(
        ado,
        [
            "template",
            "space",
            "--from-experiment",
            "vllm-bench-deployment",
            "--output-file",
            file_name,
        ],
    )
    assert result.exit_code == 0, result.output
    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(file_name.read_text())
    )
    assert (
        space_configuration.experiments[0].experimentIdentifier
        == "vllm-bench-deployment"
    )
    assert space_configuration.experiments[0].actuatorIdentifier == "vllm_performance"


def test_template_space_from_experiment_with_actuator_prefix(
    tmp_path: pathlib.Path, random_identifier: Callable[[], str]
) -> None:
    """Actuator prefix in --from-experiment disambiguates multi-actuator ids."""
    runner = CliRunner()
    file_name = tmp_path / random_identifier()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            tmp_path,
            "template",
            "space",
            "--from-experiment",
            "robotic_lab.peptide_mineralization",
            "--output-file",
            file_name,
        ],
    )
    assert result.exit_code == 0
    assert f"Success! File saved as {file_name}" in result.output

    space_configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(file_name.read_text())
    )
    assert (
        space_configuration.experiments[0].experimentIdentifier
        == "peptide_mineralization"
    )
    assert space_configuration.experiments[0].actuatorIdentifier == "robotic_lab"


def test_template_space_from_experiment_minified_domains(
    tmp_path: pathlib.Path, random_identifier: Callable[[], str]
) -> None:
    """Template output omits variableType and null fields from each property domain."""
    runner = CliRunner()
    file_name = tmp_path / random_identifier()
    result = runner.invoke(
        ado,
        [
            "template",
            "space",
            "--from-experiment",
            "peptide_mineralization",
            "--output-file",
            file_name,
        ],
    )
    assert result.exit_code == 0

    raw = yaml.safe_load(file_name.read_text())
    for prop_entry in raw.get("entitySpace", []):
        prop_name = prop_entry.get("identifier", "<unknown>")
        domain_dict = prop_entry.get("propertyDomain", {})
        assert "variableType" not in domain_dict, (
            f"variableType should be absent from {prop_name!r} in template output"
        )
        assert (
            "domainRange" not in domain_dict or domain_dict["domainRange"] is not None
        ), f"domainRange should be absent (or non-null) in {prop_name!r}"
        assert "interval" not in domain_dict or domain_dict["interval"] is not None, (
            f"interval should be absent (or non-null) in {prop_name!r}"
        )
        prob_fn = domain_dict.get("probabilityFunction", {})
        assert "parameters" not in prob_fn or prob_fn["parameters"] is not None, (
            f"probabilityFunction.parameters should be absent (or non-null) in {prop_name!r}"
        )
