# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable

import yaml
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration


def test_template_space(
    tmp_path: pathlib.Path, random_identifier: Callable[[], str]
) -> None:
    runner = CliRunner()
    file_name = tmp_path / random_identifier()
    result = runner.invoke(
        ado,
        ["--override-ado-app-dir", tmp_path, "template", "space", "-o", file_name],
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
            "--override-ado-app-dir",
            tmp_path,
            "template",
            "space",
            "--from-experiment",
            "peptide_mineralization",
            "-o",
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
