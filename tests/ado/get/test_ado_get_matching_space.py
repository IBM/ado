# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import datetime as dt
import pathlib
from collections.abc import Callable

import pandas as pd
from testcontainers.community.mysql import MySqlContainer
from typer.testing import CliRunner

from ado.cli.core.cli import app as ado
from ado.core import CoreResourceKinds, DiscoverySpaceResource
from ado.core.discoveryspace.config import (
    DiscoverySpaceConfiguration,
    ParameterizationMatch,
    SpaceHierarchy,
    VersionMatch,
)
from ado.core.resources import ADOResource
from ado.metastore.project import ProjectContext
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference
from ado.utilities.output import pydantic_model_as_yaml
from tests.conftest import requires_sqlite_3_38
from tests.utilities.cli_rendering import render_dataframe_to_cli_output


@requires_sqlite_3_38
def test_get_space_matching_space_id(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    create_resources: Callable[[list[ADOResource]], None],
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    make_continuous_space_resource: Callable[..., DiscoverySpaceResource],
    backdate_resource: Callable[[str, CoreResourceKinds, dt.datetime], None],
    experiment_reference: ExperimentReference,
) -> None:
    """ado get space --matching-space-id returns matching spaces and displays match columns."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    exp_a = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp1", "experimentVersion": "1.0.0"}
    )
    exp_b = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp2", "experimentVersion": "1.0.0"}
    )
    exp_c = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp3", "experimentVersion": "1.0.0"}
    )
    exp_d = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp4", "experimentVersion": "1.0.0"}
    )
    exp_e = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp5", "experimentVersion": "1.0.0"}
    )

    # The entity space and measurement space hierarchies are independent:
    # a smaller entity space with more experiments is an entity subspace and
    # a measurement superspace (and vice versa).
    spaces = [
        make_continuous_space_resource(
            "reference-space", [0, 10], [exp_a, exp_b], name="Reference"
        ),
        make_continuous_space_resource(
            "space-identical", [0, 10], [exp_a, exp_b], name="Identical"
        ),
        make_continuous_space_resource(
            "space-smaller-entity-space-more-experiments",
            [2, 8],
            [exp_a, exp_b, exp_c],
            name="SmallerEntitySpaceMoreExperiments",
        ),
        make_continuous_space_resource(
            "space-bigger-entity-space-fewer-experiments",
            [0, 20],
            [exp_a],
            name="BiggerEntitySpaceFewerExperiments",
        ),
        make_continuous_space_resource(
            "space-overlapping", [5, 15], [exp_a, exp_d], name="Overlap"
        ),
        make_continuous_space_resource(
            "space-disjoint-measurement-space",
            [0, 10],
            [exp_e],
            name="DisjointMeasurementSpace",
        ),
        make_continuous_space_resource(
            "space-disjoint-entity-space",
            [100, 200],
            [exp_a, exp_b],
            name="DisjointEntitySpace",
        ),
    ]
    create_resources(spaces)

    # Distinct ages make the AGE column and the AGE-descending row order deterministic
    now = dt.datetime.now(dt.UTC)
    for days_ago, space in zip(range(len(spaces), 0, -1), spaces, strict=True):
        backdate_resource(
            space.identifier,
            CoreResourceKinds.DISCOVERYSPACE,
            now - dt.timedelta(days=days_ago),
        )

    result = runner.invoke(
        ado, ["get", "space", "--matching-space-id", "reference-space", "--no-trunc"]
    )
    assert result.exit_code == 0, result.output

    # The two disjoint spaces are filtered out
    expected_output = render_dataframe_to_cli_output(
        pd.DataFrame(
            {
                "IDENTIFIER": [
                    "reference-space",
                    "space-identical",
                    "space-smaller-entity-space-more-experiments",
                    "space-bigger-entity-space-fewer-experiments",
                    "space-overlapping",
                ],
                "NAME": [
                    "Reference",
                    "Identical",
                    "SmallerEntitySpaceMoreExperiments",
                    "BiggerEntitySpaceFewerExperiments",
                    "Overlap",
                ],
                "AGE": ["7d0h", "6d0h", "5d0h", "4d0h", "3d0h"],
                "ENTITY_SPACE_MATCH": [
                    SpaceHierarchy.EQUAL_SPACE.value,
                    SpaceHierarchy.EQUAL_SPACE.value,
                    SpaceHierarchy.SUB_SPACE.value,
                    SpaceHierarchy.SUPER_SPACE.value,
                    SpaceHierarchy.OVERLAPPING.value,
                ],
                "MEASUREMENT_SPACE_MATCH": [
                    SpaceHierarchy.EQUAL_SPACE.value,
                    SpaceHierarchy.EQUAL_SPACE.value,
                    SpaceHierarchy.SUPER_SPACE.value,
                    SpaceHierarchy.SUB_SPACE.value,
                    SpaceHierarchy.OVERLAPPING.value,
                ],
                "VERSION_MATCH": [VersionMatch.IDENTICAL.value] * 5,
                "PARAMETERIZATION_MATCH": [ParameterizationMatch.NOT_APPLICABLE.value]
                * 5,
            }
        ),
        do_not_truncate_columns=True,
    )
    assert expected_output in result.output, (
        f"Expected output:\n{expected_output}\nnot found in:\n{result.output}"
    )


@requires_sqlite_3_38
def test_get_space_matching_space_file(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    create_resources: Callable[[list[ADOResource]], None],
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    make_continuous_space_resource: Callable[..., DiscoverySpaceResource],
    make_continuous_space_configuration: Callable[..., DiscoverySpaceConfiguration],
    backdate_resource: Callable[[str, CoreResourceKinds, dt.datetime], None],
    experiment_reference: ExperimentReference,
) -> None:
    """ado get space --matching-space matches against a space YAML configuration file."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    exp_a = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp1", "experimentVersion": "1.0.0"}
    )
    exp_b = experiment_reference.model_copy(
        update={"experimentIdentifier": "exp2", "experimentVersion": "1.0.0"}
    )

    space_target = make_continuous_space_resource(
        "space-target", [0, 10], experiments=[exp_a, exp_b], name="Target"
    )
    create_resources([space_target])
    backdate_resource(
        "space-target",
        CoreResourceKinds.DISCOVERYSPACE,
        dt.datetime.now(dt.UTC) - dt.timedelta(days=3),
    )

    file_space_config = make_continuous_space_configuration(
        [0, 10], experiments=[exp_a, exp_b]
    )
    space_file = tmp_path / "space.yaml"
    space_file.write_text(pydantic_model_as_yaml(file_space_config))

    result = runner.invoke(
        ado, ["get", "space", "--matching-space", str(space_file), "--no-trunc"]
    )
    assert result.exit_code == 0, result.output

    expected_output = render_dataframe_to_cli_output(
        pd.DataFrame(
            {
                "IDENTIFIER": ["space-target"],
                "NAME": ["Target"],
                "AGE": ["3d0h"],
                "ENTITY_SPACE_MATCH": [SpaceHierarchy.EQUAL_SPACE.value],
                "MEASUREMENT_SPACE_MATCH": [SpaceHierarchy.EQUAL_SPACE.value],
                "VERSION_MATCH": [VersionMatch.IDENTICAL.value],
                "PARAMETERIZATION_MATCH": [ParameterizationMatch.NOT_APPLICABLE.value],
            }
        ),
        do_not_truncate_columns=True,
    )
    assert expected_output in result.output, (
        f"Expected output:\n{expected_output}\nnot found in:\n{result.output}"
    )


@requires_sqlite_3_38
def test_get_space_matching_space_version_and_parameterization(
    tmp_path: pathlib.Path,
    mysql_test_instance: MySqlContainer,
    create_resources: Callable[[list[ADOResource]], None],
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    make_continuous_space_resource: Callable[..., DiscoverySpaceResource],
    backdate_resource: Callable[[str, CoreResourceKinds, dt.datetime], None],
    experiment_reference: ExperimentReference,
    defaultParameterization: list[ConstitutivePropertyValue],
    customParameterization: list[ConstitutivePropertyValue],
) -> None:
    """ado get space --matching-space-id --details displays version and parameterization matches."""
    runner = CliRunner()
    create_active_ado_context(
        runner=runner, path=tmp_path, project_context=valid_ado_project_context
    )

    exp_ref = experiment_reference.model_copy(
        update={
            "experimentIdentifier": "exp1",
            "experimentVersion": "1.0.0",
            "parameterization": defaultParameterization,
        }
    )
    exp_cand = experiment_reference.model_copy(
        update={
            "experimentIdentifier": "exp1",
            "experimentVersion": "1.1.0",
            "parameterization": customParameterization,
        }
    )

    reference_space = make_continuous_space_resource(
        "reference-space", [0, 10], experiments=[exp_ref], name="Reference"
    )
    space_different_parameterization = make_continuous_space_resource(
        "space-different-parameterization",
        [0, 10],
        experiments=[exp_cand],
        name="Different",
    )

    create_resources([reference_space, space_different_parameterization])
    now = dt.datetime.now(dt.UTC)
    backdate_resource(
        "reference-space",
        CoreResourceKinds.DISCOVERYSPACE,
        now - dt.timedelta(days=7),
    )
    backdate_resource(
        "space-different-parameterization",
        CoreResourceKinds.DISCOVERYSPACE,
        now - dt.timedelta(days=6),
    )

    result = runner.invoke(
        ado,
        [
            "get",
            "space",
            "--matching-space-id",
            "reference-space",
            "--details",
            "--no-trunc",
        ],
    )
    assert result.exit_code == 0, result.output

    expected_output = render_dataframe_to_cli_output(
        pd.DataFrame(
            {
                "IDENTIFIER": ["reference-space", "space-different-parameterization"],
                "NAME": ["Reference", "Different"],
                "DESCRIPTION": [None, None],
                "AGE": ["7d0h", "6d0h"],
                "ENTITY_SPACE_MATCH": [SpaceHierarchy.EQUAL_SPACE.value] * 2,
                "MEASUREMENT_SPACE_MATCH": [SpaceHierarchy.EQUAL_SPACE.value] * 2,
                "VERSION_MATCH": [
                    VersionMatch.IDENTICAL.value,
                    VersionMatch.MAJOR_ONLY.value,
                ],
                "PARAMETERIZATION_MATCH": [
                    ParameterizationMatch.IDENTICAL.value,
                    ParameterizationMatch.DIFFERENT.value,
                ],
            }
        ),
        do_not_truncate_columns=True,
    )
    assert expected_output in result.output, (
        f"Expected output:\n{expected_output}\nnot found in:\n{result.output}"
    )
