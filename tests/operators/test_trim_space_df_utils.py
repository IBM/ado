# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for the DataFrame helpers in trim.samplers.no_priors_utils."""

import logging
import pathlib
from collections.abc import Callable

import pytest
import yaml
from typer.testing import CliRunner

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: nocover
    from yaml import SafeLoader  # type: ignore[assignment]

from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext

pytest.importorskip("trim")
pytest.importorskip("trim_custom_experiments")

import trim_custom_experiments.experiments  # noqa: E402, F401
from trim.samplers.no_priors_utils import (  # noqa: E402
    get_df_all_entities_no_measurements,
    get_df_at_least_one_measured_value,
    get_list_of_entities_from_df_and_space,
    get_source_and_target,
)

TARGET = "wallClockRuntime"
CONSTITUTIVE_PROPERTIES = ["provider", "cpu_family", "vcpu_size", "nodes"]


def test_get_df_all_entities_no_measurements(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should return one row per point of the entity space."""
    df = get_df_all_entities_no_measurements(ml_multi_cloud_space)

    assert list(df.columns) == ["identifier", *CONSTITUTIVE_PROPERTIES]
    assert len(df) == ml_multi_cloud_space.entitySpace.size
    assert df["identifier"].is_unique


def test_get_df_at_least_one_measured_value_names_target_column(
    ml_multi_cloud_space: DiscoverySpace,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Should expose the mean-aggregated target under its bare identifier."""
    with caplog.at_level(logging.WARNING, logger="trim.samplers.no_priors_utils"):
        df = get_df_at_least_one_measured_value(
            ml_multi_cloud_space, [TARGET], add_measurement_id=True
        )

    assert list(df.columns) == ["identifier", *CONSTITUTIVE_PROPERTIES, TARGET]
    assert not df.empty
    assert df[TARGET].notna().all()
    assert [
        r for r in caplog.records if r.name == "trim.samplers.no_priors_utils"
    ] == []


def test_get_df_at_least_one_measured_value_from_space_identifier(
    tmp_path: pathlib.Path,
    valid_ado_project_context: ProjectContext,
    create_active_ado_context: Callable[
        [CliRunner, pathlib.Path, ProjectContext], None
    ],
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should resolve a space identifier through the active context."""
    create_active_ado_context(
        runner=CliRunner(), path=tmp_path, project_context=valid_ado_project_context
    )

    from_identifier = get_df_at_least_one_measured_value(
        ml_multi_cloud_space.uri, [TARGET], add_measurement_id=True
    )
    from_space = get_df_at_least_one_measured_value(
        ml_multi_cloud_space, [TARGET], add_measurement_id=True
    )

    assert not from_identifier.empty
    assert from_identifier.equals(from_space)


def test_get_df_at_least_one_measured_value_unknown_target(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should omit a target that has no measurements in the space."""
    df = get_df_at_least_one_measured_value(ml_multi_cloud_space, ["not_measured"])

    assert list(df.columns) == CONSTITUTIVE_PROPERTIES


def test_get_source_and_target_partitions_space(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should split the entity space into measured and unmeasured entities."""
    source_df, target_df = get_source_and_target(ml_multi_cloud_space, TARGET)

    assert TARGET in source_df.columns
    assert TARGET not in target_df.columns
    assert source_df[TARGET].notna().all()
    assert set(source_df["identifier"]).isdisjoint(target_df["identifier"])
    assert len(source_df) + len(target_df) == ml_multi_cloud_space.entitySpace.size


def test_get_source_and_target_never_measured_target(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should return an empty source and the full space as target."""
    source_df, target_df = get_source_and_target(ml_multi_cloud_space, "not_measured")

    assert source_df.empty
    assert len(target_df) == ml_multi_cloud_space.entitySpace.size


def test_get_list_of_entities_from_df_and_space(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should convert unmeasured rows into entities of the space."""
    _, target_df = get_source_and_target(ml_multi_cloud_space, TARGET)

    entities = get_list_of_entities_from_df_and_space(
        df=target_df, space=ml_multi_cloud_space
    )

    assert [e.identifier for e in entities] == list(target_df["identifier"])
    assert all(ml_multi_cloud_space.entitySpace.isEntityInSpace(e) for e in entities)
    assert {e.generatorid for e in entities} == {"no_priors_characterization"}


@pytest.fixture
def unmeasured_space(
    empty_sample_store: SQLSampleStore,
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
) -> DiscoverySpace:
    """Discovery space on an empty sample store."""
    configuration = DiscoverySpaceConfiguration.model_validate(
        yaml.load(
            pathlib.Path("tests/resources/trim/space_minimal.yaml").read_text(),
            Loader=SafeLoader,
        )
    )
    return create_space(configuration, empty_sample_store.identifier)


def test_get_source_and_target_unmeasured_space(
    unmeasured_space: DiscoverySpace,
) -> None:
    """Should return an empty source and every entity as target."""
    source_df, target_df = get_source_and_target(unmeasured_space, "pressure")

    assert source_df.empty
    assert len(target_df) == unmeasured_space.entitySpace.size
    assert list(target_df.columns) == [
        "identifier",
        *[cp.identifier for cp in unmeasured_space.entitySpace.constitutiveProperties],
    ]
