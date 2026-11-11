# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for the DataFrame helpers in trim.samplers.no_priors_utils.

These tests use the sample store, discovery space and project context fixtures
of the top-level ``tests/`` package (``ml_multi_cloud_space``, ``create_space``,
``empty_sample_store``), like the other TRIM integration tests in
``tests/operators/``.
"""

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
from ado.schema.observed_property import ObservedProperty, ObservedPropertyValue
from ado.schema.property import AbstractPropertyDescriptor
from ado.schema.reference import ExperimentReference
from ado.schema.result import ValidMeasurementResult

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
    for row in df.to_dict(orient="records"):
        point = {cp: row[cp] for cp in CONSTITUTIVE_PROPERTIES}
        expected = ml_multi_cloud_space.entitySpace.entity_for_point(point).identifier
        assert row["identifier"] == expected


def test_get_df_at_least_one_measured_value_names_target_column(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should expose the mean-aggregated target under its bare identifier."""
    df = get_df_at_least_one_measured_value(
        ml_multi_cloud_space, [TARGET], add_measurement_id=True
    )

    assert list(df.columns) == ["identifier", *CONSTITUTIVE_PROPERTIES, TARGET]
    assert not df.empty


def test_get_df_at_least_one_measured_value_drops_entities_missing_target(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should drop entities measured only for other properties of the experiment."""
    _, unmeasured = get_source_and_target(ml_multi_cloud_space, TARGET)
    point = {cp: unmeasured.iloc[0][cp] for cp in CONSTITUTIVE_PROPERTIES}
    entity = ml_multi_cloud_space.entity_for_point(point)
    store = ml_multi_cloud_space.sample_store
    store.addEntities([entity])
    store.add_measurement_results(
        results=[
            ValidMeasurementResult(
                entityIdentifier=entity.identifier,
                measurements=[
                    ObservedPropertyValue(
                        value="ok",
                        property=ObservedProperty(
                            targetProperty=AbstractPropertyDescriptor(
                                identifier="status"
                            ),
                            experimentReference=ExperimentReference(
                                experimentIdentifier="benchmark_performance",
                                actuatorIdentifier="replay",
                            ),
                        ),
                    )
                ],
            )
        ],
        skip_relationship_to_request=True,
    )

    with_target = get_df_at_least_one_measured_value(
        ml_multi_cloud_space, [TARGET], add_measurement_id=True
    )
    with_status = get_df_at_least_one_measured_value(
        ml_multi_cloud_space, ["status"], add_measurement_id=True
    )

    assert entity.identifier not in set(with_target["identifier"])
    assert with_target[TARGET].notna().all()
    assert entity.identifier in set(with_status["identifier"])


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

    df_from_identifier = get_df_at_least_one_measured_value(
        ml_multi_cloud_space.uri, [TARGET], add_measurement_id=True
    )
    df_from_space = get_df_at_least_one_measured_value(
        ml_multi_cloud_space, [TARGET], add_measurement_id=True
    )

    assert not df_from_identifier.empty
    assert df_from_identifier.equals(df_from_space)


def test_get_df_at_least_one_measured_value_sees_new_measurements(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """Should reflect a measurement stored after the entity's measurements were cached."""
    df_before = get_df_at_least_one_measured_value(
        ml_multi_cloud_space, [TARGET], add_measurement_id=True
    )
    measured_entity_id = df_before["identifier"].iloc[0]
    value_before = df_before.loc[
        df_before["identifier"] == measured_entity_id, TARGET
    ].iloc[0]
    new_measured_value = value_before + 1000.0

    ml_multi_cloud_space.sample_store.add_measurement_results(
        results=[
            ValidMeasurementResult(
                entityIdentifier=measured_entity_id,
                measurements=[
                    ObservedPropertyValue(
                        value=new_measured_value,
                        property=ObservedProperty(
                            targetProperty=AbstractPropertyDescriptor(
                                identifier=TARGET
                            ),
                            experimentReference=ExperimentReference(
                                experimentIdentifier="benchmark_performance",
                                actuatorIdentifier="replay",
                            ),
                        ),
                    )
                ],
            )
        ],
        skip_relationship_to_request=True,
    )

    df_after = get_df_at_least_one_measured_value(
        ml_multi_cloud_space, [TARGET], add_measurement_id=True
    )

    assert len(df_after) == len(df_before)
    value_after = df_after.loc[
        df_after["identifier"] == measured_entity_id, TARGET
    ].iloc[0]
    assert value_after != pytest.approx(value_before)


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
            (
                pathlib.Path(__file__).parents[1] / "resources/trim/space_minimal.yaml"
            ).read_text(),
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
