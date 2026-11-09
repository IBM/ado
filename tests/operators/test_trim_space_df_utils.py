# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration tests for the DataFrame helpers in trim.samplers.no_priors_utils."""

import logging

import pytest

from ado.core.discoveryspace.space import DiscoverySpace

pytest.importorskip("trim")

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
    assert caplog.records == []


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
