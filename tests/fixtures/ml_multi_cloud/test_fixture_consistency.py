# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
from collections.abc import Callable

from orchestrator.core import ActuatorConfigurationResource
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.resources import ADOResource, CoreResourceKinds
from orchestrator.core.samplestore.sql import SQLSampleStore


def test_create_ml_multicloud_sample_store(
    get_single_resource_by_identifier: Callable[
        [str, CoreResourceKinds], ADOResource | None
    ],
    ml_multi_cloud_sample_store: SQLSampleStore,
) -> None:
    assert ml_multi_cloud_sample_store.identifier is not None
    sample_store_resource = get_single_resource_by_identifier(
        identifier=ml_multi_cloud_sample_store.identifier,
        kind=CoreResourceKinds.SAMPLESTORE,
    )
    assert (
        sample_store_resource is not None
    ), "The SQL Store couldn't retrieve the sample store"


def test_create_ml_multicloud_space(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    assert ml_multi_cloud_space is not None
    assert ml_multi_cloud_space.uri.startswith("space-")
    assert ml_multi_cloud_space.uri.endswith(
        str(ml_multi_cloud_space.sample_store.identifier)
    )


def test_create_ml_multicloud_actuatorconfiguration(
    get_single_resource_by_identifier: Callable[
        [str, CoreResourceKinds], ADOResource | None
    ],
    ml_multi_cloud_correct_actuatorconfiguration: ActuatorConfigurationResource,
) -> None:
    assert ml_multi_cloud_correct_actuatorconfiguration is not None
    actuatorconfiguration_resource = get_single_resource_by_identifier(
        identifier=ml_multi_cloud_correct_actuatorconfiguration.identifier,
        kind=CoreResourceKinds.ACTUATORCONFIGURATION,
    )
    assert (
        actuatorconfiguration_resource is not None
    ), "The SQL Store couldn't retrieve the actuatorconfiguration"
