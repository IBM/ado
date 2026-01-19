# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pytest

from orchestrator.core.resources import CoreResourceKinds


@pytest.fixture(
    params=[
        pytest.param(
            (CoreResourceKinds.SAMPLESTORE, "random_sample_store_resource_from_file"),
            id=CoreResourceKinds.SAMPLESTORE.value,
        ),
        pytest.param(
            (CoreResourceKinds.DISCOVERYSPACE, "random_space_resource_from_file"),
            id=CoreResourceKinds.DISCOVERYSPACE.value,
        ),
        pytest.param(
            (CoreResourceKinds.OPERATION, "random_operation_resource_from_file"),
            id=CoreResourceKinds.OPERATION.value,
        ),
    ]
)
def resource_generator_from_file(
    request: pytest.FixtureRequest,
) -> tuple[CoreResourceKinds, str]:
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (CoreResourceKinds.SAMPLESTORE, "random_sample_store_resource_from_db"),
            id=CoreResourceKinds.SAMPLESTORE.value,
        ),
        pytest.param(
            (CoreResourceKinds.DISCOVERYSPACE, "random_space_resource_from_db"),
            id=CoreResourceKinds.DISCOVERYSPACE.value,
        ),
        pytest.param(
            (CoreResourceKinds.OPERATION, "random_operation_resource_from_db"),
            id=CoreResourceKinds.OPERATION.value,
        ),
    ]
)
def resource_generator_from_db(
    request: pytest.FixtureRequest,
) -> tuple[CoreResourceKinds, str]:
    return request.param
