# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for operator algorithm versioning."""

import pydantic
import pytest

from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    GenericOperatorParameters,
    OperatorMetadata,
    OperatorReference,
)
from ado.modules.operators.collections import (
    explore,
    resolve_operator_reference,
)
from ado.modules.operators.errors import OperatorVersionMismatchError


class _ExampleConfig(GenericOperatorParameters):
    value: int = 1


def test_operator_metadata_reference() -> None:
    """Test the OperatorReference produced by OperatorMetadata"""

    meta = OperatorMetadata(
        name="my_op",
        version="2.0.0",
        configuration_model=_ExampleConfig,
        example_configuration=_ExampleConfig(),
        type=DiscoveryOperationEnum.EXPLORE,
    )
    ref = meta.reference
    assert ref.operatorName == "my_op"
    assert ref.operationType == DiscoveryOperationEnum.EXPLORE
    assert ref.operatorVersion == "2.0.0"
    assert meta.operatorIdentifier == "my_op@2.0.0"


def test_operator_metadata_rejects_non_semver_version() -> None:
    """OperatorMetadata rejects PEP 440 dev version strings."""
    with pytest.raises(pydantic.ValidationError, match="SemVer"):
        OperatorMetadata(
            name="my_op",
            version="1.0.2.dev17+5e50632",
            configuration_model=_ExampleConfig,
            example_configuration=_ExampleConfig(),
            type=DiscoveryOperationEnum.EXPLORE,
        )


def test_operator_reference_identifier_without_pinned_version() -> None:
    """OperatorReference without operatorVersion resolves from registry."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.EXPLORE,
    )
    assert ref.operatorIdentifier == explore.operators["random_walk"].operatorIdentifier


def test_resolve_operator_reference_pins_when_version_omitted() -> None:
    """Omitted operatorVersion is pinned from the registry."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.EXPLORE,
    )
    resolved = resolve_operator_reference(ref)
    assert resolved.operatorVersion == explore.operators["random_walk"].version


def test_resolve_operator_reference_exact_match() -> None:
    """Explicit operatorVersion matching registry resolves successfully."""

    random_walk_version = explore.operators["random_walk"].version
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorVersion=random_walk_version,
    )
    # This should not raise an error or change operatorVersion field value
    resolved = resolve_operator_reference(ref)
    assert resolved.operatorVersion == random_walk_version


def test_resolve_operator_reference_mismatch_raises() -> None:
    """Explicit operatorVersion mismatch raises OperatorVersionMismatchError."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.EXPLORE,
        operatorVersion="1.0.0",
    )
    with pytest.raises(
        OperatorVersionMismatchError, match="Algorithm version mismatch"
    ):
        resolve_operator_reference(ref)


def test_discovery_operation_configuration_pins_operator_version() -> None:
    """DiscoveryOperationConfiguration validation sets operatorVersion."""
    config = DiscoveryOperationConfiguration(
        module=OperatorReference(
            operatorName="random_walk",
            operationType=DiscoveryOperationEnum.EXPLORE,
        ),
        parameters={"numberEntities": 1, "batchSize": 1},
    )
    assert isinstance(config.module, OperatorReference)
    assert config.module.operatorVersion == explore.operators["random_walk"].version


def test_discovery_operation_resource_configuration_round_trip() -> None:
    """Pinned operatorVersion round-trips through model_dump / model_validate."""
    resource_config = DiscoveryOperationResourceConfiguration(
        spaces=["space-test123"],
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName="random_walk",
                operationType=DiscoveryOperationEnum.EXPLORE,
            ),
            parameters={"numberEntities": 1, "batchSize": 1},
        ),
    )
    random_walk_version = explore.operators["random_walk"].version
    dumped = resource_config.model_dump()
    assert dumped["operation"]["module"]["operatorVersion"] == random_walk_version

    restored = DiscoveryOperationResourceConfiguration.model_validate(dumped)
    assert isinstance(restored.operation.module, OperatorReference)
    assert restored.operation.module.operatorVersion == random_walk_version
