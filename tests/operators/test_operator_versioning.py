# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for operator algorithm versioning."""

import pydantic
import pytest

from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    OperatorMetadata,
    OperatorReference,
)
from orchestrator.modules.operators.collections import (
    explore,
    resolve_operator_reference,
)
from orchestrator.modules.operators.errors import OperatorVersionMismatchError


class _ExampleConfig(pydantic.BaseModel):
    value: int = 1


def test_operator_metadata_identifier_uses_at_separator() -> None:
    """OperatorMetadata.operatorIdentifier returns '{name}@{version}'."""
    meta = OperatorMetadata(
        name="my_op",
        version="2.0.0",
        configuration_model=_ExampleConfig,
        example_configuration=_ExampleConfig(),
        type=DiscoveryOperationEnum.SEARCH,
    )
    assert meta.operatorIdentifier == "my_op@0.1.0"


def test_operator_metadata_reference_includes_version() -> None:
    """OperatorMetadata.reference carries the algorithm version."""
    meta = OperatorMetadata(
        name="my_op",
        version="2.0.0",
        configuration_model=_ExampleConfig,
        example_configuration=_ExampleConfig(),
        type=DiscoveryOperationEnum.SEARCH,
    )
    ref = meta.reference
    assert ref.operatorName == "my_op"
    assert ref.operationType == DiscoveryOperationEnum.SEARCH
    assert ref.operatorVersion == "2.0.0"


def test_operator_metadata_rejects_non_semver_version() -> None:
    """OperatorMetadata rejects PEP 440 dev version strings."""
    with pytest.raises(pydantic.ValidationError, match="SemVer"):
        OperatorMetadata(
            name="my_op",
            version="1.0.2.dev17+5e50632",
            configuration_model=_ExampleConfig,
            example_configuration=_ExampleConfig(),
            type=DiscoveryOperationEnum.SEARCH,
        )


def test_operator_reference_identifier_with_pinned_version() -> None:
    """OperatorReference.operatorIdentifier uses pinned operatorVersion."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.SEARCH,
        operatorVersion="2.0.0",
    )
    assert ref.operatorIdentifier == "random_walk@2.0.0"


def test_operator_reference_identifier_without_pinned_version() -> None:
    """OperatorReference without operatorVersion resolves from registry."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.SEARCH,
    )
    assert ref.operatorIdentifier == explore.operators["random_walk"].operatorIdentifier


def test_resolve_operator_reference_pins_when_version_omitted() -> None:
    """Omitted operatorVersion is pinned from the registry."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.SEARCH,
    )
    resolved = resolve_operator_reference(ref)
    assert resolved.operatorVersion == explore.operators["random_walk"].version
    assert resolved.operatorVersion == "2.0.0"


def test_resolve_operator_reference_exact_match() -> None:
    """Explicit operatorVersion matching registry resolves successfully."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.SEARCH,
        operatorVersion="2.0.0",
    )
    resolved = resolve_operator_reference(ref)
    assert resolved.operatorVersion == "2.0.0"


def test_resolve_operator_reference_mismatch_raises() -> None:
    """Explicit operatorVersion mismatch raises OperatorVersionMismatchError."""
    ref = OperatorReference(
        operatorName="random_walk",
        operationType=DiscoveryOperationEnum.SEARCH,
        operatorVersion="1.0.0",
    )
    with pytest.raises(
        OperatorVersionMismatchError, match="Algorithm version mismatch"
    ):
        resolve_operator_reference(ref)


def test_discovery_operation_configuration_pins_operator_version() -> None:
    """DiscoveryOperationConfiguration validation pins operatorVersion."""
    config = DiscoveryOperationConfiguration(
        module=OperatorReference(
            operatorName="random_walk",
            operationType=DiscoveryOperationEnum.SEARCH,
        ),
        parameters={"numberEntities": 1, "batchSize": 1},
    )
    assert isinstance(config.module, OperatorReference)
    assert config.module.operatorVersion == "2.0.0"


def test_discovery_operation_resource_configuration_round_trip() -> None:
    """Pinned operatorVersion round-trips through model_dump / model_validate."""
    resource_config = DiscoveryOperationResourceConfiguration(
        spaces=["space-test123"],
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName="random_walk",
                operationType=DiscoveryOperationEnum.SEARCH,
            ),
            parameters={"numberEntities": 1, "batchSize": 1},
        ),
    )
    dumped = resource_config.model_dump()
    assert dumped["operation"]["module"]["operatorVersion"] == "2.0.0"

    restored = DiscoveryOperationResourceConfiguration.model_validate(dumped)
    assert isinstance(restored.operation.module, OperatorReference)
    assert restored.operation.module.operatorVersion == "2.0.0"
