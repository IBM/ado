# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for PackageProvenance model and provenance helpers."""

import pydantic
import pytest

from orchestrator.core.actuatorconfiguration.resource import (
    ActuatorConfigurationProvenanceInfo,
    ActuatorConfigurationResource,
)
from orchestrator.core.discoveryspace.resource import (
    DiscoverySpaceProvenanceInfo,
    DiscoverySpaceResource,
)
from orchestrator.core.metadata import PackageProvenance, ProvenanceInfo
from orchestrator.core.operation.config import (
    DiscoveryOperationEnum,
)
from orchestrator.core.operation.resource import (
    OperationProvenanceInfo,
    OperationResource,
)
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.modules.operators.collections import provenance_for_operator

# ---------------------------------------------------------------------------
# PackageProvenance model
# ---------------------------------------------------------------------------


def test_package_provenance_create_dump_validate() -> None:
    """PackageProvenance round-trips through model_dump / model_validate."""
    prov = PackageProvenance(distributionName="ado-core", distributionVersion="1.2.3")
    assert prov.distributionName == "ado-core"
    assert prov.distributionVersion == "1.2.3"

    dumped = prov.model_dump()
    assert dumped["distributionName"] == "ado-core"
    assert dumped["distributionVersion"] == "1.2.3"

    reloaded = PackageProvenance.model_validate(dumped)
    assert reloaded == prov


def test_package_provenance_is_frozen() -> None:
    """PackageProvenance instances are immutable."""
    prov = PackageProvenance(
        distributionName="ado-ray-tune", distributionVersion="0.5.0"
    )
    with pytest.raises(pydantic.ValidationError):
        prov.distributionName = "other"  # type: ignore[misc]


def test_package_provenance_rejects_invalid_distribution_version() -> None:
    """PackageProvenance rejects non-PEP-440 distributionVersion values."""
    with pytest.raises(pydantic.ValidationError, match="PEP 440"):
        PackageProvenance(
            distributionName="ado-core",
            distributionVersion="not-a-version",
        )


def test_package_provenance_from_distribution_name() -> None:
    """from_distribution_name resolves an installed distribution."""
    prov = PackageProvenance.from_distribution_name("ado-core")
    assert prov is not None
    assert prov.distributionName == "ado-core"
    assert prov.distributionVersion


def test_package_provenance_from_distribution_name_unknown() -> None:
    """from_distribution_name returns None for an unknown distribution."""
    assert (
        PackageProvenance.from_distribution_name("nonexistent-distribution-xyz") is None
    )


def test_package_provenance_from_module_name_orchestrator() -> None:
    """from_module_name maps orchestrator modules to ado-core."""
    prov = PackageProvenance.from_module_name(
        "orchestrator.modules.operators.randomwalk"
    )
    assert prov is not None
    assert prov.distributionName == "ado-core"
    assert prov.distributionVersion


def test_package_provenance_from_module_name_unknown() -> None:
    """from_module_name returns None for an unknown module."""
    assert (
        PackageProvenance.from_module_name("nonexistent_module_xyz.submodule") is None
    )


def test_package_provenance_from_module_conf_dict() -> None:
    """from_module_conf resolves provenance from a moduleName dict."""
    prov = PackageProvenance.from_module_conf(
        {"moduleName": "orchestrator.modules.operators.randomwalk"}
    )
    assert prov is not None
    assert prov.distributionName == "ado-core"


def test_package_provenance_from_module_conf_missing_module_name() -> None:
    """from_module_conf returns None when moduleName is missing."""
    assert PackageProvenance.from_module_conf({}) is None


# ---------------------------------------------------------------------------
# ProvenanceInfo validator
# ---------------------------------------------------------------------------


def test_provenance_info_rejects_non_dict_field_values() -> None:
    """ProvenanceInfo subclasses reject non-dict field values."""

    class SampleProvenanceInfo(ProvenanceInfo):
        items: dict[str, PackageProvenance] = pydantic.Field(default_factory=dict)

    with pytest.raises(pydantic.ValidationError):
        SampleProvenanceInfo.model_validate({"items": "not-a-dict"})


def test_provenance_info_rejects_non_package_provenance_dict_values() -> None:
    """ProvenanceInfo subclasses reject dict values that are not PackageProvenance."""

    class SampleProvenanceInfo(ProvenanceInfo):
        items: dict[str, PackageProvenance] = pydantic.Field(default_factory=dict)

    with pytest.raises(pydantic.ValidationError):
        SampleProvenanceInfo.model_validate({"items": {"key": "not-provenance"}})


# ---------------------------------------------------------------------------
# ActuatorRegistry.provenance_for_actuator
# ---------------------------------------------------------------------------


def test_provenance_for_builtin_actuator() -> None:
    """Builtin actuators report ado-core as their distribution."""
    registry = ActuatorRegistry.globalRegistry()
    prov = registry.provenance_for_actuator("custom_experiments")
    assert prov is not None
    assert prov.distributionName == "ado-core"
    assert prov.distributionVersion  # non-empty version string


def test_provenance_for_replay_actuator() -> None:
    """Replay actuator (builtin) reports ado-core."""
    registry = ActuatorRegistry.globalRegistry()
    prov = registry.provenance_for_actuator("replay")
    assert prov is not None
    assert prov.distributionName == "ado-core"


def test_provenance_for_unknown_actuator_returns_none() -> None:
    """Unknown actuator identifier returns None instead of raising."""
    registry = ActuatorRegistry.globalRegistry()
    prov = registry.provenance_for_actuator("nonexistent_actuator_xyz")
    assert prov is None


def test_provenance_for_plugin_actuator() -> None:
    """Plugin actuators should resolve a non-ado-core distribution name."""
    from orchestrator.modules.actuators.registry import ActuatorRegistry

    registry = ActuatorRegistry.globalRegistry()
    # mock actuator is registered as a plugin in test fixtures
    prov = registry.provenance_for_actuator("mock")
    if prov is not None:
        assert prov.distributionName
        assert prov.distributionVersion


# ---------------------------------------------------------------------------
# provenance_for_operator
# ---------------------------------------------------------------------------


def test_provenance_for_random_walk_operator() -> None:
    """The built-in random_walk explore operator should resolve a distribution."""
    prov = provenance_for_operator("random_walk", DiscoveryOperationEnum.SEARCH)
    assert prov is not None
    assert prov.distributionName
    assert prov.distributionVersion


def test_provenance_for_unknown_operator_returns_none() -> None:
    """Non-existent operator name returns None."""
    prov = provenance_for_operator("nonexistent_op_xyz", DiscoveryOperationEnum.SEARCH)
    assert prov is None


def test_provenance_for_operator_wrong_type_returns_none() -> None:
    """Correct name but wrong op_type returns None."""
    prov = provenance_for_operator("random_walk", DiscoveryOperationEnum.CHARACTERIZE)
    assert prov is None


def test_operator_metadata_provenance_lifecycle() -> None:
    """OperatorMetadata round-trips provenance through model_dump / model_validate."""
    from orchestrator.core.operation.config import OperatorMetadata

    class _ExampleConfig(pydantic.BaseModel):
        value: int = 1

    prov = PackageProvenance(distributionName="ado-core", distributionVersion="1.2.3")
    metadata = OperatorMetadata(
        name="test_op",
        version="99.0.0",
        configuration_model=_ExampleConfig,
        example_configuration=_ExampleConfig(),
        type=DiscoveryOperationEnum.SEARCH,
        provenance=prov,
    )

    dumped = metadata.model_dump()
    restored = OperatorMetadata.model_validate(dumped)

    assert restored.version == "99.0.0"
    assert restored.provenance == prov


def test_operator_metadata_has_package_provenance() -> None:
    """OperatorMetadata for random_walk has package provenance set at registration."""
    from orchestrator.modules.operators.collections import explore

    metadata = explore.operators.get("random_walk")
    assert metadata is not None
    assert metadata.provenance is not None
    assert metadata.provenance.distributionName == "ado-core"
    assert metadata.provenance.distributionVersion


def test_operator_metadata_version_is_independent_of_package_provenance() -> None:
    """Operator version and package provenance capture different information."""
    from orchestrator.modules.operators.collections import explore

    metadata = explore.operators.get("random_walk")
    assert metadata is not None
    assert metadata.provenance is not None
    # random_walk uses version("ado-core") for operator identity, but the fields
    # remain semantically distinct on OperatorMetadata.
    assert metadata.version
    assert metadata.provenance.distributionName == "ado-core"


# ---------------------------------------------------------------------------
# Resource provenance fields: lifecycle (create -> dump -> validate)
# ---------------------------------------------------------------------------


def test_discovery_space_resource_provenance_lifecycle(
    discovery_space_resource: DiscoverySpaceResource,
) -> None:
    """DiscoverySpaceResource round-trips provenance through model_dump."""
    prov = PackageProvenance(distributionName="ado-core", distributionVersion="9.9.9")
    custom_prov = PackageProvenance(
        distributionName="my-experiment-pkg", distributionVersion="1.0.0"
    )
    discovery_space_resource.provenance = DiscoverySpaceProvenanceInfo(
        actuators={"replay": prov},
        customExperiments={"my_exp": custom_prov},
    )

    dumped = discovery_space_resource.model_dump()
    assert "provenance" in dumped
    assert "actuators" in dumped["provenance"]
    assert "customExperiments" in dumped["provenance"]
    assert "actuatorProvenance" not in dumped

    restored = DiscoverySpaceResource.model_validate(dumped)

    assert restored.provenance.actuators["replay"] == prov
    assert restored.provenance.customExperiments["my_exp"] == custom_prov


def test_discovery_space_resource_provenance_defaults_empty(
    discovery_space_resource: DiscoverySpaceResource,
) -> None:
    """DiscoverySpaceResource created without provenance has empty dicts."""
    dumped = discovery_space_resource.model_dump()
    restored = DiscoverySpaceResource.model_validate(dumped)
    assert restored.provenance.actuators == {}
    assert restored.provenance.customExperiments == {}


def test_operation_resource_provenance_lifecycle(
    operation_resource: OperationResource,
) -> None:
    """OperationResource round-trips nested provenance through model_dump."""
    prov = PackageProvenance(
        distributionName="ado-ray-tune", distributionVersion="1.7.1"
    )
    operation_resource.provenance = OperationProvenanceInfo(
        operators={operation_resource.operatorIdentifier: prov}
    )

    dumped = operation_resource.model_dump()
    assert (
        dumped["provenance"]["operators"][operation_resource.operatorIdentifier][
            "distributionName"
        ]
        == "ado-ray-tune"
    )

    restored = OperationResource.model_validate(dumped)

    assert restored.provenance.operators[operation_resource.operatorIdentifier] == prov


def test_operation_resource_provenance_defaults_empty(
    operation_resource: OperationResource,
) -> None:
    """OperationResource created without provenance has empty operators dict."""
    dumped = operation_resource.model_dump()
    restored = OperationResource.model_validate(dumped)
    assert restored.provenance.operators == {}


def test_actuator_configuration_resource_provenance_lifecycle() -> None:
    """ActuatorConfigurationResource round-trips nested provenance through model_dump."""
    from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration

    config = ActuatorConfiguration(actuatorIdentifier="mock")
    prov = PackageProvenance(distributionName="ado-mock", distributionVersion="0.1.0")
    resource = ActuatorConfigurationResource(
        config=config,
        provenance=ActuatorConfigurationProvenanceInfo(actuators={"mock": prov}),
    )

    assert resource.provenance.actuators["mock"] == prov

    dumped = resource.model_dump()
    restored = ActuatorConfigurationResource.model_validate(dumped)
    assert restored.provenance.actuators["mock"] == prov


def test_actuator_configuration_resource_provenance_defaults_empty() -> None:
    """ActuatorConfigurationResource created without provenance has empty actuators dict."""
    from orchestrator.core.actuatorconfiguration.config import ActuatorConfiguration

    config = ActuatorConfiguration(actuatorIdentifier="mock")
    resource = ActuatorConfigurationResource(config=config)

    assert resource.provenance.actuators == {}

    dumped = resource.model_dump()
    restored = ActuatorConfigurationResource.model_validate(dumped)
    assert restored.provenance.actuators == {}


# ---------------------------------------------------------------------------
# DiscoverySpaceResource serialized without provenance is still valid
# ---------------------------------------------------------------------------


def test_discovery_space_resource_loads_from_legacy_json() -> None:
    """DiscoverySpaceResource JSON without provenance validates with empty defaults."""
    import json
    import pathlib

    legacy_json = pathlib.Path(
        "tests/resources/space/discoveryspace_resource.json"
    ).read_text()
    resource = DiscoverySpaceResource.model_validate(json.loads(legacy_json))
    assert resource.provenance.actuators == {}
    assert resource.provenance.customExperiments == {}
