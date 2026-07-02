# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
import pathlib

import pydantic
import pytest
import sqlalchemy

from orchestrator.core import SampleStoreResource
from orchestrator.core.discoveryspace.resource import DiscoverySpaceProvenanceInfo
from orchestrator.core.metadata import PackageProvenance, ProvenanceInfo
from orchestrator.core.resources import CoreResourceKinds
from orchestrator.core.samplestore.config import SampleStoreConfiguration
from orchestrator.metastore.sqlstore import SQLStore
from orchestrator.utilities.pydantic import do_not_populate_ado_provenance_context


def test_provenance_info_roundtrip_with_ado() -> None:
    """ProvenanceInfo create, dump, and validate preserves ado provenance."""
    ado = PackageProvenance(
        distributionName="ado-core",
        distributionVersion="1.0.0",
    )
    provenance = ProvenanceInfo(ado=ado)
    dump = provenance.model_dump()
    deser = ProvenanceInfo.model_validate(dump)
    assert deser.ado == ado


def test_provenance_info_validator_rejects_bad_plugin_map_values() -> None:
    """Plugin map fields must contain PackageProvenance values."""
    with pytest.raises(pydantic.ValidationError):
        DiscoverySpaceProvenanceInfo(
            ado=None,
            actuators={"replay": "not-a-package-provenance"},  # type: ignore[arg-type]
        )


def test_provenance_info_custom_validator_rejects_non_package_provenance_in_maps() -> (
    None
):
    """The after validator rejects plugin map values that are not PackageProvenance."""
    provenance = DiscoverySpaceProvenanceInfo.model_construct(
        ado=None,
        actuators={"replay": 123},  # type: ignore[dict-item]
        customExperiments={},
    )
    with pytest.raises(ValueError, match="must be PackageProvenance"):
        provenance.validate_provenance_field_values()


def test_discovery_space_provenance_info_with_ado_and_plugins() -> None:
    """Subclass provenance accepts ado alongside plugin maps."""
    ado = PackageProvenance(
        distributionName="ado-core",
        distributionVersion="2.0.0",
    )
    plugin = PackageProvenance(
        distributionName="ado-ray-tune",
        distributionVersion="1.5.0",
    )
    provenance = DiscoverySpaceProvenanceInfo(
        ado=ado,
        actuators={"replay": plugin},
        customExperiments={},
    )
    deser = DiscoverySpaceProvenanceInfo.model_validate(provenance.model_dump())
    assert deser.ado == ado
    assert deser.actuators["replay"] == plugin


def test_provenance_info_populates_ado_on_construction() -> None:
    """ProvenanceInfo() auto-populates ado when population is enabled."""
    from importlib.metadata import version

    provenance = ProvenanceInfo()
    assert provenance.ado is not None
    assert provenance.ado.distributionName == "ado-core"
    assert provenance.ado.distributionVersion == version("ado-core")


def test_provenance_info_preserves_explicit_ado_on_construction() -> None:
    """Explicit ado is not overwritten on construction."""
    existing = PackageProvenance(
        distributionName="ado-core",
        distributionVersion="0.1.0",
    )
    provenance = ProvenanceInfo(ado=existing)
    assert provenance.ado == existing


def test_provenance_info_load_context_leaves_ado_none_when_missing() -> None:
    """Metastore load context preserves missing ado as None."""
    provenance = ProvenanceInfo.model_validate(
        {},
        context=do_not_populate_ado_provenance_context,
    )
    assert provenance.ado is None


def test_sample_store_resource_populates_ado_on_construction(
    sample_store_resource: SampleStoreResource,
) -> None:
    """SampleStoreResource default provenance includes ado-core on construction."""
    resource = SampleStoreResource(
        identifier=sample_store_resource.identifier,
        config=sample_store_resource.config,
    )
    assert resource.provenance.ado is not None
    assert resource.provenance.ado.distributionName == "ado-core"


def test_add_resource_persists_construction_time_ado(
    sql_store: SQLStore,
    sample_store_resource: SampleStoreResource,
) -> None:
    """addResource stores ado provenance set at construction time."""
    resource = SampleStoreResource(
        identifier=sample_store_resource.identifier,
        config=sample_store_resource.config,
    )
    assert resource.provenance.ado is not None
    sql_store.addResource(resource)
    loaded = sql_store.getResource(
        identifier=resource.identifier,
        kind=CoreResourceKinds.SAMPLESTORE,
    )
    assert loaded.provenance.ado is not None
    assert loaded.provenance.ado.distributionName == "ado-core"


def _legacy_sample_store_resource_json(
    identifier: str,
    configuration: SampleStoreConfiguration,
) -> dict:
    """Build stored JSON for a legacy sample store without ado provenance."""
    return {
        "version": "v2",
        "kind": "samplestore",
        "identifier": identifier,
        "config": configuration.model_dump(mode="json"),
        "status": [],
        "metadata": {},
        "provenance": {"ado": None},
    }


def test_get_resource_load_context_preserves_missing_ado(
    sql_store: SQLStore,
    ml_multi_cloud_sample_store_configuration: SampleStoreConfiguration,
) -> None:
    """getResource leaves ado unset for legacy resources without ado provenance."""
    identifier = "legacy-sample-store"
    representation = _legacy_sample_store_resource_json(
        identifier=identifier,
        configuration=ml_multi_cloud_sample_store_configuration,
    )
    with sql_store.engine.begin() as connectable:
        connectable.execute(
            sqlalchemy.text(
                r"INSERT INTO resources"
                r"(identifier, kind, version, data)"
                r"VALUES(:identifier, :kind, :version, :data)"
            ),
            {
                "identifier": identifier,
                "kind": CoreResourceKinds.SAMPLESTORE.value,
                "version": "v2",
                "data": json.dumps(representation),
            },
        )

    loaded = sql_store.getResource(
        identifier=identifier,
        kind=CoreResourceKinds.SAMPLESTORE,
    )
    assert loaded.provenance.ado is None


def test_update_resource_does_not_inject_ado(
    sql_store: SQLStore,
    ml_multi_cloud_sample_store_configuration: SampleStoreConfiguration,
) -> None:
    """updateResource leaves ado unset for legacy resources without ado provenance."""
    identifier = "legacy-sample-store-update"
    representation = _legacy_sample_store_resource_json(
        identifier=identifier,
        configuration=ml_multi_cloud_sample_store_configuration,
    )
    with sql_store.engine.begin() as connectable:
        connectable.execute(
            sqlalchemy.text(
                r"INSERT INTO resources"
                r"(identifier, kind, version, data)"
                r"VALUES(:identifier, :kind, :version, :data)"
            ),
            {
                "identifier": identifier,
                "kind": CoreResourceKinds.SAMPLESTORE.value,
                "version": "v2",
                "data": json.dumps(representation),
            },
        )

    loaded = sql_store.getResource(
        identifier=identifier,
        kind=CoreResourceKinds.SAMPLESTORE,
    )
    assert loaded.provenance.ado is None

    sql_store.updateResource(loaded)
    reloaded = sql_store.getResource(
        identifier=identifier,
        kind=CoreResourceKinds.SAMPLESTORE,
    )
    assert reloaded.provenance.ado is None


def test_legacy_discovery_space_resource_loads_without_ado() -> None:
    """Legacy discovery space fixtures validate with ado None under load context."""
    file = pathlib.Path("tests/resources/space/discoveryspace_resource.json")
    from orchestrator.core import DiscoverySpaceResource

    space = DiscoverySpaceResource.model_validate(
        json.loads(file.read_text()),
        context=do_not_populate_ado_provenance_context,
    )
    assert space.provenance.ado is None
