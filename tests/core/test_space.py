# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import datetime
import re
from collections.abc import Callable

import pytest
import yaml

import orchestrator.core
from orchestrator.core import DiscoverySpaceResource
from orchestrator.core.discoveryspace.config import DiscoverySpaceConfiguration
from orchestrator.core.discoveryspace.space import (
    DiscoverySpace,
    SpaceInconsistencyError,
)
from orchestrator.core.samplestore.base import ActiveSampleStore
from orchestrator.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.actuators.registry import ActuatorRegistry
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.measurementspace import (
    MeasurementSpace,
    MeasurementSpaceConfiguration,
)
from orchestrator.schema.reference import (
    ExperimentReference,
)


def test_discovery_space(
    pfas_space: DiscoverySpace, pfas_space_configuration: DiscoverySpaceConfiguration
) -> None:
    assert not pfas_space.sample_store.isPassive

    # Since discovery_space was created using a non-builtin sample store we can't
    # compare the DiscoverySpaceConfiguration object returned from the discovery_space
    # with the one used to init it (the sample store was changed during init)

    assert pfas_space.config.entitySpace == pfas_space_configuration.entitySpace

    assert (
        pfas_space.measurementSpace.experimentReferences
        == pfas_space_configuration.experiments
    )


def test_space_describe(pfas_space: DiscoverySpace) -> None:
    from rich.console import Console

    Console().print(pfas_space)

    assert True


def test_discovery_space_with_parameterized_experiments(
    parameterized_references: list[ExperimentReference],
    valid_ado_project_context: ProjectContext,
    global_registry: ActuatorRegistry,
    create_sample_store: Callable[[SampleStoreConfiguration], ActiveSampleStore],
) -> None:

    from orchestrator.core.samplestore.config import (
        SampleStoreConfiguration,
    )

    ms = MeasurementSpace.measurementSpaceFromExperimentReferences(
        experimentReferences=parameterized_references
    )
    sample_store = create_sample_store(
        SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="orchestrator.core.samplestore.sql",
                ),
            )
        )
    )

    # create a compatible entityspace
    es = ms.compatibleEntitySpace()  # type: EntitySpaceRepresentation

    # Try creating a space
    DiscoverySpace(
        project_context=valid_ado_project_context,
        identifier="test_space",
        sample_store=sample_store,
        entitySpace=es,
        measurementSpace=ms,
    )

    # Test you can't create a space that has both
    # - a custom parameterization for an optional property of an experiment
    # - has moved the optional property of an experiment into the entityspace

    cps = es.constitutiveProperties

    # Add an optional cp to the entity space
    propToAdd = ms.experiments[0].optionalProperties[0]

    # Check this parameter has in fact been parameterized
    assert propToAdd.identifier in [
        v.property.identifier for v in parameterized_references[0].parameterization
    ]

    # Create entityspace
    es = EntitySpaceRepresentation(constitutiveProperties=[*cps, propToAdd])

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Identified an entity space dimension, {propToAdd}, "
            "that also has a custom parameterization in the measurement space. "
            "It is inconsistent for a property to have a custom parameterization in the measurement space "
            "and also be a dimension of the entityspace"
        ),
    ):
        ms.checkEntitySpaceCompatible(es)

    with pytest.raises(SpaceInconsistencyError):
        DiscoverySpace(
            project_context=valid_ado_project_context,
            identifier="test_space",
            sample_store=sample_store,
            entitySpace=es,
            measurementSpace=ms,
        )


# TODO: This test require explicit entity-space and matching source
# Currently we have an explicit space and a non-matching source of molecule data
# def test_matching_source_entities(discovery_space: orchestrator.model.space.DiscoverySpace):
#
#     table = discovery_space.matchingEntitiesTable()
#     assert len(discovery_space.matchingEntities()) == 101
#     assert table.size != 0
#     assert "identifier" in table.columns
@pytest.mark.xfail
def test_discoveryspace_with_replay_actuator_and_references_rich_print(
    discovery_space_resource: DiscoverySpaceResource,
) -> None:

    ### This is expected to fail
    # - One experiment uses "replay" actuator which means it is external and defined by a samplestore
    # - The experiment is given in the discoveryspace resource config via a reference (instead of as a full Experiment)
    # Because of the second the pretty goes to the registry to get the experiment definition
    # Because of the first tbere is no definition (the discoveryspace resource does not cause the samplestore to loaded - this requires creating the space)
    # I'm not sure if there is a way to solve this

    from rich.console import Console

    assert hasattr(discovery_space_resource, "__rich__")
    Console().print(discovery_space_resource)


def test_discoveryspace_with_normal_actuator_rich_print(
    discovery_space_resource_no_replay: DiscoverySpaceResource,
) -> None:

    from rich.console import Console

    assert hasattr(discovery_space_resource_no_replay, "__rich__")
    Console().print(discovery_space_resource_no_replay)


def test_discovery_space_resource(
    discovery_space_resource: DiscoverySpaceResource,
) -> None:

    assert discovery_space_resource.identifier is not None
    assert discovery_space_resource.identifier == "test_space"

    assert (
        discovery_space_resource.kind
        == orchestrator.core.resources.CoreResourceKinds.DISCOVERYSPACE
    )

    assert discovery_space_resource.created < datetime.datetime.now(
        datetime.timezone.utc
    )
    assert isinstance(discovery_space_resource.metadata, dict)
    assert discovery_space_resource.config is not None
    assert isinstance(discovery_space_resource.config, DiscoverySpaceConfiguration)
    assert discovery_space_resource.config.sampleStoreIdentifier is not None
    assert len(discovery_space_resource.status) == 1


def test_discovery_space_config_file_valid(
    valid_discovery_space_config_file: str,
) -> None:
    import pathlib

    valid_discovery_space_config_file = pathlib.Path(valid_discovery_space_config_file)
    orchestrator.core.discoveryspace.config.DiscoverySpaceConfiguration.model_validate(
        yaml.safe_load(valid_discovery_space_config_file.read_text())
    )


def test_discovery_space_config_experiment_field_conversion_parameterized(
    measurement_space_from_multiple_parameterized_experiments: MeasurementSpace,
    global_registry: ActuatorRegistry,
) -> None:

    es = (
        measurement_space_from_multiple_parameterized_experiments.compatibleEntitySpace()
    )
    ds_config = DiscoverySpaceConfiguration(
        entitySpace=es.constitutiveProperties,
        experiments=measurement_space_from_multiple_parameterized_experiments.experimentReferences,
        sampleStoreIdentifier="does-not-matter",
    )

    assert not isinstance(
        ds_config.experiments, MeasurementSpaceConfiguration
    ), "Expected the discovery space configuration fixtures experiment field to be a list of experiment references"
    config_copy = ds_config.convert_experiments_to_measurement_space_config()
    assert isinstance(
        config_copy.experiments, MeasurementSpaceConfiguration
    ), "Expected the experiment field of the copy of the discovery space configuration to be a MeasurementSpaceConfiguration"

    assert (
        config_copy == config_copy.convert_experiments_to_measurement_space_config()
    ), "Expected converting a config to config would do nothing"

    assert (
        ds_config.experiments
        == config_copy.convert_experiments_to_reference_list().experiments
    ), (
        "Expected the experiment reference list after converting the measurement space configuration"
        " version back would match the original reference list"
    )

    assert config_copy.convert_experiments_to_reference_list() == ds_config


def test_discovery_space_config_experiment_field_conversion(
    measurement_space_from_discovery_configuration: MeasurementSpace,
    global_registry: ActuatorRegistry,
) -> None:
    ms = measurement_space_from_discovery_configuration
    es = ms.compatibleEntitySpace()

    ds_config = DiscoverySpaceConfiguration(
        entitySpace=es.constitutiveProperties,
        experiments=ms.experimentReferences,
        sampleStoreIdentifier="does-not-matter",
    )

    assert not isinstance(
        ds_config.experiments, MeasurementSpaceConfiguration
    ), "Expected the discovery space configuration fixtures experiment field to be a list of experiment references"
    config_copy = ds_config.convert_experiments_to_measurement_space_config()
    assert isinstance(
        config_copy.experiments, MeasurementSpaceConfiguration
    ), "Expected the experiment field of the copy of the discovery space configuration to be a MeasurementSpaceConfiguration"

    assert (
        config_copy == config_copy.convert_experiments_to_measurement_space_config()
    ), "Expected converting a config to config would do nothing"

    assert (
        ds_config.experiments
        == config_copy.convert_experiments_to_reference_list().experiments
    ), (
        "Expected the experiment reference list after converting the measurement space configuration"
        " version back would match the original reference list"
    )

    assert config_copy.convert_experiments_to_reference_list() == ds_config


def test_sampled_entities(ml_multi_cloud_space: DiscoverySpace) -> None:

    assert (len(ml_multi_cloud_space.sampledEntities())) == 0


def test_measured_entities_table(ml_multi_cloud_space: DiscoverySpace) -> None:

    assert ml_multi_cloud_space.measuredEntitiesTable().shape[0] == 0


def test_matching_entities(ml_multi_cloud_space: DiscoverySpace) -> None:

    assert (len(ml_multi_cloud_space.matchingEntities())) == 42


def test_matching_entities_table(ml_multi_cloud_space: DiscoverySpace) -> None:

    assert ml_multi_cloud_space.matchingEntitiesTable().shape[0] == 42


def test_missing_entities_table(ml_multi_cloud_space: DiscoverySpace) -> None:

    assert ml_multi_cloud_space.matchingEntitiesTable().shape[0] == 42
