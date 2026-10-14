# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest

import ado.modules.operators.collections
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.operation import OperationException
from ado.core.operation.resource import (
    DiscoveryOperationResourceConfiguration,
    OperationExitStateEnum,
    OperationResourceEventEnum,
)
from ado.modules.operators.randomwalk import (
    AllCountSource,
    BaseSamplerConfiguration,
    EntityFilter,
    FilterModeEnum,
    resolve_number_entities_to_sample,
)
from ado.schema.domain import PropertyDomain
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.property import ConstitutiveProperty


def _entity_space(domain: PropertyDomain) -> EntitySpaceRepresentation:
    """Build a one-property entity space from a domain fixture."""

    return EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(identifier="x", propertyDomain=domain)
        ]
    )


def _selector() -> BaseSamplerConfiguration:
    """Return a sequential selector sampler configuration."""

    return BaseSamplerConfiguration(samplerType="selector", mode="sequential")


def _generator() -> BaseSamplerConfiguration:
    """Return a sequential generator sampler configuration."""

    return BaseSamplerConfiguration(samplerType="generator", mode="sequential")


def test_selector_all_uses_matching_entities_when_space_is_larger(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    entity_space = _entity_space(property_domain_implicit_discrete_with_values)
    matching_entity_count = 3
    assert matching_entity_count < entity_space.size

    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_selector(),
        entity_space=entity_space,
        matching_entity_count=matching_entity_count,
        sample_store_entity_count=entity_space.size - 1,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 3
    assert resolved.source == AllCountSource.MATCHING_ENTITIES


def test_selector_requires_matching_entity_count(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    with pytest.raises(ValueError, match="matching_entity_count is required"):
        resolve_number_entities_to_sample(
            number_entities="all",
            sampler_config=_selector(),
            entity_space=_entity_space(property_domain_implicit_discrete_with_values),
            matching_entity_count=None,
            sample_store_entity_count=5,
            filter_mode=FilterModeEnum.noFilter,
        )


def test_selector_all_with_no_matching_entities_samples_zero(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_selector(),
        entity_space=_entity_space(property_domain_implicit_discrete_with_values),
        matching_entity_count=0,
        sample_store_entity_count=0,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 0


def test_generator_all_uses_entity_space_size(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    entity_space = _entity_space(property_domain_implicit_discrete_with_values)
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_generator(),
        entity_space=entity_space,
        matching_entity_count=None,
        sample_store_entity_count=1,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == entity_space.size
    assert resolved.source == AllCountSource.ENTITY_SPACE


def test_generator_all_with_no_entity_space_uses_sample_store_count() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_generator(),
        entity_space=None,
        matching_entity_count=None,
        sample_store_entity_count=6,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 6
    assert resolved.source == AllCountSource.SAMPLE_STORE


def test_generator_count_can_exceed_entities_already_in_the_store(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    entity_space = _entity_space(property_domain_implicit_discrete_with_values)
    requested = entity_space.size - 1
    resolved = resolve_number_entities_to_sample(
        number_entities=requested,
        sampler_config=_generator(),
        entity_space=entity_space,
        matching_entity_count=None,
        sample_store_entity_count=0,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == requested
    assert resolved.source is None


def test_unfiltered_selector_raises_when_request_exceeds_matching_entities(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    entity_space = _entity_space(property_domain_implicit_discrete_with_values)
    matching_entity_count = 3
    requested = matching_entity_count + 1
    assert requested <= entity_space.size

    with pytest.raises(ValueError, match="matching entities"):
        resolve_number_entities_to_sample(
            number_entities=requested,
            sampler_config=_selector(),
            entity_space=entity_space,
            matching_entity_count=matching_entity_count,
            sample_store_entity_count=entity_space.size,
            filter_mode=FilterModeEnum.noFilter,
        )


def test_filtered_selector_allows_request_above_matching_count(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    entity_space = _entity_space(property_domain_implicit_discrete_with_values)
    matching_entity_count = 3
    requested = matching_entity_count + 1
    assert requested <= entity_space.size

    resolved = resolve_number_entities_to_sample(
        number_entities=requested,
        sampler_config=_selector(),
        entity_space=entity_space,
        matching_entity_count=matching_entity_count,
        sample_store_entity_count=entity_space.size,
        filter_mode=FilterModeEnum.unmeasured,
    )

    assert resolved.count == requested


def test_request_above_space_size_raises_for_selector_even_with_a_filter(
    property_domain_implicit_discrete_with_values: PropertyDomain,
) -> None:
    entity_space = _entity_space(property_domain_implicit_discrete_with_values)
    with pytest.raises(ValueError, match="entity space"):
        resolve_number_entities_to_sample(
            number_entities=entity_space.size + 1,
            sampler_config=_selector(),
            entity_space=entity_space,
            matching_entity_count=1,
            sample_store_entity_count=entity_space.size,
            filter_mode=FilterModeEnum.unmeasured,
        )


def test_generator_all_raises_for_non_discrete_space(
    property_domain_implicit_continuous_with_domain_range: PropertyDomain,
) -> None:
    with pytest.raises(ValueError, match="non-discrete"):
        resolve_number_entities_to_sample(
            number_entities="all",
            sampler_config=_generator(),
            entity_space=_entity_space(
                property_domain_implicit_continuous_with_domain_range
            ),
            matching_entity_count=None,
            sample_store_entity_count=2,
            filter_mode=FilterModeEnum.noFilter,
        )


def test_generator_all_raises_for_unbounded_discrete_space(
    property_domain_implicit_discrete_with_interval: PropertyDomain,
) -> None:
    with pytest.raises(ValueError, match="unbounded"):
        resolve_number_entities_to_sample(
            number_entities="all",
            sampler_config=_generator(),
            entity_space=_entity_space(property_domain_implicit_discrete_with_interval),
            matching_entity_count=None,
            sample_store_entity_count=0,
            filter_mode=FilterModeEnum.noFilter,
        )


def test_request_above_sample_store_count_raises_when_there_is_no_entity_space() -> (
    None
):
    with pytest.raises(ValueError, match="sample store"):
        resolve_number_entities_to_sample(
            number_entities=5,
            sampler_config=_generator(),
            entity_space=None,
            matching_entity_count=None,
            sample_store_entity_count=2,
            filter_mode=FilterModeEnum.noFilter,
        )


def _configure_selector(
    config: DiscoveryOperationResourceConfiguration,
    *,
    number_entities: int,
    filter_mode: FilterModeEnum,
) -> DiscoveryOperationResourceConfiguration:
    """Point an existing random-walk operation at a sequential selector."""

    parameters = config.operation.parameters
    parameters.numberEntities = number_entities
    parameters.samplerConfig = BaseSamplerConfiguration(
        mode="sequential",
        samplerType="selector",
    )
    parameters.filter = EntityFilter(filterMode=filter_mode)
    return config


def test_random_walk_selector_raises_when_count_exceeds_matching_entities(
    ml_multi_cloud_space: DiscoverySpace,
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
) -> None:
    """An unfiltered selector cannot request more entities than the store can supply."""

    matching = len(ml_multi_cloud_space.matchingEntities())
    assert matching < ml_multi_cloud_space.entitySpace.size
    requested = matching + 1
    assert requested <= ml_multi_cloud_space.entitySpace.size

    function = ado.modules.operators.collections.explore.operators[
        "random_walk"
    ].function
    assert function is not None
    with pytest.raises(OperationException) as exc_info:
        function(
            ml_multi_cloud_space,
            **_configure_selector(
                ml_multi_cloud_operation_configuration,
                number_entities=requested,
                filter_mode=FilterModeEnum.noFilter,
            ).operation.parameters.model_dump(),
        )

    finished = [
        status
        for status in exc_info.value.operation.status
        if status.event == OperationResourceEventEnum.FINISHED
    ]
    assert finished[-1].exit_state == OperationExitStateEnum.ERROR
    assert finished[-1].message is not None
    assert "matching entities" in finished[-1].message
    assert exc_info.value.operation.metadata["entities_submitted"] == 0


def test_random_walk_selector_filter_succeeds_when_fewer_entities_pass(
    ml_multi_cloud_space: DiscoverySpace,
    ml_multi_cloud_operation_configuration: DiscoveryOperationResourceConfiguration,
) -> None:
    """A filter that admits fewer than numberEntities still finishes successfully."""

    # One experiment cannot be partially measured, so every entity is rejected.
    assert len(ml_multi_cloud_space.measurementSpace.experiments) == 1
    matching = len(ml_multi_cloud_space.matchingEntities())
    function = ado.modules.operators.collections.explore.operators[
        "random_walk"
    ].function
    assert function is not None

    operation_output = function(
        ml_multi_cloud_space,
        **_configure_selector(
            ml_multi_cloud_operation_configuration,
            number_entities=matching + 1,
            filter_mode=FilterModeEnum.partial,
        ).operation.parameters.model_dump(),
    )

    assert operation_output.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
    assert operation_output.operation.metadata["entities_submitted"] == 0
    assert operation_output.operation.metadata["experiments_requested"] == 0
