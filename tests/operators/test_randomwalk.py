# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pathlib
from collections.abc import Callable
from typing import Literal

import pytest
import yaml

import ado.modules.operators.collections
from ado.core.discoveryspace.samplers import (
    ExplicitEntitySpaceGridSampleGenerator,
    WalkModeEnum,
)
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.operation import OperationException
from ado.core.operation.resource import (
    DiscoveryOperationResourceConfiguration,
    OperationExitStateEnum,
    OperationResourceEventEnum,
)
from ado.modules.operators.randomwalk import (
    BaseSamplerConfiguration,
    CustomSamplerConfiguration,
    EntityFilter,
    FilterModeEnum,
    SamplerModuleConf,
    resolve_number_entities_to_sample,
)
from ado.schema.domain import PropertyDomain
from ado.schema.entityspace import EntitySpaceRepresentation
from ado.schema.property import ConstitutiveProperty


def _discrete_space(size: int) -> EntitySpaceRepresentation:
    return EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(
                identifier="n",
                propertyDomain=PropertyDomain(values=list(range(size))),
            )
        ]
    )


def _continuous_space() -> EntitySpaceRepresentation:
    return EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(
                identifier="x",
                propertyDomain=PropertyDomain(domainRange=[0, 1]),
            )
        ]
    )


def _unbounded_discrete_space() -> EntitySpaceRepresentation:
    return EntitySpaceRepresentation(
        constitutiveProperties=[
            ConstitutiveProperty(
                identifier="n",
                propertyDomain=PropertyDomain(interval=1),
            )
        ]
    )


def _selector(
    mode: Literal["random", "sequential", "randomgrouped", "sequentialgrouped"] = (
        "sequential"
    ),
) -> BaseSamplerConfiguration:
    grouping = ["n"] if mode in {"randomgrouped", "sequentialgrouped"} else []
    return BaseSamplerConfiguration(
        samplerType="selector", mode=mode, grouping=grouping
    )


def _generator() -> BaseSamplerConfiguration:
    return BaseSamplerConfiguration(samplerType="generator", mode="sequential")


def test_selector_all_uses_matching_entities_when_space_is_larger() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_selector(),
        entity_space=_discrete_space(8),
        matching_entity_count=3,
        sample_store_entity_count=5,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 3
    assert resolved.source == "the number of matching entities in the sample store"


def test_selector_requires_matching_entity_count() -> None:
    with pytest.raises(ValueError, match="matching_entity_count is required"):
        resolve_number_entities_to_sample(
            number_entities="all",
            sampler_config=_selector(),
            entity_space=_discrete_space(8),
            matching_entity_count=None,
            sample_store_entity_count=5,
            filter_mode=FilterModeEnum.noFilter,
        )


def test_grouped_selector_all_uses_matching_entities() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_selector("sequentialgrouped"),
        entity_space=_discrete_space(8),
        matching_entity_count=3,
        sample_store_entity_count=5,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 3


def test_selector_all_on_non_discrete_space_uses_matching_entities() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_selector(),
        entity_space=_continuous_space(),
        matching_entity_count=2,
        sample_store_entity_count=4,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 2


def test_selector_all_with_no_entity_space_uses_matching_entities() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_selector(),
        entity_space=None,
        matching_entity_count=4,
        sample_store_entity_count=4,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 4


def test_selector_all_with_no_matching_entities_samples_zero() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_selector(),
        entity_space=_discrete_space(8),
        matching_entity_count=0,
        sample_store_entity_count=0,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 0


def test_generator_all_uses_entity_space_size() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=_generator(),
        entity_space=_discrete_space(8),
        matching_entity_count=None,
        sample_store_entity_count=3,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 8
    assert resolved.source == "the size of the entity space"


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


def test_generator_count_can_exceed_entities_already_in_the_store() -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities=5,
        sampler_config=_generator(),
        entity_space=_discrete_space(8),
        matching_entity_count=None,
        sample_store_entity_count=0,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 5
    assert resolved.source is None


def test_custom_sampler_all_uses_entity_space_size() -> None:
    sampler_config = CustomSamplerConfiguration(
        module=SamplerModuleConf(
            moduleClass="ExplicitEntitySpaceGridSampleGenerator",
            moduleName="ado.core.discoveryspace.samplers",
        ),
        parameters=ExplicitEntitySpaceGridSampleGenerator.parameters_model()(
            mode=WalkModeEnum.SEQUENTIAL
        ),
    )

    resolved = resolve_number_entities_to_sample(
        number_entities="all",
        sampler_config=sampler_config,
        entity_space=_discrete_space(8),
        matching_entity_count=None,
        sample_store_entity_count=1,
        filter_mode=FilterModeEnum.noFilter,
    )

    assert resolved.count == 8


def test_unfiltered_selector_raises_when_request_exceeds_matching_entities() -> None:
    with pytest.raises(ValueError, match="matching entities"):
        resolve_number_entities_to_sample(
            number_entities=4,
            sampler_config=_selector(),
            entity_space=_discrete_space(8),
            matching_entity_count=3,
            sample_store_entity_count=5,
            filter_mode=FilterModeEnum.noFilter,
        )


@pytest.mark.parametrize(
    "filter_mode",
    [FilterModeEnum.measured, FilterModeEnum.unmeasured, FilterModeEnum.partial],
)
def test_filtered_selector_allows_request_above_matching_count(
    filter_mode: FilterModeEnum,
) -> None:
    resolved = resolve_number_entities_to_sample(
        number_entities=4,
        sampler_config=_selector(),
        entity_space=_discrete_space(8),
        matching_entity_count=3,
        sample_store_entity_count=5,
        filter_mode=filter_mode,
    )

    assert resolved.count == 4


def test_request_above_space_size_raises_for_selector_even_with_a_filter() -> None:
    with pytest.raises(ValueError, match="entity space"):
        resolve_number_entities_to_sample(
            number_entities=9,
            sampler_config=_selector(),
            entity_space=_discrete_space(8),
            matching_entity_count=3,
            sample_store_entity_count=5,
            filter_mode=FilterModeEnum.unmeasured,
        )


def test_generator_all_raises_for_non_discrete_space() -> None:
    with pytest.raises(ValueError, match="non-discrete"):
        resolve_number_entities_to_sample(
            number_entities="all",
            sampler_config=_generator(),
            entity_space=_continuous_space(),
            matching_entity_count=None,
            sample_store_entity_count=2,
            filter_mode=FilterModeEnum.noFilter,
        )


def test_generator_all_raises_for_unbounded_discrete_space() -> None:
    with pytest.raises(ValueError, match="unbounded"):
        resolve_number_entities_to_sample(
            number_entities="all",
            sampler_config=_generator(),
            entity_space=_unbounded_discrete_space(),
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


def _ml_multi_cloud_operation(
    *,
    number_entities: int,
    sampler_type: Literal["selector", "generator"],
    filter_mode: FilterModeEnum,
) -> DiscoveryOperationResourceConfiguration:
    config = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(
            pathlib.Path(
                "examples/ml-multi-cloud/randomwalk_ml_multicloud_operation.yaml"
            ).read_text()
        )
    )
    parameters = config.operation.parameters
    parameters.numberEntities = number_entities
    parameters.samplerConfig = BaseSamplerConfiguration(
        mode="sequential",
        samplerType=sampler_type,
    )
    parameters.filter = EntityFilter(filterMode=filter_mode)
    return config


def _random_walk_function() -> Callable:
    function = ado.modules.operators.collections.explore.operators[
        "random_walk"
    ].function
    assert function is not None
    return function


def test_random_walk_selector_raises_when_count_exceeds_matching_entities(
    ml_multi_cloud_space: DiscoverySpace,
) -> None:
    """An unfiltered selector cannot request more entities than the store can supply."""

    matching = len(ml_multi_cloud_space.matchingEntities())
    assert matching < ml_multi_cloud_space.entitySpace.size
    requested = matching + 1
    assert requested <= ml_multi_cloud_space.entitySpace.size

    config = _ml_multi_cloud_operation(
        number_entities=requested,
        sampler_type="selector",
        filter_mode=FilterModeEnum.noFilter,
    )
    with pytest.raises(OperationException) as exc_info:
        _random_walk_function()(
            ml_multi_cloud_space, **config.operation.parameters.model_dump()
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
) -> None:
    """A filter that admits fewer than numberEntities still finishes successfully."""

    # One experiment cannot be partially measured, so every entity is rejected.
    assert len(ml_multi_cloud_space.measurementSpace.experiments) == 1
    matching = len(ml_multi_cloud_space.matchingEntities())
    config = _ml_multi_cloud_operation(
        number_entities=matching + 1,
        sampler_type="selector",
        filter_mode=FilterModeEnum.partial,
    )

    operation_output = _random_walk_function()(
        ml_multi_cloud_space, **config.operation.parameters.model_dump()
    )

    assert operation_output.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
    assert operation_output.operation.metadata["entities_submitted"] == 0
    assert operation_output.operation.metadata["experiments_requested"] == 0
