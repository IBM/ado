# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import re

import pytest

from orchestrator.schema.experiment import Experiment
from orchestrator.schema.virtual_property import (
    PropertyAggregationMethod,
    PropertyAggregationMethodEnum,
    VirtualObservedProperty,
    VirtualObservedPropertyValue,
)

# Test aggregate_from_observed_properties -> test via MeasurementResult.seriesRepresentation
# Test from_observed_properties_matching_identifier -> test via Entity.virtualObservedPropertiesFromIdentifier


@pytest.fixture(params=list(PropertyAggregationMethodEnum))
def aggregation_test_data(
    request: pytest.FixtureRequest,
) -> tuple[
    PropertyAggregationMethodEnum, list[int], tuple[float, float] | tuple[float, None]
]:
    import numpy as np

    values = np.asarray([1, 1, 1, 2, 2, 2, 10])
    retval = (None,)

    if request.param == PropertyAggregationMethodEnum.mean:
        retval = (np.mean(values), np.std(values) / np.sqrt(len(values)))
    elif request.param == PropertyAggregationMethodEnum.median:
        retval = (
            np.median(values),
            np.median(np.absolute(values - np.median(values))),
        )
    elif request.param == PropertyAggregationMethodEnum.min:
        retval = min(values), None
    elif request.param == PropertyAggregationMethodEnum.max:
        retval = max(values), None
    elif request.param == PropertyAggregationMethodEnum.std:
        retval = values.std(), None
    elif request.param == PropertyAggregationMethodEnum.variance:
        retval = values.var(), None

    return request.param, list(values), retval


@pytest.fixture
def virtual_properties(
    experiment: Experiment,
    aggregation_test_data: tuple[
        PropertyAggregationMethodEnum,
        list[int],
        tuple[float, float] | tuple[float, None],
    ],
) -> tuple[
    VirtualObservedProperty, list[int], tuple[float, float] | tuple[float, None]
]:
    observedProperty = experiment.observedProperties[0]
    identifier, values, results = aggregation_test_data
    method = PropertyAggregationMethod(identifier=identifier)

    return (
        VirtualObservedProperty(
            baseObservedProperty=observedProperty, aggregationMethod=method
        ),
        values,
        results,
    )


def test_property_aggregation(
    aggregation_test_data: tuple[
        PropertyAggregationMethodEnum,
        list[int],
        tuple[float, float] | tuple[float, None],
    ],
) -> None:
    identifier, values, results = aggregation_test_data
    method = PropertyAggregationMethod(identifier=identifier)
    assert method.function(values) == results

    with pytest.raises(
        ValueError,
        match="Values are required when applying property aggregation methods",
    ):
        method.function(values=[])


def test_virtual_properties(
    virtual_properties: tuple[
        VirtualObservedProperty, list[int], tuple[float, float] | tuple[float, None]
    ],
) -> None:
    virtual_property, values, results = virtual_properties
    assert virtual_property.aggregationMethod.function(values) == results
    assert virtual_property.aggregate(values) == VirtualObservedPropertyValue(
        property=virtual_property, value=results[0], uncertainty=results[1]
    )


def test_virtual_property_identifiers(
    virtual_properties: tuple[
        VirtualObservedProperty, list[int], tuple[float, float] | tuple[float, None]
    ],
) -> None:
    virtual_property, _values, _results = virtual_properties

    assert (
        virtual_property.identifier
        == f"{virtual_property.baseObservedProperty.identifier}-{virtual_property.aggregationMethod.identifier.value}"
    )
    assert (
        virtual_property.virtualTargetPropertyIdentifier
        == f"{virtual_property.baseObservedProperty.targetProperty.identifier}-{virtual_property.aggregationMethod.identifier.value}"
    )

    assert str(virtual_property) == f"vp-{virtual_property.identifier}"


def test_is_virtual_property_identifier() -> None:
    for e in PropertyAggregationMethodEnum:
        e = e.value
        assert VirtualObservedProperty.isVirtualPropertyIdentifier(f"my-property-{e}")
        assert VirtualObservedProperty.isVirtualPropertyIdentifier(f"myproperty-{e}")
        assert (
            VirtualObservedProperty.isVirtualPropertyIdentifier(f"myproperty{e}")
            is False
        )
        assert (
            VirtualObservedProperty.isVirtualPropertyIdentifier(f"my-property{e}")
            is False
        )
        assert (
            VirtualObservedProperty.isVirtualPropertyIdentifier(f"my-property_{e}")
            is False
        )
        assert VirtualObservedProperty.isVirtualPropertyIdentifier(f"{e}") is False


def test_parse_identifier() -> None:
    for e in PropertyAggregationMethodEnum:
        e = e.value
        component, method = VirtualObservedProperty.parseIdentifier(f"my-property-{e}")
        assert component == "my-property"
        assert method == e

        component, method = VirtualObservedProperty.parseIdentifier(f"myproperty-{e}")
        assert component == "myproperty"
        assert method == e

        with pytest.raises(
            ValueError,
            match=re.escape(
                "There must be at least one dash (-) in a virtual property identifier "
                "separating the aggregation method from the property name"
            ),
        ):
            VirtualObservedProperty.parseIdentifier(f"myproperty{e}")

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"'property{e}' is not a valid PropertyAggregationMethodEnum"
            ),
        ):
            VirtualObservedProperty.parseIdentifier(f"my-property{e}")

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"'property_{e}' is not a valid PropertyAggregationMethodEnum"
            ),
        ):
            VirtualObservedProperty.parseIdentifier(f"my-property_{e}")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "There must be at least one dash (-) in a virtual property identifier "
                "separating the aggregation method from the property name"
            ),
        ):
            VirtualObservedProperty.parseIdentifier(f"{e}")
