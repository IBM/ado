# Copyright IBM Corporation 2025, 2026
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


class TestAggregationWithNoneValues:
    """Aggregation functions must ignore None entries (treated as missing data)."""

    def test_mean_ignores_none(self) -> None:
        """Mean of [1.0, None, 3.0] must equal mean of [1.0, 3.0]."""
        import numpy as np

        method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.mean
        )
        value, uncertainty = method.function([1.0, None, 3.0])
        assert value == pytest.approx(np.mean([1.0, 3.0]))
        assert uncertainty == pytest.approx(np.std([1.0, 3.0]) / np.sqrt(2))

    def test_mean_nested_list_with_none(self) -> None:
        """Mean handles a single nested list whose elements may be None."""
        method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.mean
        )
        value, _ = method.function([[0.1, None, 0.3]])
        assert value == pytest.approx(0.2)

    def test_mean_all_none_returns_none(self) -> None:
        """Mean of all-None input must return (None, None)."""
        method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.mean
        )
        assert method.function([None, None, None]) == (None, None)

    def test_median_ignores_none(self) -> None:
        """Median of [1.0, None, 3.0] must equal median of [1.0, 3.0]."""
        import numpy as np

        method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.median
        )
        value, _ = method.function([1.0, None, 3.0])
        assert value == pytest.approx(np.median([1.0, 3.0]))

    def test_median_all_none_returns_none(self) -> None:
        """Median of all-None input must return (None, None)."""
        method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.median
        )
        assert method.function([None, None]) == (None, None)

    def test_min_ignores_none(self) -> None:
        """Min of [2.0, None, 5.0] must be 2.0."""
        method = PropertyAggregationMethod(identifier=PropertyAggregationMethodEnum.min)
        value, uncertainty = method.function([2.0, None, 5.0])
        assert value == pytest.approx(2.0)
        assert uncertainty is None

    def test_min_all_none_returns_none(self) -> None:
        """Min of all-None input must return (None, None)."""
        method = PropertyAggregationMethod(identifier=PropertyAggregationMethodEnum.min)
        assert method.function([None]) == (None, None)

    def test_max_ignores_none(self) -> None:
        """Max of [2.0, None, 5.0] must be 5.0."""
        method = PropertyAggregationMethod(identifier=PropertyAggregationMethodEnum.max)
        value, uncertainty = method.function([2.0, None, 5.0])
        assert value == pytest.approx(5.0)
        assert uncertainty is None

    def test_max_all_none_returns_none(self) -> None:
        """Max of all-None input must return (None, None)."""
        method = PropertyAggregationMethod(identifier=PropertyAggregationMethodEnum.max)
        assert method.function([None]) == (None, None)

    def test_std_ignores_none(self) -> None:
        """Std of [1.0, None, 3.0] must equal std of [1.0, 3.0]."""
        import numpy as np

        method = PropertyAggregationMethod(identifier=PropertyAggregationMethodEnum.std)
        value, uncertainty = method.function([1.0, None, 3.0])
        assert value == pytest.approx(np.std([1.0, 3.0]))
        assert uncertainty is None

    def test_std_all_none_returns_none(self) -> None:
        """Std of all-None input must return (None, None)."""
        method = PropertyAggregationMethod(identifier=PropertyAggregationMethodEnum.std)
        assert method.function([None, None]) == (None, None)

    def test_variance_ignores_none(self) -> None:
        """Variance of [1.0, None, 3.0] must equal variance of [1.0, 3.0]."""
        import numpy as np

        method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.variance
        )
        value, uncertainty = method.function([1.0, None, 3.0])
        assert value == pytest.approx(np.var([1.0, 3.0]))
        assert uncertainty is None

    def test_variance_all_none_returns_none(self) -> None:
        """Variance of all-None input must return (None, None)."""
        method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum.variance
        )
        assert method.function([None]) == (None, None)


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
