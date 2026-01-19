# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
from typing import Annotated, Any

import numpy as np
import pydantic
from typing_extensions import Self

from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property_value import PropertyValue


class PropertyAggregationMethodEnum(enum.Enum):
    mean = "mean"
    median = "median"
    variance = "variance"
    std = "std"
    min = "min"
    max = "max"


class PropertyAggregationMethod(pydantic.BaseModel):
    identifier: Annotated[PropertyAggregationMethodEnum, pydantic.Field()] = (
        PropertyAggregationMethodEnum.mean
    )

    def function(self, values: list) -> float | tuple[Any, Any] | tuple[Any, None]:
        """
        Apply property aggregation methods to values.

        Parameters:
        values (list): A list of values to apply aggregation methods to.

        Returns:
        The result of applying the aggregation method to the values.

        Raises:
        ValueError: If values is empty or none.
        """
        if not values:
            raise ValueError(
                "Values are required when applying property aggregation methods"
            )
        return functionMap[self.identifier](values)


def median(values: list) -> float:
    x = np.asarray(values)
    return np.median(values), np.median(np.absolute(x - np.median(x)))


functionMap = {
    PropertyAggregationMethodEnum.mean: lambda x: (
        np.asarray(x).mean(),
        np.asarray(x).std() / np.sqrt(len(x)),
    ),
    PropertyAggregationMethodEnum.median: median,
    PropertyAggregationMethodEnum.min: lambda x: (min(x), None),
    PropertyAggregationMethodEnum.max: lambda x: (max(x), None),
    PropertyAggregationMethodEnum.std: lambda x: (np.asarray(x).std(), None),
    PropertyAggregationMethodEnum.variance: lambda x: (np.asarray(x).var(), None),
}


class VirtualObservedProperty(pydantic.BaseModel):
    baseObservedProperty: ObservedProperty
    aggregationMethod: PropertyAggregationMethod

    @classmethod
    def isVirtualPropertyIdentifier(cls, identifier: str) -> bool:

        components = identifier.split("-")

        if len(components) < 2:
            retval = False
        else:
            try:
                PropertyAggregationMethodEnum(components[-1])
            except ValueError:
                retval = False
            else:
                retval = True

        return retval

    @classmethod
    def parseIdentifier(cls, identifier: str) -> str:

        components = identifier.split("-")

        if len(components) < 2:
            raise ValueError(
                "There must be at least one dash (-) "
                "in a virtual property identifier separating the aggregation method from the property name"
            )

        try:
            method = PropertyAggregationMethodEnum(components[-1])
        except ValueError:
            raise

        return "-".join(components[:-1]), method.value

    def __str__(self) -> str:

        return f"vp-{self.identifier}"

    @property
    def identifier(self) -> str:

        return f"{self.baseObservedProperty.identifier}-{self.aggregationMethod.identifier.value}"

    @property
    def virtualTargetPropertyIdentifier(self) -> str:

        return f"{self.baseObservedProperty.targetProperty.identifier}-{self.aggregationMethod.identifier.value}"

    def aggregate(self, values: list) -> "VirtualObservedPropertyValue":
        """
        Aggregate a list of values and return a VirtualObservedPropertyValue object.

        Parameters:
        values (typing.List): A list of values to be aggregated.

        Returns:
        VirtualObservedPropertyValue: A VirtualObservedPropertyValue object containing the aggregated value and uncertainty.

        Raises:
        ValueError: If values is empty or none.
        """
        value, uncertainty = self.aggregationMethod.function(values)
        return VirtualObservedPropertyValue(
            property=self, value=value, uncertainty=uncertainty
        )

    def aggregate_from_observed_properties(
        self, observed_property_values: list[ObservedPropertyValue]
    ) -> "VirtualObservedPropertyValue":
        """
        Aggregate values from observed property values.

        Parameters:
        observed_property_values (list[PropertyValue]): A list of observed property values.

        Returns:
        The aggregated value.

        Raises:
        ValueError: If there are no observed property values whose identifier matches that
        of the VirtualObservedProperty.
        """
        # Filter observed_property_values for ones related to this virtual observed property
        values = [
            v
            for v in observed_property_values
            if v.property.identifier == self.baseObservedProperty.identifier
        ]
        return self.aggregate([v.value for v in values])

    @classmethod
    def from_observed_properties_matching_identifier(
        cls, observed_properties: list[ObservedProperty], identifier: str
    ) -> list[Self] | None:

        if not VirtualObservedProperty.isVirtualPropertyIdentifier(identifier):
            raise ValueError(f"{identifier} is not a valid virtual property identifier")

        property_identifier, aggregation_method = (
            VirtualObservedProperty.parseIdentifier(identifier)
        )

        aggregation_method = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum(aggregation_method)
        )

        virtual_observed_properties = None

        # Check if it's an observed property
        if op := next(
            (o for o in observed_properties if o.identifier == property_identifier),
            None,
        ):
            virtual_observed_property = VirtualObservedProperty(
                baseObservedProperty=op, aggregationMethod=aggregation_method
            )

            # This can only ha
            if virtual_observed_property.identifier != identifier:
                raise ValueError(
                    f"InternalInconsistency: A VirtualObservedProperty instance created by parsing the identifier string, {identifier}, has a different value of the identifier property, {virtual_observed_property.identifier}, when it should be the same"
                )

            virtual_observed_properties = [virtual_observed_property]

        # Otherwise, check if it is a target property
        else:

            target_properties = [
                p
                for p in observed_properties
                if p.targetProperty.identifier == property_identifier
            ]

            if target_properties:
                virtual_observed_properties = [
                    VirtualObservedProperty(
                        baseObservedProperty=tp, aggregationMethod=aggregation_method
                    )
                    for tp in target_properties
                ]

        return virtual_observed_properties


class VirtualObservedPropertyValue(PropertyValue):

    property: Annotated[
        VirtualObservedProperty,
        pydantic.Field(description="The ConstitutiveProperty with the value"),
    ]
