# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""Module containing discovery state interfaces"""

import importlib.metadata
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.schema.experiment import Experiment
from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.property import (
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
    MeasuredPropertyTypeEnum,
    Property,
    PropertyDescriptor,
)
from orchestrator.schema.property_value import ConstitutivePropertyValue
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.result import (
    DuplicateMeasurementResultError,
    MeasurementResult,
    ValidMeasurementResult,
)
from orchestrator.schema.virtual_property import (
    PropertyAggregationMethod,
    PropertyAggregationMethodEnum,
    VirtualObservedProperty,
    VirtualObservedPropertyValue,
)

if typing.TYPE_CHECKING:  # pragma: nocover
    import pandas as pd
    from rich.console import RenderableType


def entity_identifier_from_properties_and_values(point: dict[str, typing.Any]) -> str:
    """
    Creates an entity identifier based on a set of constitutive property ids and values for those properties

    Parameters:
        point: A dictionary of constitutive property id, value pairs

    Returns:
        An entity identifier
    """

    parts = [f"{key}.{point[key]}" for key in point]
    return "-".join(parts)


class Entity(pydantic.BaseModel):
    """Represents an entity in a discovery space

    Entities have properties which have values.
    An entities properties can be measured (ObservedProperty) or deduced from its identity (ConstitutiveProperty)

    An ObservedProperty represents a specific way to measure a target property.  It is associated with an Experiment.

    A ConstitutiveProperty does not have a target property and is not associated with an Experiment

    Some operations can process both types of properties, others depend on if the property is one type or another.

    """

    identifier: Annotated[
        str | None,
        pydantic.Field(
            description="An id that uniquely defines this entity w.r.t others."
            "If one is not supplied it is generated from the constitutive properties"
        ),
    ] = None
    generatorid: Annotated[
        str,
        pydantic.Field(description="The id of the generator that created this entity"),
    ] = "unk"
    constitutive_property_values: Annotated[
        tuple[ConstitutivePropertyValue, ...],
        pydantic.Field(
            frozen=True,
            description="A list of ConstitutivePropertyValue objects giving values for constitutive properties",
        ),
    ]
    measurement_results: Annotated[
        list["ValidMeasurementResult"],
        pydantic.Field(
            default_factory=list,
            description="A list of ValidMeasurementResult objects giving values for observed properties. "
            "InvalidMeasurementResults are not supported.",
        ),
    ]
    metadata: Annotated[
        dict | None, pydantic.Field(description="Additional metadata on this entity")
    ] = None
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "version": importlib.metadata.version(distribution_name="ado-core")
        },
    )

    @property
    def propertyValues(self) -> list[ConstitutivePropertyValue | ObservedPropertyValue]:
        v = []
        for result in self.measurement_results:
            v.extend(result.measurements)
        v.extend(self.constitutive_property_values)
        return v

    @property
    def properties(self) -> list[ObservedProperty | ConstitutivePropertyDescriptor]:
        """
        Return a list of unique properties from the entity's measurement results.

        Returns:
            list[typing.Union[ObservedProperty, ConstitutiveProperty]]: A list of unique properties.
        """
        known_property_identifiers = set()
        unique_properties = []

        for result in self.measurement_results:
            for measurement in result.measurements:
                if (
                    identifier := measurement.property.identifier
                ) not in known_property_identifiers:
                    unique_properties.append(measurement.property)
                    known_property_identifiers.add(identifier)

        return unique_properties + self.constitutiveProperties

    @classmethod
    def identifier_from_property_values(
        cls, property_values: typing.Iterable[ConstitutivePropertyValue]
    ) -> str:
        """Returns the identifier that would be generated for an entity with the given constitutive property values

        Raise ValueError if all members of property_values do not refer to ConstitutiveProperties
        """

        if not all(
            isinstance(pv.property, ConstitutivePropertyDescriptor)
            for pv in property_values
        ):
            raise ValueError("All values must be for ConstitutiveProperties")

        return entity_identifier_from_properties_and_values(
            {pv.property.identifier: pv.value for pv in property_values}
        )

    @pydantic.field_validator("constitutive_property_values", mode="after")
    @classmethod
    def guarantee_constitutive_properties_are_unique(
        cls, values: tuple[ConstitutivePropertyValue]
    ) -> tuple[ConstitutivePropertyValue]:

        if not values:
            return values

        unique_property_identifiers = {value.property.identifier for value in values}
        if len(unique_property_identifiers) != len(values):
            from collections import Counter

            property_occurrences = Counter(
                [value.property.identifier for value in values]
            )
            raise ValueError(
                "Constitutive properties must be unique. "
                "The occurrences of each of the constitutive properties were: "
                f"{property_occurrences}"
            )

        return values

    @pydantic.field_validator("measurement_results", mode="after")
    @classmethod
    def guarantee_unique_measurement_results(
        cls, measurement_results: list["ValidMeasurementResult"]
    ) -> list["ValidMeasurementResult"]:

        if not measurement_results:
            return measurement_results

        unique_results_uid = {result.uid for result in measurement_results}
        if len(unique_results_uid) != len(measurement_results):
            from collections import Counter

            uid_occurrences = Counter([result.uid for result in measurement_results])
            raise DuplicateMeasurementResultError(
                f"There were {len(measurement_results) - len(unique_results_uid)} "
                "duplicate MeasurementResults (they had the same UID). "
                f"Occurrences were: {uid_occurrences}"
            )

        return measurement_results

    @pydantic.model_validator(mode="after")
    def check_identifier(self) -> "Entity":
        """Checks if an external identifier was passed and if not generates one"""

        self.identifier = (
            self.identifier
            if self.identifier is not None
            else Entity.identifier_from_property_values(
                self.constitutive_property_values
            )
        )

        return self

    def __str__(self) -> str:
        return f"{self.identifier} ({self.generatorid})"

    def __rich__(self) -> "RenderableType":
        """Render this entity using rich."""
        import pandas as pd
        import rich.box
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        from orchestrator.utilities.rich import dataframe_to_rich_table

        content = [
            Text.assemble(
                ("Identifier: ", "bold"),
                (self.identifier, "bold green"),
                ("Generator: ", "bold"),
                self.generatorid,
                "",
            )
        ]

        # Constitutive properties table
        data = [
            [cv.property.identifier, cv.value]
            for cv in self.constitutive_property_values
        ]
        df = pd.DataFrame(data, columns=["name", "value"])

        content.extend(
            [
                Text("Constitutive properties:", style="bold"),
                Panel(dataframe_to_rich_table(df), box=rich.box.SIMPLE_HEAD),
            ]
        )

        # Observed properties table
        data = [
            [
                op.identifier,
                op.experimentReference,
                op.targetProperty.identifier,
                [v.value for v in self.valuesForProperty(op)],
            ]
            for op in self.observedProperties
        ]
        df = pd.DataFrame(
            data, columns=["name", "experiment", "target-property", "values"]
        )
        content.extend(
            [
                Text("Observed properties:", style="bold"),
                Panel(dataframe_to_rich_table(df), box=rich.box.SIMPLE_HEAD),
            ]
        )

        # Associated experiments
        content.extend(
            [
                Text("Associated experiments:", style="bold"),
                Panel(
                    Group(*[Text(str(e)) for e in self.experimentReferences]),
                    box=rich.box.HORIZONTALS,
                    padding=(0, 2),
                ),
            ]
        )

        return Group(*content)

    @property
    def observedProperties(self) -> list[ObservedProperty]:
        """Returns the measured properties"""

        return [
            p for p in self.properties if p.propertyType in MeasuredPropertyTypeEnum
        ]

    @property
    def observedPropertyValues(self) -> list[ObservedPropertyValue]:

        return [
            p for p in self.propertyValues if isinstance(p.property, ObservedProperty)
        ]

    @property
    def constitutiveProperties(self) -> list[ConstitutivePropertyDescriptor]:
        """
        Returns a list of unique constitutive properties descriptors

        Returns:
        List[ConstitutivePropertyDescriptor]: A list of unique constitutive properties.
        """
        known_property_identifiers = set()
        unique_properties = []

        for constitutive_property in self.constitutive_property_values:
            if (
                identifier := constitutive_property.property.identifier
                not in known_property_identifiers
            ):
                unique_properties.append(constitutive_property.property)
                known_property_identifiers.add(identifier)

        return unique_properties

    def observedPropertiesFromExperimentReference(
        self, experimentReference: ExperimentReference
    ) -> list[ObservedProperty]:
        """Returns all the properties of the entity that are measured by experimentReference

        If there are no observed properties for the experiment this method returns an empty list

        :param experimentReference: An ExperimentReference instance"""

        return [
            v
            for v in self.observedProperties
            if v.experimentReference == experimentReference
        ]

    def propertyValuesFromExperiment(
        self, experiment: Experiment
    ) -> list[ObservedPropertyValue]:
        """Returns all the property values of the entity measured by experiment

        If there are no measured properties for experiment this method returns an empty list

        :param experiment: An Experiment instance"""

        return [
            v
            for v in self.observedPropertyValues
            if v.property.experimentReference == experiment.reference
        ]

    def measurement_results_for_experiment_reference(
        self, experiment_reference: ExperimentReference
    ) -> list[MeasurementResult]:
        return [
            result
            for result in self.measurement_results
            if result.experimentReference == experiment_reference
        ]

    def propertyValuesFromExperimentReference(
        self, experimentReference: ExperimentReference
    ) -> list[ObservedPropertyValue]:
        """Returns all the property values of the entity measured by experiment

        If there are no measured properties for experiment this method returns an empty list

        :param experimentReference: An ExperimentReference instance"""

        return [
            v
            for v in self.observedPropertyValues
            if v.property.experimentReference == experimentReference
        ]

    def virtualObservedPropertiesFromIdentifier(
        self, identifier: str
    ) -> list[VirtualObservedProperty] | None:
        """Returns a list of VirtualObservedProperty instances given a virtual property identifier

        A virtual property identifier has two parts - the base property identifier and the aggregation method identifier
        The base property identifier can be the identifier of an ObservedProperty or a target property.
        The aggregation method identifier must be one of the know aggregation method identifiers

        This method finds the observed properties of the receiver that match the base property identifier
        and creates virtual properties for them.
        If the base property identifier matches an ObservedProperty then a single VirtualObservedProperty is created.
        If the base property identifier matches a TargetProperty then a VirtualObservedProperty is created
        for each ObservedProperty that has that TargetProperty

        Exceptions:
            Raises ValueError if identifier is not a virtual property identifier

        Returns:
            None if the receiver has no observed or target properties that match the base property ident.
            Otherwise, a list of VirtualObservedProperty instances.
            This list will be of length 1 if the base-property-identifier matches an ObservedProperty.
            If the base property identifier matches a TargetProperty then a VirtualObservedProperty is created
            for each ObservedProperty that has that TargetProperty.

        """

        return VirtualObservedProperty.from_observed_properties_matching_identifier(
            self.observedProperties, identifier
        )

    def valuesForProperty(
        self,
        property: (
            Property | PropertyDescriptor | ObservedProperty | VirtualObservedProperty
        ),
    ) -> list[
        ConstitutivePropertyValue | ObservedPropertyValue | VirtualObservedPropertyValue
    ]:
        """Returns all values for given observed property. If none exit returns an empty list"""

        if isinstance(property, VirtualObservedProperty):
            values = self.valuesForProperty(property.baseObservedProperty)
            if len(values) > 0:
                check = [property.aggregate([v.value for v in values])]
            else:
                check = []
        else:
            check = list(
                filter(
                    lambda x: x.property.identifier == property.identifier,
                    self.propertyValues,
                )
            )

        return check

    def valueForProperty(
        self,
        property: (
            ObservedProperty
            | ConstitutiveProperty
            | ConstitutivePropertyDescriptor
            | VirtualObservedProperty
        ),
    ) -> (
        ConstitutivePropertyValue | ObservedPropertyValue | VirtualObservedPropertyValue
    ):
        """Returns an PropertyValue for Property if one exists otherwise None

        If the property is an ObservedProperty and multiple values exist the first is returned.
        """

        check = self.valuesForProperty(property)
        return check[0] if len(check) != 0 else None

    def valuesForTargetProperty(
        self, targetProperty: Property | PropertyDescriptor
    ) -> list[ObservedPropertyValue]:
        """Returns all PropertyValue instances for targetProperty if one exists otherwise empty list"""

        return list(
            filter(
                lambda x: x.property.targetProperty.identifier
                == targetProperty.identifier,
                self.observedPropertyValues,
            )
        )

    def valueForConstitutivePropertyIdentifier(
        self, identifier: str
    ) -> ConstitutivePropertyValue:
        """Returns the value of the ConstitutiveProperty with the given identifier if one exists otherwise None"""

        check = list(
            filter(
                lambda x: x.property.identifier == identifier,
                self.constitutive_property_values,
            )
        )
        return check[0] if len(check) != 0 else None

    def valuesForObservedPropertyIdentifier(
        self, identifier: str
    ) -> list[ObservedPropertyValue]:
        """Returns the values of the ObservedProperty with the given identifier if one exists otherwise an empty list"""

        return list(
            filter(
                lambda x: x.property.identifier == identifier,
                self.observedPropertyValues,
            )
        )

    def add_measurement_result(
        self, result: typing.Union["ValidMeasurementResult"]
    ) -> None:
        """
        Adds a ValidMeasurementResult object to the entity.
        """
        if any(
            result.uid == existing_result.uid
            for existing_result in self.measurement_results
        ):
            raise DuplicateMeasurementResultError(
                f"Entity {self.identifier} already contained a MeasurementResult with id {result.uid}.\n"
                f"Measurements were {self.measurement_results}."
            )

        self.measurement_results.append(result)

    @property
    def experimentReferences(self) -> list[ExperimentReference]:
        """Returns a list of the Experiments that measure the observed properties"""

        return list({op.experimentReference for op in self.observedProperties})

    def seriesRepresentation(
        self,
        experimentReferences: list[ExperimentReference] | None = None,
        constitutiveOnly: bool = False,
        virtualTargetPropertyIdentifiers: list[str] | None = None,
        aggregationMethod: PropertyAggregationMethodEnum | None = None,
    ) -> "pd.Series":
        """Returns a pandas series containing the receivers constitutive and optional observed property values

        The keys of the pandas series are the property identifiers and the values, the property values

        If there is more than one value for a property the behaviour depends on the aggregate parameter.
        If it is False, the default, the value for the property will be a list of the values.

        If aggregate is True multiple numeric values will be averaged to produce one value.
        The name of property in this case will be the corresponding virtual property name

        If one, more or all values are non-numeric then a list of the values is returned i.e. same as aggregate=False
        """

        import pandas as pd

        def add_value(
            value: ConstitutivePropertyValue | ObservedPropertyValue,
            references: list[ExperimentReference],
            restrictConstitutive: bool = False,
        ) -> bool:
            """Checks if a property value should be added to the series

            The rules are
            - ConstitutiveProperties are always added
            - ObservedProperties are added if restrictConstitutive is False AND either
                A   no/None experiment references are passed
                B.  the property is from an experiment in the references list

            """

            if isinstance(value, ConstitutivePropertyValue):
                return True

            if restrictConstitutive:
                return False

            return bool(
                not references or value.property.experimentReference in references
            )

        d = {}
        observed_property_map = {op.identifier: op for op in self.observedProperties}
        for v in self.propertyValues:
            if add_value(v, experimentReferences, constitutiveOnly):
                if d.get(v.property.identifier) is None:
                    d[v.property.identifier] = []

                d[v.property.identifier].append(v.value)

        if virtualTargetPropertyIdentifiers is not None:
            for ident in virtualTargetPropertyIdentifiers:
                vops = self.virtualObservedPropertiesFromIdentifier(ident)
                for vop in vops:
                    v = self.valueForProperty(vop)
                    if v is not None:
                        if d.get(vop.identifier) is None:
                            d[vop.identifier] = []

                        d[vop.identifier].append(v.value)

        # This is so the dict can be modified in the loop
        props = list(d.keys())
        for o in props:
            if observed_property_map.get(o):
                if aggregationMethod:
                    vop = VirtualObservedProperty(
                        baseObservedProperty=observed_property_map[o],
                        aggregationMethod=PropertyAggregationMethod(
                            identifier=aggregationMethod
                        ),
                    )

                    try:
                        # Replace the observed property id with the virtual property id
                        mean = vop.aggregate(d[o])
                        d.pop(o)
                        d[vop.identifier] = mean.value
                    except (
                        ValueError,
                        TypeError,
                    ) as error:
                        # We can't take the mean of all the values for some reason
                        # An example is that the values are arrays of different length
                        import logging

                        log = logging.getLogger("entity")
                        log.debug(
                            f"Unable to calculate mean of value for {o} due to {error}. Will not aggregate"
                        )
                else:
                    # Remove list if only one entry
                    if len(d[o]) == 1:
                        d[o] = d[o][0]
            else:
                # Remove list for constitutive properties
                if isinstance(d[o], list):
                    d[o] = d[o][0]

        d["identifier"] = self.identifier
        d["generatorid"] = self.generatorid

        return pd.Series(d)

    def experimentSeries(
        self,
        experimentReferences: list[ExperimentReference] | None = None,
        virtualTargetPropertyIdentifiers: list[str] | None = None,
        aggregationMethod: PropertyAggregationMethodEnum | None = None,
    ) -> list["pd.Series"]:
        """Returns a tuple of series' where each series contains the observed property values for a specific experiment (protocol).

        The key of each observed property value is the target property identifier c.f. seriesRepresentation where it is the observed property identifier
        Each series also contains the entities constitutive property values.

        The value of the "experiment_id" is the relevant ExperimentReference instance

        Params:
            experimentReferences: The experiments to add results for. If empty, no series will be returned.
            virtualTargetPropertyIdentifiers: virtual properties that should be added to the output
            aggregate: If True properties with multiple values are averaged if possible

        """

        import pandas as pd

        if experimentReferences is None:
            experimentReferences = self.experimentReferences

        resultsDict = {}
        for e in experimentReferences:

            if len(self.propertyValuesFromExperimentReference(e)) == 0:
                continue

            refString = f"{e}"
            if resultsDict.get(refString) is None:
                resultsDict[refString] = {}

            # Constitutive properties
            d = resultsDict[refString]
            d["identifier"] = self.identifier
            d["generatorid"] = self.generatorid
            d["experiment_id"] = e

            # There should be only one value for a constitutive property
            for v in self.constitutive_property_values:
                d[v.property.identifier] = v.value

            # Experiment properties
            # Keep them separate from "d" until we average multiple values
            # to avoid trying to average string values
            expProperties = {}
            for v in self.propertyValuesFromExperimentReference(e):
                if expProperties.get(v.property.targetProperty.identifier) is None:
                    expProperties[v.property.targetProperty.identifier] = []

                expProperties[v.property.targetProperty.identifier].append(v.value)

            #
            # Add any virtual properties
            #
            if virtualTargetPropertyIdentifiers is not None:
                for ident in virtualTargetPropertyIdentifiers:
                    vops = self.virtualObservedPropertiesFromIdentifier(ident)
                    for vop in vops:
                        v = self.valueForProperty(vop)
                        if v is not None:
                            if (
                                expProperties.get(vop.virtualTargetPropertyIdentifier)
                                is None
                            ):
                                expProperties[vop.virtualTargetPropertyIdentifier] = []

                            expProperties[vop.virtualTargetPropertyIdentifier].append(
                                v.value
                            )

            props = list(expProperties.keys())
            prop_map = {
                op.targetProperty.identifier: op
                for op in self.observedPropertiesFromExperimentReference(e)
            }
            for o in props:
                if len(expProperties[o]) > 1:
                    if aggregationMethod:
                        vop = VirtualObservedProperty(
                            baseObservedProperty=prop_map[o],
                            aggregationMethod=PropertyAggregationMethod(
                                identifier=aggregationMethod
                            ),
                        )

                        try:
                            # Replace the target property with the virtual target property id
                            mean = vop.aggregate(expProperties[o])
                            expProperties.pop(o)
                            expProperties[vop.virtualTargetPropertyIdentifier] = (
                                mean.value
                            )
                        except (
                            ValueError,
                            TypeError,
                        ) as error:
                            # We can't take the mean of all the values for some reason
                            # An example is that the values are arrays of different length
                            import logging

                            log = logging.getLogger("entity")
                            log.debug(
                                f"Unable to calculate mean of value for {o} due to {error}. Will not aggregate"
                            )
                    else:
                        pass
                else:
                    expProperties[o] = expProperties[o][0]

            d.update(expProperties)

        return [pd.Series(d) for d in resultsDict.values()]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False

        return {
            (constitutive_property.value, constitutive_property.property.identifier)
            for constitutive_property in self.constitutive_property_values
        } == {
            (constitutive_property.value, constitutive_property.property.identifier)
            for constitutive_property in other.constitutive_property_values
        }


def CheckRequiredObservedPropertyValuesPresent(
    entity: Entity,
    experiment: Experiment,
    exactMatch: bool = True,
) -> bool:
    """
    Checks if an Entity has values for the observed properties required by Experiment

    :param: entity: The Entity instance to check
    :param: experiment: The Experiment that has required properties
    :param: exactMatch: If True an exact match, including parameterization, must be made.
        If the entity has a result of a different parameterization to the one required
        by Experiment this function will return False
        If False, if the entity has a result from any parameterization of the base experiment
        required by Experiment this function will returh True

    :return:
        True if Entity has values for the required observed properties, false otherwise
        NOTE: If an Experiment has no required observed properties this method returns True
    """

    retval = True
    for p in experiment.requiredObservedProperties:
        if entity.valueForProperty(p) is None:
            retval = False
            break

    if not exactMatch:
        retval = True
        for requiredProperty in experiment.requiredObservedProperties:
            found = False
            for ref in entity.experimentReferences:
                # Check if base experiment match
                if (
                    ref.experimentIdentifier
                    == requiredProperty.experimentReference.experimentIdentifier
                ):
                    # The entity has observed properties from a parameterization of the base experiment
                    # Check if the values are present
                    for pv in entity.propertyValuesFromExperimentReference(ref):
                        # Check if the pv measures the targetProperty
                        if (
                            pv.property.targetProperty
                            == requiredProperty.targetProperty
                        ):
                            # We've found one value, we can exit
                            found = True
                            break

            # If we couldn't find a value for this required property across
            # all the measurements then we can exit
            if not found:
                retval = False
                break

    return retval


def CheckRequiredConstitutivePropertyValuesPresent(
    entity: Entity, experiment: Experiment
) -> bool:
    """
    Checks if an Entity has the constitutive properties required by Experiment

    :param: entity: An Entity instance
    :param: experiment: An Experiment instance

    :return:
        True if Entity has values for the required properties, false otherwise

        NOTE: If the experiment does not define any required constitutive properties this function will return True
    """

    retval = True
    for p in experiment.requiredConstitutiveProperties:
        if entity.valueForProperty(p) is None:
            retval = False
            break

    return retval
