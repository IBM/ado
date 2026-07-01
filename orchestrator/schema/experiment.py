# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import enum
import importlib.metadata
import sys
import typing
from collections.abc import Iterable
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.schema.observed_property import (
    ObservedProperty,
)
from orchestrator.schema.property import (
    AbstractPropertyDescriptor,
    ConcretePropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
    MeasuredPropertyTypeEnum,
    Property,
    PropertyDescriptor,
)
from orchestrator.schema.property_value import (
    ConstitutivePropertyValue,
    PropertyValue,
    validate_point_against_properties,
)
from orchestrator.schema.reference import (
    ExperimentReference,
    check_parameterization_validity,
    identifier_for_parameterized_experiment,
    reference_string_from_fields,
)
from orchestrator.schema.virtual_property import (
    PropertyAggregationMethod,
    PropertyAggregationMethodEnum,
    VirtualObservedProperty,
)
from orchestrator.utilities.pydantic import StrictSemVerStr, semver_major

if typing.TYPE_CHECKING:  # pragma: nocover
    from rich.console import RenderableType

    from orchestrator.schema.entity import Entity


class Experiment(pydantic.BaseModel):
    """Represents an experiment that can measure properties of an entities"""

    actuatorIdentifier: Annotated[
        str,
        pydantic.Field(
            description="The id of the actuator that can execute this experiment or parameterized versions of it"
        ),
    ]
    identifier: Annotated[
        str,
        pydantic.Field(
            description="The name of the experiment. "
            "Must be unique in the scope of the catalog of this experiments actuator."
        ),
    ]
    metadata: Annotated[
        dict,
        pydantic.Field(
            default_factory=dict,
            description="Metadata about the experiment. Sufficient to track its source. "
            "Can be custom format per actuator.",
        ),
    ]
    targetProperties: Annotated[
        list[AbstractPropertyDescriptor | ConcretePropertyDescriptor],
        pydantic.Field(
            description="The target properties this experiment aims to measure "
            "(can be ConcreteProperty or AbstractProperty instances)"
        ),
    ]
    requiredProperties: Annotated[
        tuple[ObservedProperty | ConstitutiveProperty, ...],
        pydantic.Field(
            default_factory=tuple,
            frozen=True,
            description="The properties this experiment needs values of as inputs "
            "(ObservedProperty or ConstitutiveProperty)",
        ),
    ]
    deprecated: Annotated[
        bool,
        pydantic.Field(
            description="Marks whether an experiment is deprecated or not. Defaults to False."
        ),
    ] = False
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "version": importlib.metadata.version(distribution_name="ado-core")
        },
    )
    optionalProperties: Annotated[
        tuple[ConstitutiveProperty, ...],
        pydantic.Field(
            default_factory=tuple,
            frozen=True,
            description="The optional properties this experiment can take as input. "
            "Must have default values specified in parameterization",
        ),
    ]
    defaultParameterization: Annotated[
        tuple[ConstitutivePropertyValue, ...],
        pydantic.Field(
            default_factory=tuple,
            validate_default=True,
            frozen=True,
            description="Default values for the optional properties",
        ),
    ]
    version: Annotated[
        StrictSemVerStr | None,
        pydantic.Field(
            description=(
                "Algorithm version for this experiment following strict SemVer "
                "(MAJOR.MINOR.PATCH). MAJOR identifies the memoisation boundary: "
                "results from different major versions are never reused. MINOR covers "
                "backward-compatible extensions (new outputs/inputs). PATCH covers "
                "bug fixes and refactoring that do not change observable outputs."
            ),
        ),
    ] = None

    @property
    def major_version_identifier(self) -> str:
        """Return the major version identifier.

        For versioned experiments this is ``'{identifier}@v{major}'``,
        e.g. ``'solve_mip@v1'``.  For unversioned experiments (legacy) this
        is identical to :attr:`identifier`.

        Two experiments with the same major version identifier perform the same
        science — their results are interchangeable.

        Returns:
            The major version identifier string.
        """
        if self.version is not None:
            return f"{self.identifier}@v{semver_major(self.version)}"
        return self.identifier

    @property
    def fully_qualified_identifier(self) -> str:
        """Return the fully-qualified identifier encoding the exact version.

        For versioned experiments this is ``'{identifier}@{version}'``,
        e.g. ``'solve_mip@1.0.3'``.  For unversioned experiments this is
        identical to :attr:`identifier`.

        Returns:
            The fully-qualified identifier string.
        """
        if self.version is not None:
            return f"{self.identifier}@{self.version}"
        return self.identifier

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
        """Two experiments are equal when they share the same major version identifier.

        Experiments with the same base name and same major version are
        considered to perform the same science regardless of minor/patch
        version differences.

        Returns:
            True if both experiments have the same actuator and major version
            identifier.
        """
        if not isinstance(other, Experiment):
            return False
        if isinstance(other, ParameterizedExperiment):
            return False
        return (
            self.actuatorIdentifier == other.actuatorIdentifier
            and self.major_version_identifier == other.major_version_identifier
        )

    def __hash__(self) -> int:
        return hash((self.actuatorIdentifier, self.major_version_identifier))

    @classmethod
    def experimentWithAbstractPropertyIdentifiers(
        cls,
        identifier: str,
        actuatorIdentifier: str,
        targetProperties: [str],
        propertyType: MeasuredPropertyTypeEnum = MeasuredPropertyTypeEnum.MEASURED_PROPERTY_TYPE,
        requiredConstitutiveProperties: [str] = None,
        metadata: dict | None = None,
        deprecated: bool = False,
    ) -> "Experiment":
        """Factory method for creating an Experiment instance when you have a list of abstract property ids

        :param identifier: The name of the measurement
            Must be unique in the scope of this experiments actuator i.e. the actuator can use this id to identify
            the experiment to run.
        :param actuatorIdentifier: The id of the actuator that can execute this experiment.
            This id and its format just need to be common between the Experiment and its Actuator.
            The purpose is that an actuator can check if it "owns" this Experiment
        :param targetProperties: A list of the (abstract) identifiers of the properties this experiment will attempt to measure
        :param propertyType: The type of the properties.
        :param requiredConstitutiveProperties: A list of the identifiers of the constitutive properties this experiment requires values for as inputs
        :param metadata: Metadata about the experiment. Sufficient to track its source. Can be custom format per actuator
        :param deprecated: Marks an experiment as deprecated.
        """

        targetProperties = [
            AbstractPropertyDescriptor(identifier=t, propertyType=propertyType)
            for t in targetProperties
        ]
        if requiredConstitutiveProperties is not None:
            requiredConstitutiveProperties = [
                ConstitutiveProperty(identifier=t)
                for t in requiredConstitutiveProperties
            ]
        else:
            requiredConstitutiveProperties = []

        metadata = metadata if metadata is not None else {}

        return cls(
            identifier=identifier,
            actuatorIdentifier=actuatorIdentifier,
            targetProperties=targetProperties,
            requiredProperties=requiredConstitutiveProperties,
            metadata=metadata,
            deprecated=deprecated,
        )

    @pydantic.field_validator("optionalProperties")
    def validate_optional_properties(
        cls,
        optionalProperties: list[ConstitutiveProperty],
        values: "pydantic.FieldValidationInfo",
    ) -> list[ConstitutiveProperty]:

        # Check all optional properties have unique identifiers
        optional_properties_identifiers = {p.identifier for p in optionalProperties}
        if len(optional_properties_identifiers) != len(
            [p.identifier for p in optionalProperties]
        ):
            count = {}
            for p in optionalProperties:
                count[p.identifier] = count.get(p.identifier, 0) + 1

            # VV: Report just the optionalProperties with conflicting names
            duplicates = [
                p
                for p in optionalProperties
                if p.identifier in [k for k, v in count.items() if v > 1]
            ]
            raise ValueError(
                f"Optional properties provided containing duplicate identifiers: {duplicates}"
            )

        # Check no optional property is a required property
        required_properties_identifiers = {
            p.identifier for p in values.data.get("requiredProperties")
        }
        required_and_optional_properties_identifiers = (
            optional_properties_identifiers.intersection(
                required_properties_identifiers
            )
        )
        if len(required_and_optional_properties_identifiers) != 0:
            raise ValueError(
                "The following optional properties were also in the required properties: "
                f"{required_and_optional_properties_identifiers}"
            )

        return optionalProperties

    @pydantic.field_validator("defaultParameterization")
    def validate_default_parameterization(
        cls,
        value: list[ConstitutivePropertyValue],
        values: "pydantic.FieldValidationInfo",
    ) -> list[ConstitutivePropertyValue]:

        if not value:
            if values.data.get("optionalProperties"):
                raise ValueError(
                    "optionalProperties specified without parameterization"
                )
        else:
            if not values.data.get("optionalProperties"):
                raise ValueError(
                    "default parameterization specified without optionalProperties being specified"
                )
            try:
                check_parameterization_validity(
                    values.data.get("optionalProperties"), value
                )
            except pydantic.ValidationError as error:
                raise error
            else:
                # Check all optionalProperties are parameterized
                mapping = {c.property.identifier: c for c in value}
                isNotParameterized = [
                    v
                    for v in values.data.get("optionalProperties")
                    if mapping.get(v.identifier) is None
                ]
                if len(isNotParameterized) > 0:
                    raise ValueError(
                        f"optionalProperties do not have default parameterization. Missing: {[v.identifier for v in isNotParameterized]}"
                    )

        return value

    def __str__(self) -> str:

        return reference_string_from_fields(
            actuator_identifier=self.actuatorIdentifier,
            experiment_identifier=self.identifier,
        )

    def __rich__(self) -> "RenderableType":
        """Render this experiment using rich."""
        import rich.box
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        from orchestrator.utilities.rich import get_rich_repr

        fq_identifier = f"{self.actuatorIdentifier}.{self.fully_qualified_identifier}"
        content = [
            Text.assemble(
                ("Identifier: ", "bold"),
                (
                    fq_identifier,
                    "bold green",
                ),
                overflow="fold",
            )
        ]

        if self.version is not None:
            content.append(
                Text.assemble(
                    ("Version: ", "bold"),
                    (self.version, "green"),
                    overflow="fold",
                )
            )

        if self.metadata.get("description"):
            content.extend(
                [
                    Text.assemble(
                        ("Description: ", "bold"),
                        (self.metadata["description"], "italic"),
                        overflow="fold",
                        end="\n\n",
                    ),
                ]
            )

        content.append(Text())

        # Required Inputs section
        req_inputs = []

        # Constitutive Properties subsection
        const_props = []
        if self.requiredConstitutiveProperties:
            const_props = [
                Panel(c, box=rich.box.HORIZONTALS)
                for c in self.requiredConstitutiveProperties
            ]
        else:
            const_props = [Text("No required constitutive properties specified")]

        req_inputs.extend(
            [
                Text("Constitutive Properties:", style="bold"),
                Group(*const_props),
            ]
        )

        # Observed Properties subsection
        if self.requiredObservedProperties:
            obs_props = [Text(str(o)) for o in self.requiredObservedProperties]
            req_inputs.extend(
                [
                    Text("Observed Properties:", style="bold"),
                    Panel(
                        Group(*obs_props),
                        box=rich.box.SIMPLE,
                        padding=(0, 2),
                    ),
                ]
            )

        content.extend(
            [
                Text("Required Inputs:", style="bold"),
                Panel(
                    Group(*req_inputs),
                    box=rich.box.SIMPLE,
                    padding=(0, 2),
                ),
            ]
        )

        # Optional Inputs section
        if self.optionalProperties:
            opt_inputs = []
            mapping = {c.identifier: c for c in self.optionalProperties}
            for value in self.defaultParameterization:
                prop = mapping[value.property.identifier]
                opt_inputs.append(
                    Panel(
                        Group(
                            *[
                                prop,
                                Text("Default value:", style="bold", end=" "),
                                get_rich_repr(value.value),
                            ]
                        ),
                        box=rich.box.HORIZONTALS,
                    )
                )

            content.extend(
                [
                    Text("Optional Inputs and Default Values:", style="bold"),
                    Panel(
                        Group(*opt_inputs),
                        box=rich.box.SIMPLE_HEAVY,
                        padding=(0, 2),
                    ),
                ]
            )

        # Outputs section
        content.extend(
            [
                Text("Outputs:", style="bold"),
                Panel(
                    Group(
                        *[
                            Text(f"{c.identifier}", style="green")
                            for c in self.observedProperties
                        ]
                    ),
                    box=rich.box.HORIZONTALS,
                    padding=(0, 2),
                ),
            ]
        )

        return Group(*content)

    def isValidParameterization(
        self, parameterization: list[ConstitutivePropertyValue]
    ) -> bool:
        """Returns True if the list of values given by parameterization is valid, otherwise False"""

        try:
            check_parameterization_validity(
                list(self.optionalProperties), parameterization
            )
        except (ValueError, AssertionError) as error:
            print(error)
            retval = False
        else:
            retval = True

        return retval

    @property
    def reference(self) -> ExperimentReference:
        """Return an ExperimentReference for the receiver.

        The reference carries the experiment's algorithm version so that
        memoisation and comparison use the major version identifier automatically.

        Returns:
            An ExperimentReference for this experiment.
        """
        return ExperimentReference(
            experimentIdentifier=self.identifier,
            actuatorIdentifier=self.actuatorIdentifier,
            experimentVersion=self.version,
        )

    @property
    def observedProperties(self) -> list[ObservedProperty]:
        """Returns a list of ObservedProperty instances representing the properties measured by the receiver

        Note: New ObservedProperty objects are returned on each call"""

        return [
            ObservedProperty(targetProperty=target, experimentReference=self.reference)
            for target in self.targetProperties
        ]

    def observedPropertyForTargetIdentifier(
        self, targetIdentifier: str
    ) -> ObservedProperty | None:
        """Returns an ObservedProperty instance for the given targetIdentifier or None if none exists"""

        v = [
            p
            for p in self.observedProperties
            if p.targetProperty.identifier == targetIdentifier
        ]

        return None if len(v) == 0 else v[0]

    def hasTargetPropertyWithIdentifier(self, identifier: str) -> bool:
        """Returns True if the receiver has a target property called identifier"""

        v = [p for p in self.targetProperties if p.identifier == identifier]

        return len(v) != 0

    def hasTargetProperty(self, prop: Property) -> bool:
        """Returns True if  prop is one of the receivers target properties"""

        return self.hasTargetPropertyWithIdentifier(prop.identifier)

    def hasObservedPropertyWithIdentifier(self, identifier: str) -> bool:
        """Returns True if the receiver has a target property called identifier"""

        v = [p for p in self.observedProperties if p.identifier == identifier]

        return len(v) != 0

    def hasObservedProperty(self, prop: ObservedProperty) -> bool:
        """Returns True if  prop is one of the receivers observed properties"""

        return self.hasObservedPropertyWithIdentifier(prop.identifier)

    def has_same_base_as_experiment(self, otherExperiment: "Experiment") -> bool:
        """Returns True if the base experiment is the same as the base experiment of otherExperiment"""

        return self.identifier == otherExperiment.identifier

    def has_same_base_as_experiment_reference(
        self, reference: "ExperimentReference"
    ) -> bool:
        """Returns True if the base experiment is the same as the base experiment of reference"""

        return self.identifier == reference.experimentIdentifier

    def experimentProvidesRequirements(
        self, experiment: "Experiment", exactMatch: bool = True
    ) -> bool:
        """Returns True if experiment would measure ALL required observed properties of the receiver

        If the receiver has no required observed properties this method will always return False.

        This method handles required properties that are Observed or Abstract

        params:
            experiment: The experiment to check against
            exactMath: If True `experiment` must provide exactly the same property i.e. matching parameterization.
                If False `experiment` must measure the same base experiment as each required
                input (any algorithm version satisfies — the target metric is unchanged).
        """

        retval = True
        if len(self.requiredObservedProperties) == 0:
            retval = False
        else:
            if exactMatch:
                for req in self.requiredObservedProperties:
                    if not experiment.hasObservedProperty(req):
                        retval = False
                        break
            else:
                for input_ref in self.references_of_required_input_experiments:
                    if not (
                        experiment.actuatorIdentifier == input_ref.actuatorIdentifier
                        and experiment.identifier == input_ref.experimentIdentifier
                    ):
                        retval = False
                        break

        return retval

    def virtualObservedPropertyFromIdentifier(
        self, identifier: str
    ) -> VirtualObservedProperty | None:
        """Returns a list of VirtualObservedProperty instances that could be calculated by the receiver given a virtual property identifier

        A virtual property identifier has two parts - the base property identifier and the aggregation method identifier
        The base property identifier can be the identifier of an ObservedProperty or a target property.
        The aggregation method identifier must be one of the know aggregation method identifiers

        This method finds the observed properties of the receiver that match the base property identifier
        and creates virtual properties for them.
        A single VirtualObservedProperty is created in either case
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

        if not VirtualObservedProperty.isVirtualPropertyIdentifier(identifier):
            raise ValueError(f"{identifier} is not a valid virtual property identifier")

        propertyIdentifier, aggregationMethod = VirtualObservedProperty.parseIdentifier(
            identifier
        )
        aggregationMethod = PropertyAggregationMethod(
            identifier=PropertyAggregationMethodEnum(aggregationMethod)
        )

        # Check if it's an observed property
        ops = [o for o in self.observedProperties if o.identifier == propertyIdentifier]
        if len(ops) > 0:
            # Should only be one observed property with a given identifier
            op = ops[0]
            vp = VirtualObservedProperty(
                baseObservedProperty=op, aggregationMethod=aggregationMethod
            )
            if vp.identifier != identifier:
                raise ValueError("Mismatch between property identifiers")
            retval = vp
        else:
            # Not an observed property - Check if it's a target property
            ops = [
                o
                for o in self.observedProperties
                if o.targetProperty.identifier == propertyIdentifier
            ]
            if len(ops) > 0:
                # Should only be one target property with a given identifier
                op = ops[0]
                vp = VirtualObservedProperty(
                    baseObservedProperty=op, aggregationMethod=aggregationMethod
                )

                if (
                    vp.baseObservedProperty.targetProperty.identifier
                    != propertyIdentifier
                ):
                    raise ValueError(
                        "Mismatch between property identifiers "
                        f"{vp.baseObservedProperty.targetProperty.identifier} != {propertyIdentifier}"
                    )

                retval = vp
            else:
                # Not observed or target property - return None
                retval = None

        return retval

    @property
    def requiredConstitutiveProperties(self) -> list[ConstitutiveProperty]:
        """The constitutive properties an entity must have for the experiment to operate on it

        An empty list will be returned if the experiment did not define these.
        In this case you must read the experiment documentation to discover these properties
        """

        return [
            e for e in self.requiredProperties if isinstance(e, ConstitutiveProperty)
        ]

    @property
    def requiredObservedProperties(self) -> list[ObservedProperty]:
        """The observed properties an entity must have measured values for, for the experiment to operate on it

        An empty list will be returned if the experiment does not require any other measured property values.
        """

        return [e for e in self.requiredProperties if isinstance(e, ObservedProperty)]

    @property
    def optional_property_descriptors(self) -> set[ConstitutivePropertyDescriptor]:
        """Return identifiers of optional constitutive properties on this experiment."""
        return {prop.descriptor() for prop in self.optionalProperties}

    @property
    def required_constitutive_property_descriptors(
        self,
    ) -> set[ConstitutivePropertyDescriptor]:
        """Return identifiers of required constitutive properties on this experiment."""
        return {prop.descriptor() for prop in self.requiredConstitutiveProperties}

    def constitutive_property_for_identifier(
        self, property_identifier: str
    ) -> ConstitutiveProperty | None:
        """Return a declared constitutive property by identifier, if present."""

        return next(
            (
                p
                for p in (*self.requiredConstitutiveProperties, *self.optionalProperties)
                if p.identifier == property_identifier
            ),
            None,
        )

    @property
    def references_of_required_input_experiments(
        self,
    ) -> set[ExperimentReference]:
        """Returns references to the Experiments, this experiment requires to have been measured on an Entity before it can run"""

        return {op.experimentReference for op in self.requiredObservedProperties}

    def valueForOptionalProperty(
        self, property_identifier: str
    ) -> ConstitutivePropertyValue:
        """Returns the parameterized value of the optional property property_identifier

        Raises:
            ValueError, If property_identifier is not the id of any optionalProperty"""

        values = [
            v
            for v in self.defaultParameterization
            if v.property.identifier == property_identifier
        ]

        # There should be one value - if not there is an error
        if len(values) == 0:
            props = [
                p
                for p in self.optionalProperties
                if p.identifier == property_identifier
            ]
            if len(props) == 0:
                raise ValueError(
                    f"Experiment {self.identifier} has no optional property {property_identifier}."
                    f" Known optional properties: {self.optionalProperties}"
                )
            # pragma: nocover
            raise ValueError(
                f"Experiment {self.identifier} has no value for optional property {property_identifier}."
                f" This is inconsistent as all properties and must have values on construction. "
                f"Known optional properties: {self.optionalProperties}. "
                f"Set default values: {self.defaultParameterization}"
            )

        return values[0]

    def propertyValuesFromEntity(self, entity: "Entity", target: bool = False) -> dict:
        """Given an entity returns the values for the required and optional properties of the Experiment instance

        If a required property is an ObservedProperty it may have multiple values.
        This method will return all values if more than one
        i.e. the value of the property identifier in the returned dictionary will be a list.

        If there is only one value for the property the value of the property identifier in the returned dictionary will be that value

        Parameters:
            entity: An Entity
            target: If True observed properties will be added to a key with the target property name rather than the observed property name.
                e.g. if experiment A measurements property x, then with target=False, the key in the returned dictionary will be A.x
                     if target=True it will be 'x'.
                In the later case if the experiment depends on A.x and B.x both values will be added to 'x'

        Return:
            A dictionary of property identifier: property value pairs

        Raise:
            ValueError: If entity does not have a value for a required property of the Experiment instance
        """

        # 1. Get the values for required properties - these will be constitutive or observed properties of the entity
        identifierValueMap = {}
        for prop in self.requiredProperties:
            if isinstance(prop, ObservedProperty):
                values = entity.valuesForProperty(prop)
                ident = prop.targetProperty.identifier if target else prop.identifier
                if not values:
                    raise ValueError(
                        f"Entity {entity} has no value for required observed property {prop.identifier}"
                    )

                if len(values) > 1:
                    identifierValueMap[ident] = [v.value for v in values]
                else:
                    identifierValueMap[ident] = values[0].value
            else:

                try:
                    identifierValueMap[prop.identifier] = (
                        entity.valueForConstitutivePropertyIdentifier(
                            prop.identifier
                        ).value
                    )
                except AttributeError as error:
                    raise ValueError(
                        f"Entity {entity} has no value for required constitutive property {prop.identifier}"
                    ) from error

        # 2. Get the values for the optional properties
        # These may be in the Entity (because they were used to define the EntitySpace) or the Experiment
        for optionalProperty in self.optionalProperties:
            # Check if this property was added to the entity space - if it was the Entity value for the property takes precedence
            entityValue = entity.valueForConstitutivePropertyIdentifier(
                optionalProperty.identifier
            )
            if entityValue is not None:
                # Use the Entity value
                identifierValueMap[optionalProperty.identifier] = entityValue.value
            else:
                # Use the parameterized value
                identifierValueMap[optionalProperty.identifier] = (
                    self.valueForOptionalProperty(optionalProperty.identifier).value
                )

        return identifierValueMap

    def validate_entity(
        self,
        entity: "Entity",
        disallow_extra_properties: bool = False,
        verbose: bool = False,
    ) -> bool:
        """Returns True if Experiment can be applied to entity, false otherwise

        This method only checks constitutive properties.
        - The entity has valid values for all required properties of the experiment
        - The entity has valid values for any optional properties of the experiment it contains
        - If disallow_extra_properties is True all properties of the Entity must be
         properties (required+optional) of the experiment

        If verbose=True if entity is not valid, the reason will be printed to stderr
        """

        point = {
            v.property.identifier: v.value for v in entity.constitutive_property_values
        }

        #
        # Get required and optional property sets of the experiment
        #
        required_property_identifiers = {
            cp.identifier for cp in self.requiredConstitutiveProperties
        }
        optional_property_identifiers = {
            cp.identifier for cp in self.optionalProperties
        }

        #
        # Get the equivalent sets from the entity
        #
        required_properties_present = point.keys() & required_property_identifiers
        optional_properties_present = point.keys() & optional_property_identifiers
        additional_properties_present = (
            point.keys() - required_properties_present - optional_properties_present
        )

        # First check against strict optional as it is a quick fail condition
        if additional_properties_present and disallow_extra_properties:
            if verbose:
                print(
                    f"disallow_extra_properties is set and the following entity "
                    f"properties are not required or optional properties of {self.identifier}:"
                    f"{additional_properties_present} ",
                    file=sys.stderr,
                )
            return False

        # Check if all the required properties are present with values in domain
        if not validate_point_against_properties(
            point={k: v for k, v in point.items() if k in required_properties_present},
            constitutive_properties=self.requiredConstitutiveProperties,
            verbose=verbose,
        ):
            if verbose:
                print(
                    f"The entity is missing values for required properties of {self.identifier}: {required_property_identifiers - required_properties_present}"
                )
            return False

        # All required properties are there
        # Now check optional properties, if given
        # We can set partial_match=True because:
        # - If we wanted full match of optional properties (strict_optional), but it wasn't present,
        #   we would have already exited
        if optional_properties_present and not validate_point_against_properties(
            point={k: v for k, v in point.items() if k in optional_properties_present},
            constitutive_properties=list(self.optionalProperties),
            allow_partial_matches=True,
            verbose=verbose,
        ):
            print(
                f"The entity has properties that match optional properties"
                f"of {self.identifier} - "
                f"{optional_properties_present} - "
                f"but its values for those properties are not in the domain of the optional properties",
                file=sys.stderr,
            )
            return False

        return True


class ParameterizedExperiment(Experiment):
    """Represents an Experiment where default optional parameters have been overridden

    Note: The parameterization cannot be empty or have any values which are the same as default values
    """

    parameterization: Annotated[
        list[ConstitutivePropertyValue],
        pydantic.Field(
            default_factory=list, description="Values for optional properties"
        ),
    ]
    mapping: Annotated[
        dict,
        pydantic.Field(
            default_factory=dict, description="Private attribute", exclude=True
        ),
    ]
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def parameterized_optional_property_descriptors(
        self,
    ) -> set[ConstitutivePropertyDescriptor]:
        """Return identifiers of optional properties with custom parameterization."""
        return {value.property for value in self.parameterization}

    @property
    def major_version_parameterized_identifier(self) -> str:
        """Return the major version parameterized identifier.

        Uses the base experiment's :attr:`~Experiment.major_version_identifier`
        as the prefix so that the memoisation key encodes the major algorithm
        version.  For example: ``'solve_mip@v1-time_limit_s.3600'``.

        Returns:
            major version parameterized identifier string.

        Raises:
            ValueError: If parameterization is empty.
        """
        if not self.parameterization:
            raise ValueError(
                "Parameterization cannot be empty in a ParameterizedExperiment"
            )

        return identifier_for_parameterized_experiment(
            self.major_version_identifier, self.parameterization
        )

    def __eq__(self, other: object) -> bool:  # noqa: ANN401
        """ParameterizedExperiments are equal when they share the same major_version_parameterized_identifier.

        A ParameterizedExperiment can only be equal to another
        ParameterizedExperiment; it is never equal to its parent Experiment.

        Returns:
            True if both share the same actuator and parameterized identifier.
        """
        retval = False
        if isinstance(other, ParameterizedExperiment):
            retval = (self.actuatorIdentifier == other.actuatorIdentifier) and (
                self.major_version_parameterized_identifier
                == other.major_version_parameterized_identifier
            )

        return retval

    def __str__(self) -> str:

        return reference_string_from_fields(
            actuator_identifier=self.actuatorIdentifier,
            experiment_identifier=self.major_version_parameterized_identifier,
        )

    def __hash__(self) -> int:

        return hash(str(self))

    def __rich__(self) -> "RenderableType":
        """Render this parameterized experiment using rich."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        from orchestrator.utilities.rich import get_rich_repr

        content = []
        content.append(
            Text(
                f"Parameterized Identifier: {self.major_version_parameterized_identifier}"
            )
        )
        content.append(Text())

        # Include base experiment rendering
        content.append(super().__rich__())

        # Parameterization section
        param_content = []
        mapping = {c.identifier: c for c in self.optionalProperties}
        for value in self.parameterization:
            prop = mapping[value.property.identifier]
            param_content.extend(
                [
                    prop,
                    Text("Parameterized value:", style="bold", end=" "),
                    get_rich_repr({value.value}),
                    Text(),
                ]
            )

        content.extend(
            [Text("Parameterization:", style="bold"), Panel(Group(*param_content))]
        )

        return Group(*content)

    @pydantic.field_validator("parameterization")
    def validate_not_empty_parameterization(
        cls, parameterization: list[PropertyValue], values: "pydantic.ValidationInfo"
    ) -> list[PropertyValue]:

        # Check it's not empty - it is raise error as should use ParameterizedExperiment
        if not parameterization:
            raise ValueError("Custom parameterization cannot be empty")

        # Someone could try to initialize this object with a parameterization
        # but no optionalParameters or defaultParameterization
        # Hence we have to check here
        # We only need to check if one of the two is there as once it is the base
        # class validators will check for the other

        # Check there are optional properties to parameterize
        if not values.data.get("optionalProperties"):
            raise ValueError(
                "Cannot parameterize an experiment with no optionalProperties"
            )

        return parameterization

    @pydantic.model_validator(mode="after")
    def validate_parameterization(self) -> "ParameterizedExperiment":

        check_parameterization_validity(
            list(self.optionalProperties), self.parameterization
        )

        defaultParameterizationMap = {
            v.property.identifier: v for v in self.defaultParameterization
        }

        customParameterizationMap = {
            v.property.identifier: v for v in self.parameterization
        }

        # build the mapping private var
        for p in self.optionalProperties:
            customValue = customParameterizationMap.get(p.identifier)
            self.mapping[p.identifier] = (
                customValue
                if customValue is not None
                else defaultParameterizationMap[p.identifier]
            )

        # Now check that none of the parameterized values are the same as the defaults
        for v in self.defaultParameterization:
            if (
                v.property.identifier in customParameterizationMap
                and v.value == customParameterizationMap[v.property.identifier].value
            ):
                raise ValueError(
                    f"Default value {v.value} for property {v.property.identifier} is the same as the custom value, {customParameterizationMap[v.property.identifier].value}"
                )

        # Finally update identifier

        return self

    @property
    def reference(self) -> ExperimentReference:
        """Return an ExperimentReference for this parameterized experiment.

        Returns:
            An ExperimentReference carrying the parameterization and
            algorithm version.
        """
        return ExperimentReference(
            experimentIdentifier=self.identifier,
            actuatorIdentifier=self.actuatorIdentifier,
            parameterization=self.parameterization,
            experimentVersion=self.version,
        )

    def valueForOptionalProperty(
        self, property_identifier: str
    ) -> ConstitutivePropertyValue:
        """Returns the parameterized value of the optional property property_identifier

        Raises:
            ValueError, If property_identifier is not the id of any optionalProperty"""

        try:
            retval = self.mapping[property_identifier]
        except KeyError as error:
            raise ValueError(
                f"No optional property called {property_identifier}. Known optional properties {self.optionalProperties}"
            ) from error

        return retval


class ExperimentInterfaceIssueKind(str, enum.Enum):
    """Kinds of experiment interface mismatches detected by compatibility checks."""

    MISSING_REQUIRED_CONSTITUTIVE_IN_PROVIDED = (
        "missing_required_constitutive_in_provided"
    )
    EXTRA_REQUIRED_CONSTITUTIVE_IN_PROVIDED = "extra_required_constitutive_in_provided"
    MISSING_REQUIRED_OBSERVED_IN_PROVIDED = "missing_required_observed_in_provided"
    EXTRA_REQUIRED_OBSERVED_IN_PROVIDED = "extra_required_observed_in_provided"
    PARAMETERIZED_OPTIONAL_NOT_IN_PROVIDED = "parameterized_optional_not_in_provided"
    OPTIONAL_NOT_DECLARED_IN_PROVIDED = "optional_not_declared_in_provided"
    DOMAIN_NOT_COMPATIBLE = "domain_not_compatible"
    PARAMETERIZED_VALUE_OUT_OF_DOMAIN = "parameterized_value_out_of_domain"
    OPTIONAL_DEFAULT_MISMATCH = "optional_default_mismatch"
    OUTPUT_NOT_IN_PROVIDED = "output_not_in_provided"


class ExperimentInterfaceIssue(pydantic.BaseModel):
    """A structured description of an experiment interface mismatch."""

    kind: Annotated[
        ExperimentInterfaceIssueKind,
        pydantic.Field(description="The kind of interface mismatch detected"),
    ]
    identifier: Annotated[
        str | None,
        pydantic.Field(
            default=None,
            description=(
                "Property identifier related to the mismatch. Refers to a "
                "constitutive, observed, or target property depending on kind."
            ),
        ),
    ] = None
    value: Annotated[
        typing.Any | None,
        pydantic.Field(
            default=None,
            description="Parameterized value related to the mismatch",
        ),
    ] = None
    expectedDefault: Annotated[
        typing.Any | None,
        pydantic.Field(
            default=None,
            description="Default value declared by the expected experiment",
        ),
    ] = None
    providedDefault: Annotated[
        typing.Any | None,
        pydantic.Field(
            default=None,
            description="Default value declared by the provided experiment",
        ),
    ] = None

    model_config = ConfigDict(frozen=True, extra="forbid")


def _issues_for_ids(
    ids: Iterable[Property | PropertyDescriptor | ObservedProperty],
    kind: ExperimentInterfaceIssueKind,
) -> list[ExperimentInterfaceIssue]:
    """Build interface issues with ``identifier`` set for each id."""
    return [
        ExperimentInterfaceIssue(kind=kind, identifier=prop.identifier) for prop in ids
    ]


def _check_required_constitutive_interface(
    expected: Experiment | ParameterizedExperiment,
    provided: Experiment,
) -> list[ExperimentInterfaceIssue]:
    """Check required constitutive input compatibility.

    Missing required constitutive inputs are satisfied when the property is
    declared anywhere in the provided experiment (required or optional).
    Extra required inputs are those required by provided but not by expected.
    """

    return [
        *_issues_for_ids(
            expected.required_constitutive_property_descriptors
            - provided.required_constitutive_property_descriptors,
            ExperimentInterfaceIssueKind.MISSING_REQUIRED_CONSTITUTIVE_IN_PROVIDED,
        ),
        *_issues_for_ids(
            provided.required_constitutive_property_descriptors
            - expected.required_constitutive_property_descriptors,
            ExperimentInterfaceIssueKind.EXTRA_REQUIRED_CONSTITUTIVE_IN_PROVIDED,
        ),
    ]


def _check_required_observed_interface(
    expected: Experiment | ParameterizedExperiment,
    provided: Experiment,
) -> list[ExperimentInterfaceIssue]:
    """Check required observed input compatibility.

    Extra required observed inputs are those required by provided but not by
    expected.
    """
    expected_required = set(expected.requiredObservedProperties)
    provided_required = set(provided.requiredObservedProperties)
    missing = expected_required - provided_required
    extra = provided_required - expected_required
    return [
        *_issues_for_ids(
            missing,
            ExperimentInterfaceIssueKind.MISSING_REQUIRED_OBSERVED_IN_PROVIDED,
        ),
        *_issues_for_ids(
            extra,
            ExperimentInterfaceIssueKind.EXTRA_REQUIRED_OBSERVED_IN_PROVIDED,
        ),
    ]


def _check_parameterized_optional_presence(
    expected: Experiment | ParameterizedExperiment,
    provided: Experiment,
) -> list[ExperimentInterfaceIssue]:
    """Check that parameterized optional inputs exist in the provided experiment."""

    if isinstance(expected, ParameterizedExperiment):
        missing_ids = {
            p.property for p in expected.parameterization
        } - provided.optional_property_descriptors
        return _issues_for_ids(
            missing_ids,
            ExperimentInterfaceIssueKind.PARAMETERIZED_OPTIONAL_NOT_IN_PROVIDED,
        )

    return []


def _check_constitutive_domains(
    expected: Experiment | ParameterizedExperiment, provided: Experiment
) -> list[ExperimentInterfaceIssue]:
    """Check constitutive property presence and domain compatibility.

    Optional-not-declared is reported separately from parameterized-optional
    presence. Domain checks run only when the property is found in provided.
    """
    issues: list[ExperimentInterfaceIssue] = []
    constitutive_properties_to_check = list(
        expected.requiredConstitutiveProperties
    ) + list(expected.optionalProperties)
    for expected_property in constitutive_properties_to_check:
        provided_property = provided.constitutive_property_for_identifier(
            expected_property.identifier
        )
        if provided_property is None:
            if expected_property.descriptor() in expected.optional_property_descriptors:
                issues.append(
                    ExperimentInterfaceIssue(
                        kind=ExperimentInterfaceIssueKind.OPTIONAL_NOT_DECLARED_IN_PROVIDED,
                        identifier=expected_property.identifier,
                    )
                )
            continue

        if not expected_property.propertyDomain.isSubDomain(
            provided_property.propertyDomain
        ):
            issues.append(
                ExperimentInterfaceIssue(
                    kind=ExperimentInterfaceIssueKind.DOMAIN_NOT_COMPATIBLE,
                    identifier=expected_property.identifier,
                )
            )
    return issues


def _check_parameterized_values(
    expected: Experiment | ParameterizedExperiment, provided: Experiment
) -> list[ExperimentInterfaceIssue]:
    """Check that parameterized values fall within provided property domains."""
    if not isinstance(expected, ParameterizedExperiment):
        return []

    issues: list[ExperimentInterfaceIssue] = []
    for parameterized_value in expected.parameterization:
        provided_property = provided.constitutive_property_for_identifier(
            parameterized_value.property.identifier
        )
        if provided_property is None:
            continue
        if not provided_property.propertyDomain.valueInDomain(
            parameterized_value.value
        ):
            issues.append(
                ExperimentInterfaceIssue(
                    kind=ExperimentInterfaceIssueKind.PARAMETERIZED_VALUE_OUT_OF_DOMAIN,
                    identifier=parameterized_value.property.identifier,
                    value=parameterized_value.value,
                )
            )
    return issues


def _check_optional_defaults(
    expected: Experiment | ParameterizedExperiment, provided: Experiment
) -> list[ExperimentInterfaceIssue]:
    """Check non-parameterized optional default values match between experiments.

    Parameterized optionals are skipped because defaults come from
    parameterization instead.
    """
    issues: list[ExperimentInterfaceIssue] = []
    optional_to_check = expected.optional_property_descriptors
    if isinstance(expected, ParameterizedExperiment):
        optional_to_check = (
            optional_to_check - expected.parameterized_optional_property_descriptors
        )

    for optional_property in optional_to_check:
        if optional_property not in provided.optional_property_descriptors:
            continue
        expected_default = expected.valueForOptionalProperty(
            optional_property.identifier
        ).value
        provided_default = provided.valueForOptionalProperty(
            optional_property.identifier
        ).value
        if expected_default != provided_default:
            issues.append(
                ExperimentInterfaceIssue(
                    kind=ExperimentInterfaceIssueKind.OPTIONAL_DEFAULT_MISMATCH,
                    identifier=optional_property.identifier,
                    expectedDefault=expected_default,
                    providedDefault=provided_default,
                )
            )
    return issues


def _check_target_outputs(
    expected: Experiment | ParameterizedExperiment, provided: Experiment
) -> list[ExperimentInterfaceIssue]:
    """Check that expected outputs are a subset of provided outputs."""
    return _issues_for_ids(
        set(expected.targetProperties) - set(provided.targetProperties),
        ExperimentInterfaceIssueKind.OUTPUT_NOT_IN_PROVIDED,
    )


def check_experiment_interface_compatible(
    expected_experiment: "Experiment",
    provided_experiment: "Experiment",
) -> list[ExperimentInterfaceIssue]:
    """Check that a provided experiment supports an expected experiment interface.

    Verifies that inputs declared in the expected experiment are supported by the
    provided experiment, that provided-only required inputs are also declared in
    the expected experiment, and that expected outputs are a subset of provided
    outputs. Collects all mismatches rather than stopping at the first failure.

    Checks performed (in order):

    1. ``missing_required_constitutive_in_provided`` — expected required
       constitutive input not declared in provided (required or optional).
    2. ``extra_required_constitutive_in_provided`` — provided requires a
       constitutive input that expected does not require.
    3. ``missing_required_observed_in_provided`` — expected required observed
       input not required in provided.
    4. ``extra_required_observed_in_provided`` — provided requires an observed
       input that expected does not require.
    5. ``parameterized_optional_not_in_provided`` — parameterized optional not
       declared optional in provided.
    6. ``optional_not_declared_in_provided`` — expected optional not declared
       anywhere in provided.
    7. ``domain_not_compatible`` — expected property domain not a subdomain of
       provided property domain.
    8. ``parameterized_value_out_of_domain`` — parameterized value outside
       provided property domain.
    9. ``optional_default_mismatch`` — non-parameterized optional defaults differ.
    10. ``output_not_in_provided`` — expected output not produced by provided.

    Args:
        expected_experiment: Experiment declaring the required interface.
        provided_experiment: Experiment declaring the supported interface.

    Returns:
        A list of structured interface issues. An empty list means compatibility.
    """
    return [
        *(
            _check_required_constitutive_interface(
                expected=expected_experiment, provided=provided_experiment
            )
        ),
        *(
            _check_required_observed_interface(
                expected=expected_experiment, provided=provided_experiment
            )
        ),
        *(
            _check_parameterized_optional_presence(
                expected=expected_experiment, provided=provided_experiment
            )
        ),
        *(
            _check_constitutive_domains(
                expected=expected_experiment, provided=provided_experiment
            )
        ),
        *(
            _check_parameterized_values(
                expected=expected_experiment, provided=provided_experiment
            )
        ),
        *(
            _check_optional_defaults(
                expected=expected_experiment, provided=provided_experiment
            )
        ),
        *(
            _check_target_outputs(
                expected=expected_experiment, provided=provided_experiment
            )
        ),
    ]


def experiment_type_discriminator(experiment: typing.Any) -> str:  # noqa: ANN401

    if isinstance(experiment, ParameterizedExperiment):
        return "Parameterized"
    if isinstance(experiment, Experiment):
        return "Base"
    if isinstance(experiment, dict):
        if experiment.get("parameterization", None):
            return "Parameterized"
        return "Base"

    raise ValueError(
        f"Unable to determine experiment type for experiment: {experiment}"
    )


ExperimentType = Annotated[
    Annotated[Experiment, pydantic.Tag("Base")]
    | Annotated[ParameterizedExperiment, pydantic.Tag("Parameterized")],
    pydantic.Discriminator(experiment_type_discriminator),
]
