# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import typing

import pydantic

from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.schema.entity import (
    CheckRequiredObservedPropertyValuesPresent,
    Entity,
)
from orchestrator.schema.entityspace import (
    EntitySpaceRepresentation,
)
from orchestrator.schema.experiment import (
    Experiment,
    ExperimentType,
    ParameterizedExperiment,
)
from orchestrator.schema.observed_property import ObservedProperty
from orchestrator.schema.property import AbstractPropertyDescriptor
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest
from orchestrator.schema.virtual_property import VirtualObservedProperty
from orchestrator.utilities.logging import configure_logging
from orchestrator.utilities.rich import render_to_string

if typing.TYPE_CHECKING:
    from rich.console import RenderableType

configure_logging()

moduleLogger = logging.getLogger("SpaceModule")


class MeasurementSpaceConfiguration(pydantic.BaseModel):
    """A standalone representation of a measurement space

    This configuration does not require external actuators to provide experiment details
    """

    experiments: list[ExperimentType]

    @property
    def observedProperties(self) -> list[ObservedProperty]:

        import itertools

        return list(itertools.chain(*[e.observedProperties for e in self.experiments]))


class MeasurementSpace:

    @classmethod
    def measurementSpaceFromSelection(
        cls,
        selectedExperiments: list[ExperimentReference],
        experimentCatalogs: list[ExperimentCatalog] | None = None,
    ) -> "MeasurementSpace":
        """
        A class method to create a MeasurementSpace that uses the actuator registry to find the selected experiments.
        This is useful as it is easier to let user specify experiments in an abbreviated form and then
        leverage the registry.

        Note: programmatically it is easier to create a MeasurementSpace directly using a MeasurementSpaceConfiguration
        as it removes dependencies on the actuators.

        experimentReferences: A list of Experiments.
            All properties of experiments in this list are used for the measurement space.
            If you only want to use some properties of use the observedProperties parameter
            Note: If you pass the experiment and then a selection of its properties, all the properties will be
            used
        experimentCatalogs: A list of ExperimentCatalog instances.
            These will be searched for the experiments used to measure the properties
            in addition to default catalogs.
            Use this to pass external catalogs i.e. catalogs containing experiments there is no actuator for
        """

        # Validate parameterization for the provided experiment references
        for experiment_reference in selectedExperiments:
            experiment_reference.validate_parameterization()

        log = logging.getLogger("measurement-space-from-selection")
        experiments = []
        log.debug(f"processing {selectedExperiments}")

        import orchestrator.modules.actuators.registry

        log.debug("Getting global registry")
        globalRegistry = (
            orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        )
        log.debug(f"Got global registry {globalRegistry}")

        # First get all the experiments
        processedReferences = (
            []
        )  # Keeps track of what references have been examined. Prevents include experiment with multiple properties multiple times
        for ref in selectedExperiments:
            log.debug(f"looking for experiment {ref}")
            try:
                experiment = globalRegistry.experimentForReference(
                    ref, experimentCatalogs
                )  # type: Experiment
            except (
                orchestrator.modules.actuators.registry.UnknownExperimentError,
                orchestrator.modules.actuators.registry.UnknownActuatorError,
            ):
                raise
            else:
                if ref.parameterization:
                    experiment = ParameterizedExperiment(
                        parameterization=ref.parameterization, **experiment.model_dump()
                    )

                experiments.append(experiment)
                processedReferences.append(ref)

            log.debug(experiments)

        # Validate and apply parameterization to the experiments

        # Now create the measurement space from the properties
        log.debug("Experiments are:")
        for experiment in experiments:
            log.debug(
                f"{experiment}. {[p.identifier for p in experiment.targetProperties]}"
            )

        return MeasurementSpace(
            configuration=MeasurementSpaceConfiguration(experiments=experiments)
        )

    @classmethod
    def measurementSpaceFromExperimentReferences(
        cls,
        experimentReferences: list[str | ExperimentReference],
    ) -> "MeasurementSpace":
        """
        Class method for creating a MeasurementSpace from a list of experiment references.

        experimentReferences can be a list of
        - ExperimentReference objects
        - string representations of ExperimentReferences. These are strings like {actuator id}.{experiment name}
        - a mixture of both

        """

        stringRepresentations = [
            r for r in experimentReferences if not isinstance(r, ExperimentReference)
        ]
        referenceModels = [
            r for r in experimentReferences if isinstance(r, ExperimentReference)
        ]

        references = [
            ExperimentReference.referenceFromString(x) for x in stringRepresentations
        ] + referenceModels

        return cls.measurementSpaceFromSelection(selectedExperiments=references)

    def __init__(self, configuration: MeasurementSpaceConfiguration) -> None:
        """
        configuration: A MeasurementSpaceConfiguration object describing the space.
        """

        self.log = logging.getLogger("measurement-space")

        self._observedProperties = configuration.observedProperties
        self._experiments = configuration.experiments
        self._experimentReferences = [e.reference for e in self._experiments]

        self.log.debug(
            f"Observed properties in measurement space are "
            f"{[op.identifier for op in self._observedProperties]}"
        )

    def __rich__(self) -> "RenderableType":
        """Render this measurement space using rich."""
        import pandas as pd
        import rich.box
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text

        from orchestrator.utilities.rich import dataframe_to_rich_table

        content = []

        # Experiments overview table
        data = [[e.reference, not e.deprecated] for e in self.experiments]
        df = pd.DataFrame(data, columns=["experiment", "supported"])
        content.extend(
            [
                Text("Experiments:", style="bold"),
                Panel(dataframe_to_rich_table(df), box=rich.box.SIMPLE_HEAD),
            ]
        )

        # Detailed experiment info
        for e in self.experiments:
            exp_content = []

            # Inputs table
            data = [
                [p.identifier, "required", None, "na"] for p in e.requiredProperties
            ]
            data += [
                [
                    p.identifier,
                    "optional",
                    e.valueForOptionalProperty(p.identifier).value,
                    (
                        e.valueForOptionalProperty(p.identifier)
                        not in e.defaultParameterization
                    ),
                ]
                for p in e.optionalProperties
            ]
            df = pd.DataFrame(
                data, columns=["parameter", "type", "value", "parameterized"]
            )
            exp_content.extend(
                [
                    Text("Inputs:", style="bold"),
                    Panel(dataframe_to_rich_table(df), box=rich.box.SIMPLE_HEAD),
                ]
            )

            # Outputs table
            data = [[op.targetProperty.identifier] for op in e.observedProperties]
            df = pd.DataFrame(data, columns=["target property"])
            exp_content.extend(
                [
                    Text("Outputs:", style="bold"),
                    Panel(dataframe_to_rich_table(df), box=rich.box.SIMPLE_HEAD),
                ]
            )

            content.extend(
                [
                    Panel(
                        Group(*exp_content),
                        title=Text(str(e.reference), style="bold green"),
                        box=rich.box.HORIZONTALS,
                    ),
                    Text(),
                ]
            )

        return Group(*content)

    @property
    def selfContainedConfig(
        self,
    ) -> MeasurementSpaceConfiguration:

        return MeasurementSpaceConfiguration(experiments=self.experiments)

    @property
    def experimentReferences(
        self,
    ) -> list[ExperimentReference]:

        return self._experimentReferences.copy()

    @property
    def experiments(self) -> list[Experiment | ParameterizedExperiment]:

        return self._experiments.copy()

    @property
    def supported_experiments(self) -> list[Experiment]:
        return [e for e in self._experiments if not e.deprecated]

    @property
    def deprecated_experiments(self) -> list[Experiment]:
        return [e for e in self._experiments if e.deprecated]

    @property
    def has_deprecated_experiments(self) -> bool:
        return any(e.deprecated for e in self._experiments)

    @property
    def independentExperiments(self) -> list[Experiment]:
        """Returns experiments in the measurement space that do not depend on others"""

        return [e for e in self._experiments if len(e.requiredObservedProperties) == 0]

    @property
    def dependentExperiments(self) -> list[Experiment]:
        """Returns experiments in the measurement space that depend on others"""

        return [e for e in self._experiments if len(e.requiredObservedProperties) > 0]

    @property
    def isConsistent(self) -> bool:
        """Returns True if the measurement space contains all properties required for all measurements

        That is, for every experiment in the space depending on others, those experiments (or parameterized versions of them) are in the space.

        Important: The presence of an experiment required by another experiment, is determined by matching base experiment identifiers.
        For example, if "experiment_req" is required by another experiment, presence of any parameterization of experiment-req is consistent
        e.g. experiment_req-optionalParam.customValue would match
        """

        missingDependencies = []
        known_base_experiments = [
            r.experimentIdentifier for r in self._experimentReferences
        ]
        for e in self.dependentExperiments:
            missing = [
                d.experimentReference.experimentIdentifier
                for d in e.requiredObservedProperties
                if d.experimentReference.experimentIdentifier
                not in known_base_experiments
            ]
            if len(missing) > 0:
                self.log.warning(
                    f"Experiment {e} depends on {missing} but no parameterizations of these experiments "
                    f"(default or custom) are present in this space"
                )
                missingDependencies.extend(missing)

        return len(missingDependencies) == 0

    def propertyWithIdentifierInSpace(
        self,
        identifier: str,
        format: typing.Literal["any", "target", "observed"] = "any",
    ) -> bool:
        """Returns True if the space contains a property with the given identifier

        Args:
            identifier: The property identifier to check
            format: The format to check - "any", "target", or "observed"
                - "any" (default): Checks both target and observed properties, plus virtual properties
                - "target": Only checks target properties
                - "observed": Only checks observed properties and virtual properties based on observed

        Returns:
            bool: True if the identifier is found in the specified format, False otherwise

        Note:
            Virtual property identifiers are only checked when format is "observed" or "any"
        """
        # Build set of property identifiers to check based on format
        identifiers_to_check: set[str] = set()
        if format in {"target", "any"}:
            identifiers_to_check.update(
                {op.targetProperty.identifier for op in self.observedProperties}
            )

        if format in {"observed", "any"}:
            identifiers_to_check.update(
                {op.identifier for op in self.observedProperties}
            )

        # Check if identifier is in the set
        if identifier in identifiers_to_check:
            return True

        # If not found and format is any/observed, check if it's a virtual property
        if format in {"any", "observed"}:
            try:
                prop, _ = VirtualObservedProperty.parseIdentifier(identifier)
            except ValueError:
                return False
            else:
                return prop in identifiers_to_check

        return False

    def dependentExperimentsThatCanBeAppliedToEntity(
        self, entity: Entity, excludeApplied: bool = True
    ) -> list[Experiment]:
        """
        Returns a list of dependent Experiments which can be applied to entity given its currently known properties

        :param entity: An Entity instance
        :param excludeApplied: If True Experiments which have already been applied to entity are not returned

        :return: A list of the dependent Experiment instances that can applied to Entity
        If the space has no dependent Experiment this method will always return an empty list
        """

        experiments = []
        for d in self.dependentExperiments:
            if CheckRequiredObservedPropertyValuesPresent(entity, d, exactMatch=False):
                if len(entity.propertyValuesFromExperiment(d)) == 0:
                    self.log.debug(
                        f"Experiment {d.identifier} "
                        f"can be applied to {entity} and has not been calculated"
                    )
                    experiments.append(d)
                else:
                    self.log.debug(
                        f"Experiment {d.identifier} has already been applied to {entity}"
                    )
                    if excludeApplied is not True:
                        experiments.append(d)
            else:
                self.log.debug(
                    f"Can not apply experiment {d.identifier} to {entity} "
                    f"as required input data missing"
                )

        return experiments

    def dependentExperimentsThatCanBeAppliedAfterMeasurementRequest(
        self, measurementRequest: MeasurementRequest
    ) -> dict[Experiment, list[Entity]]:
        """
        Returns information about which experiments can be applied to which entities based on the results of a
        measurementRequest

        Which Experiments can be applied depends on if an Entity has values for the required properties.
        Note, this does not consider if a result for this experiment for an entity already exists.

        :param measurementRequest: An MeasurementRequest instance

        :return: A dictionary whose keys are Experiment instances
        and whose values are list of entities that experiment can be applied to
        """

        inputExperimentReference = measurementRequest.experimentReference
        self.log.debug(f"Input Experiment Reference {inputExperimentReference}.")

        inputExperiment = self.experimentForReference(inputExperimentReference)

        entities = measurementRequest.entities
        experimentsMap = {}
        for d in self.dependentExperiments:
            # Check if the inputExperiment measures the required properties of this experiment
            if d.experimentProvidesRequirements(inputExperiment, exactMatch=False):
                for entity in entities:
                    if CheckRequiredObservedPropertyValuesPresent(
                        entity, d, exactMatch=False
                    ):
                        if len(entity.propertyValuesFromExperiment(d)) == 0:
                            self.log.info(
                                f"Experiment {d.identifier} "
                                f"can be applied based on {measurementRequest} to {entity}. No results exist for this experiment"
                            )
                        else:
                            self.log.debug(
                                f"Experiment {d.identifier} "
                                f"can be applied to {entity} based on {measurementRequest}. A result already exists for this experiment "
                                f"has already completed"
                            )

                        if experimentsMap.get(d) is None:
                            experimentsMap[d] = []

                        experimentsMap[d].append(entity)

                    else:
                        self.log.info(
                            f"Can not calculate Experiment {d.identifier} for {entity} "
                            f"as required input property values are not present in entity"
                        )
                        for p in d.requiredObservedProperties:
                            if entity.valueForProperty(p) is None:
                                self.log.debug(f"Value for property {p} is not present")
            else:
                if len(d.requiredObservedProperties) > 0:
                    self.log.info(
                        f"Can not calculate Experiment {d.identifier} "
                        f"based on {measurementRequest} as the experiment {measurementRequest.experimentReference} does not provide all required data"
                    )

        return experimentsMap

    def compatibleEntitySpace(self) -> EntitySpaceRepresentation:
        """Returns an entity space compatible with the experiments"""

        # Note: the constitutive properties are not hashable due to ProbabilityFunction
        cps = []
        for e in self.experiments:
            cps.extend(e.requiredConstitutiveProperties)

        # TODO: Different experiments may have different domains for same target
        # This method should take that into account
        # Here if there are multiple we will just be using the domain of the last one
        mapping = {cp.identifier: cp for cp in cps}

        cp_ids = list(set(mapping.keys()))
        cps = [mapping[i] for i in cp_ids]
        return EntitySpaceRepresentation(constitutiveProperties=cps)

    def checkEntitySpaceCompatible(
        self, entitySpace: "EntitySpaceRepresentation", strict: bool = True
    ) -> bool:
        """Checks if all required experiment inputs are in the entity space

        If strict is True also checks that all entitySpace dimensions are required for at least one experiment
        i.e. there are no redundant dimensions

        raises a ValueError on first identified issue"""

        retval = True
        for e in self.experiments:
            for cp in e.requiredConstitutiveProperties:
                # Using normal equivalence (in, ==) ConstitutiveProperty will compare the domains
                # Here we just want to check the identifiers
                if cp.identifier not in [
                    p.identifier for p in entitySpace.constitutiveProperties
                ]:
                    raise ValueError(
                        f"Identified a measurement space constitutive property not in entity space: {cp}. "
                        f"Entity space:{render_to_string(entitySpace)}"
                    )
                if cp.propertyDomain:
                    # Check the entity spaces domain for the CP is compatible with the experiments
                    entitySpaceCP = entitySpace.propertyWithIdentifier(cp.identifier)
                    try:
                        if not entitySpaceCP.propertyDomain.isSubDomain(
                            cp.propertyDomain
                        ):
                            raise ValueError(
                                "Identified an entity space dimension not compatible with the measurement space requirements."
                                f"\nMeasurement Space Property: {render_to_string(cp)}"
                                f"\nEntity Space Dimension: {render_to_string(entitySpaceCP)}"
                            )
                    except Exception as error:
                        print(error)
                        print(f"The experiment property was: {render_to_string(cp)}")
                        print(
                            f"The entity space property was: {render_to_string(entitySpaceCP)}"
                        )
                        raise

            # Check if any of the optional properties are in the entity space
            for cp in e.optionalProperties:
                if cp.propertyDomain:
                    entitySpaceCP = entitySpace.propertyWithIdentifier(cp.identifier)
                    if entitySpaceCP is not None:
                        # Check the entity spaces domain for the CP is compatible with the experiments
                        if not entitySpaceCP.propertyDomain.isSubDomain(
                            cp.propertyDomain
                        ):
                            raise ValueError(
                                "Identified an entity space dimension not compatible with the measurement space requirements."
                                f"\nMeasurement Space Property: {render_to_string(cp)}"
                                f"\nEntity Space Dimension: {render_to_string(entitySpaceCP)}"
                            )

                        # Check that this property does not also have a custom parameterization
                        if isinstance(
                            e, ParameterizedExperiment
                        ) and entitySpaceCP.identifier in [
                            v.property.identifier for v in e.parameterization
                        ]:
                            raise ValueError(
                                f"Identified an entity space dimension, {entitySpaceCP}, that also has a custom parameterization in the measurement space. "
                                f"It is inconsistent for a property to have a custom parameterization in the measurement space and also be a dimension of the entityspace.\n"
                                f"The experiment with the custom parameterization is:\n{render_to_string(e)} "
                            )

        if strict:
            required_input_properties = [
                prop.identifier
                for e in self.experiments
                for prop in e.requiredConstitutiveProperties
            ]
            optional_input_properties = [
                prop.identifier
                for e in self.experiments
                for prop in e.optionalProperties
            ]
            all_props = required_input_properties + optional_input_properties
            for cp in entitySpace.constitutiveProperties:
                if cp.identifier not in all_props:
                    raise ValueError(
                        f"Identified an entity space dimension, {cp}, that is not required for any experiment in the measurement space and is hence redundant {all_props}"
                    )

        return retval

    def observedPropertiesForExperimentReference(
        self,
        experimentReference: ExperimentReference,
    ) -> list[ObservedProperty]:
        """Returns a list of observed properties in the receiver associated with the reference"""
        return [
            op
            for op in self._observedProperties
            if op.experimentReference == experimentReference
        ]

    def observedPropertiesForExperiment(
        self, experiment: Experiment
    ) -> list[ObservedProperty]:
        """Returns a list of observed properties in the receiver measured by the experiment

        IMPORTANT: This may be a subset of the properties of experiment"""

        return self.observedPropertiesForExperimentReference(experiment.reference)

    @property
    def targetProperties(
        self,
    ) -> list[AbstractPropertyDescriptor]:
        """Returns a list of AbstractProperties in the measurement space"""

        # If more that one observed property measures same target we just need to
        # keep one of them
        mapping = {
            ap.identifier: ap for exp in self.experiments for ap in exp.targetProperties
        }
        target_ids = list(set(mapping.keys()))

        return [mapping[i] for i in target_ids]

    @property
    def observedProperties(
        self,
    ) -> list[ObservedProperty]:
        """Returns a list of ObservedProperties in the measurement space"""

        return self._observedProperties.copy()

    def experimentForReference(self, reference: ExperimentReference) -> Experiment:

        s = [e for e in self._experiments if e.reference == reference]
        return s[0] if len(s) > 0 else None

    def numberExperimentsApplied(self, entity: Entity) -> int:
        """Returns the number of experiments in the MeasurementSpace which have been applied to entity"""

        count = 0
        for experiment in self.experiments:
            # If the entity has at least one measured value from this experiment then it has been applied to it
            measuredValues = entity.propertyValuesFromExperimentReference(
                experiment.reference
            )
            if len(measuredValues) > 0:
                count += 1

        return count

    def __str__(self) -> str:

        return (
            f"Measurement space consisting of {len(self._observedProperties)} properties "
            f"from {len(self._experiments)} experiments"
        )
