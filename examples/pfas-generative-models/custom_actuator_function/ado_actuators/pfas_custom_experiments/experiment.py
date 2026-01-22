# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging

import orchestrator.schema.entity
import orchestrator.schema.experiment
import orchestrator.schema.property_value
from orchestrator.schema.observed_property import ObservedPropertyValue

moduleLog = logging.getLogger()

"""
Custom experiments must follow this signature

def $CUSTOM_EXPERIMENT_NAME(entity: orchestrator.model.data.Entity, experiment:Experiment) -> [PropertyValue])

entity: The Entity provides access to the measured properties. The function will not be called
unless all the properties defined for it in the configuration YAML have been calculated.
experiment: The Experiment represent an experiment defined in custom_experiments.yaml that is set to call this function

Each custom experiment can calculate multiple values.
Return the values a list of PropertyValue objectsconnected to the correct ObservedProperty in the entity.
Use entity.observedPropertyForExperimentAndIdentifier() to get it

def mycustomexperiment(entity):

    #Do the calculation for candidate_score

    #Package the results for ado
    property = entity.observedPropertyForExperimentAndTarget('mycustomexperiment', 'candidate_score)
    value = PropertyValue(value, property, NUMERIC_VALUE_TYPE)

    return [value]

"""


def acid_test(
    entity: orchestrator.schema.entity.Entity,
    experiment: orchestrator.schema.experiment.Experiment,
    parameters: dict | None = None,
) -> list[ObservedPropertyValue]:
    """

    :param entity: The entity to be measured
    :param experiment: The Experiment object representing the exact Experiment to perform
        Required as multiple experiments can measure this property
    :param parameters:
    :return:
    """

    # Get the name of the observed property
    moduleLog.debug(f"Measuring {entity} with {experiment}")

    try:
        pkaObserved = next(
            p
            for p in experiment.requiredObservedroperties
            if p.targetProperty.identifier == "pka"
        )
    except KeyError:
        moduleLog.warning(
            f"Inconsistency detected: Objective Function experiment {experiment} \
                calling acid_test is expected to define a required property whose target-property is pKa and it does not"
        )
        raise

    # Get its value
    pka = entity.valueForProperty(pkaObserved)

    moduleLog.debug(f"Entity {entity}. Pka is {pka}. Value type {type(pka.value)}")

    value = 0 if pka.value >= 0 else 1

    # Create the result
    isAcidProp = next(
        p
        for p in experiment.observedProperties
        if p.targetProperty.identifier == "isAcid"
    )
    pv = ObservedPropertyValue(
        value=value,
        property=isAcidProp,
        valueType=orchestrator.schema.property_value.ValueTypeEnum.NUMERIC_VALUE_TYPE,
    )
    return [pv]
