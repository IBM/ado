# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

from orchestrator.core import DataContainerResource
from orchestrator.core.datacontainer.resource import DataContainer
from orchestrator.core.discoveryspace.config import EntityFilter
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    FunctionOperationInfo,
    OperatorFunctionConf,
)
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.core.samplestore.sql import SQLSampleStore
from orchestrator.modules.operators.collections import modify_operation
from orchestrator.schema.domain import PropertyDomain
from orchestrator.schema.entityspace import EntitySpaceRepresentation
from orchestrator.schema.property import (
    ConstitutiveProperty,
    NonMeasuredPropertyTypeEnum,
)
from orchestrator.schema.reference import ExperimentReference


class RifferlaParameters(pydantic.BaseModel):
    model_config = ConfigDict(extra="forbid")

    failed_metric: Annotated[
        str,
        pydantic.Field(
            description="The target property that indicates measurement validity"
        ),
    ]
    failed_value: Annotated[
        typing.Any,
        pydantic.Field(
            description="The value of the target property that indicates if a measurement was invalid"
        ),
    ]
    metric: Annotated[
        str, pydantic.Field(description="The target property that we are refining on")
    ]
    experiment_identifier: Annotated[
        str | None,
        pydantic.Field(
            description="The name of the experiment measuring the target property. "
            "If None the first experiment found measuring the property is used ",
        ),
    ] = None
    actuator_identifier: Annotated[
        str | None,
        pydantic.Field(
            description="The name of the actuator that implements experiment_identifier "
            "If experiment_identifier is not given this field is not used",
        ),
    ] = None
    mode: Annotated[str, pydantic.Field(description="The refinement mode. min or max")]
    min_mi_threshold: Annotated[
        float, pydantic.Field(description="The min mi threshold")
    ] = 0.8
    find_valid_intersection: Annotated[
        bool, pydantic.Field(description="If True tries to find a valid intersection")
    ] = False
    ignore_columns: Annotated[
        list,
        pydantic.Field(
            default_factory=lambda: ["identifier", "generatorid"],
            description="List of constitutive properties not to consider",
        ),
    ]
    entity_filter: Annotated[
        EntityFilter, pydantic.Field(description="What entities should be used")
    ] = EntityFilter.SAMPLED

    @classmethod
    def defaultOperation(cls) -> DiscoveryOperationConfiguration:
        return DiscoveryOperationConfiguration(
            module=OperatorFunctionConf(
                operatorName="rifferla",
                operationType=DiscoveryOperationEnum.MODIFY,
            ),
            parameters=cls(
                failed_metric="is_valid",
                failed_value=1,
                metric="train_tokens_per_second",
                mode="max",
            ),
        )


@modify_operation(
    name="rifferla",
    configuration_model=RifferlaParameters,
    description="Refines a space to produce one that is denser in entities that have min/max of a given observed property."
    "It does this by identifying which entity space dimensions should be fixed to set values, which explored, and setting range limits for those dimensions. "
    "The method leverages Mutual Information to identify dimensions correlated with the desired observed property.",
    configuration_model_default=RifferlaParameters.defaultOperation(),
)
def rifferla(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **parameters: object,
) -> OperationOutput:
    """
    This function assumes the given space *was* already sampled using LHU_sampler. This operation then analyzes the result
    and returns the new space and also the values for the non-selected dimensions.
    """

    config = RifferlaParameters.model_validate(parameters)

    import pandas as pd

    from .space_analysis import (
        calculate_mutual_information,
        get_valid_value_ranges,
        mi_pareto_selection,
    )

    if not isinstance(
        discoverySpace.entitySpace,
        EntitySpaceRepresentation,
    ):
        raise ValueError("Rifferla can only operate on discovery spaces ")

    # If not experiment identifier is given try to figure it out - required for any multi-experiment measurement space
    if config.experiment_identifier is None:
        print("No experiment reference given - will find first matching")
        op = None
        reference = None
        for experiment in discoverySpace.measurementSpace.experiments:
            op = experiment.observedPropertyForTargetIdentifier(config.metric)
            if op is not None:
                reference = experiment.reference

        if reference is None:
            raise ValueError(
                f"Unable to find  metric {config.metric} in measurement space {discoverySpace.measurementSpace.experimentReferences}"
            )
        print(f"Found matching property {op} in {reference}")
    else:
        if config.actuator_identifier is None:
            raise ValueError(
                "If you supply the experiment identifier you must supply the actuator identifier"
            )
        reference = ExperimentReference(
            actuatorIdentifier=config.actuator_identifier,
            experimentIdentifier=config.experiment_identifier,
        )

    # Check that the given metric+experiment exists in measurement space
    exp = discoverySpace.measurementSpace.experimentForReference(reference)
    if exp is None:
        raise ValueError(
            f"Unable to find experiment with identifier {reference} "
            f"in measurement space ({discoverySpace.measurementSpace.experimentReferences})"
        )
    failedProperty = exp.observedPropertyForTargetIdentifier(config.failed_metric)
    if failedProperty is None:
        raise ValueError(
            f"Unable to find failed metric {config.failed_metric} in experiment {reference}"
        )
    targetProperty = exp.observedPropertyForTargetIdentifier(config.metric)
    if targetProperty is None:
        raise ValueError(
            f"Unable to find failed metric {config.failed_metric} in experiment {reference}"
        )

    # Get the correct entity subset
    # MATCHING is problematic as it may find sampled entities which haven't had the target experiment applied to them
    # ALL is problematic as it may return random entities
    if config.entity_filter == EntityFilter.SAMPLED:
        print("Getting sampled entities")
        all_entities = discoverySpace.sampledEntities()
    elif config.entity_filter == EntityFilter.MATCHING:
        print("Getting matching entities")
        all_entities = discoverySpace.matchingEntities()
    else:
        print("Getting all entities")
        all_entities = discoverySpace.sample_store.entities

    print(f"Number of entities: {len(all_entities)}")

    # Assumes all entities in source have been measured by the chosen experiment
    # Filter out any experiments not in measurement space
    df = pd.DataFrame(
        data=[
            e.seriesRepresentation(
                experimentReferences=discoverySpace.measurementSpace.experimentReferences
            )
            for e in all_entities
        ]
    )
    failed_target_column = failedProperty.identifier
    data_columns = [
        c.identifier for c in discoverySpace.entitySpace.constitutiveProperties
    ]

    if config.ignore_columns is not None and len(config.ignore_columns) > 0:
        data_columns = [c for c in data_columns if c not in config.ignore_columns]

    target_column = targetProperty.identifier
    df_working = df[df[failed_target_column] != config.failed_value]
    # df_working2 = df_working[df_working[target_column] > 0.0]

    valid_value_ranges = get_valid_value_ranges(
        df,
        data_columns,
        data_columns,
        [],
        None,
        failed_target_column,
        failed_values=[config.failed_value],
        #     find_valid_intersection=config.find_valid_intersection,
    )
    columns_with_one_valid_value = []
    for k, v in valid_value_ranges.items():
        if len(v) <= 1:
            columns_with_one_valid_value.append(k)
    data_columns_mult = [
        c for c in data_columns if c not in columns_with_one_valid_value
    ]

    mi_output = calculate_mutual_information(
        df_working, data_columns_mult, data_columns_mult, [], None, target_column
    )
    mi_result = mi_output.mutual_information
    # print(mi_result)
    # mi_result2 = calculate_mutual_information(df_working2, data_columns, data_columns, [], None, target_column)
    new_dimensions = mi_pareto_selection(
        mi_result, min_mi_threshold=config.min_mi_threshold
    )
    # new_dimensions2 = mi_pareto_selection(mi_result2, min_mi_threshold=min_mi_threshold)

    if config.mode == "min":
        target_row = df_working[
            df_working[target_column] == df_working[target_column].min()
        ]
    else:
        target_row = df_working[
            df_working[target_column] == df_working[target_column].max()
        ]
    selected_non_important_values = {}
    for col in data_columns:
        if col in new_dimensions:
            continue
        selected_non_important_values[col] = target_row[col].values[0]

    print(
        f"[Rifferla space refinement] Summary (analyzed space: {discoverySpace.uri}):"
        f"\nFiltering for failed experiments discovered that only the following "
        f"values are valid:\n"
        f" {valid_value_ranges}\nFurthermore, an MI and pareto analysis reveled that the following dimensions are "
        f"worth searching further:\n{new_dimensions}\nBased on the Mutual information result: {mi_result}\n"
        f"Finally, the other dimensions should be set to:\n"
        f"{selected_non_important_values}\nAnalysis done, creating new space..."
    )

    new_const_properties_list = []
    for ocp in discoverySpace.entitySpace.config:
        orig_value_type = type(ocp.propertyDomain.values[0])
        orig_variableType = ocp.propertyDomain.variableType
        if ocp.identifier in new_dimensions:
            # TODO: better conversion in categorial?
            new_values = []
            for v in valid_value_ranges[ocp.identifier]:
                # if type(v) != str:
                #     if isinstance(v, float) and int(v) == v:
                #         new_values.append(str(int(v)))
                #     else:
                #         new_values.append(str(v))
                # else:
                #     new_values.append(v)
                if type(v) is not orig_value_type:
                    if orig_value_type is int:
                        new_values.append(int(v))
                    elif orig_value_type is float:
                        new_values.append(float(v))
                    elif orig_value_type is str:
                        if isinstance(v, float) and int(v) == v:
                            new_values.append(str(int(v)))
                        else:
                            new_values.append(str(v))
                else:
                    new_values.append(v)
            new_domain = PropertyDomain(
                values=new_values, variableType=orig_variableType
            )
            new_cp = ConstitutiveProperty(
                identifier=ocp.identifier,
                propertyDomain=new_domain,
                propertyType=NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE,
            )
            new_const_properties_list.append(new_cp)
        else:
            # TODO: better conversion in categorial?
            new_value = selected_non_important_values[ocp.identifier]
            if type(new_value) is not orig_value_type:
                if orig_value_type is int:
                    new_value = int(new_value)
                elif orig_value_type is float:
                    new_value = float(new_value)
                elif orig_value_type is str:
                    if isinstance(new_value, float) and int(new_value) == new_value:
                        new_value = str(int(new_value))
                    else:
                        new_value = str(new_value)
            # if type(new_value) != str:
            #     if isinstance(new_value, float) and int(new_value) == new_value:
            #         new_value = str(int(new_value))
            #     else:
            #         new_value = str(new_value)
            new_domain = PropertyDomain(
                values=[new_value], variableType=orig_variableType
            )
            new_cp = ConstitutiveProperty(
                identifier=ocp.identifier,
                propertyDomain=new_domain,
                propertyType=NonMeasuredPropertyTypeEnum.CONSTITUTIVE_PROPERTY_TYPE,
            )
            new_const_properties_list.append(new_cp)

    new_space = EntitySpaceRepresentation(new_const_properties_list)

    if discoverySpace.properties.stochastic:
        # TODO: remove?
        new_sample_store = SQLSampleStore(
            identifier=None,
            storageLocation=discoverySpace.project_context.metadataStore,
            parameters=None,
        )
    else:
        new_sample_store = discoverySpace.sample_store

    new_discovery_space = DiscoverySpace(
        project_context=discoverySpace.project_context,
        sample_store=new_sample_store,
        entitySpace=new_space,
        measurementSpace=discoverySpace.measurementSpace,
        properties=discoverySpace.properties,
    )
    new_discovery_space.saveSpace()

    # TODO: remove?
    if discoverySpace.properties.stochastic:
        print(
            "Rifferla space refinement] ...done. Copying entities (since it is a probabilistic space)..."
        )
        copy_entities = list(discoverySpace.sample_store.entities)
        new_discovery_space.sample_store.add_external_entities(copy_entities)

    print(
        f"Rifferla space refinement] ...done. Reined discovery space has the following URI and entity storage:"
        f"\nSPACE ID: {new_discovery_space.uri}"
        f"\nENTITY STORAGE: {new_discovery_space.sample_store.uri}"
    )

    return OperationOutput(
        resources=[
            new_discovery_space.resource,
            DataContainerResource(
                config=DataContainer(
                    data={
                        "valid_ranges": valid_value_ranges,
                        "mutual_information": mi_result,
                        "search_dimensions": new_dimensions,
                        "fix_dimensions": selected_non_important_values,
                    }
                )
            ),
        ],
        other=[new_discovery_space],
    )
