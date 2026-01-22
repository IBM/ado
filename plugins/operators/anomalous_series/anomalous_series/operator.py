# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
import itertools
import typing
from typing import Annotated

import pandas as pd
import pydantic
from pydantic import ConfigDict

from orchestrator.core import DataContainerResource
from orchestrator.core.datacontainer.resource import DataContainer, TabularData
from orchestrator.core.discoveryspace.config import EntityFilter
from orchestrator.core.discoveryspace.space import DiscoverySpace
from orchestrator.core.operation.config import FunctionOperationInfo
from orchestrator.core.operation.operation import OperationOutput
from orchestrator.modules.operators.collections import characterize_operation


class SeriesBehaviourEnum(enum.Enum):
    TRUE = "true"
    CONSTANT = "constant"
    MONOTONIC_INCREASE = "monotonic_increase"
    MONOTONIC_DECREASE = "monotonic_decrease"


class ExpectedSeriesBehaviour(pydantic.BaseModel):
    property: Annotated[
        str, pydantic.Field(description="The identifier of an observed property")
    ]
    behaviour: Annotated[
        SeriesBehaviourEnum,
        pydantic.Field(description="The expected behaviour of the series to test"),
    ]


class PropertyTypeEnum(enum.Enum):
    target = "target"
    observed = "observed"


class DetectAnomalousSeries(pydantic.BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_filter: Annotated[
        EntityFilter, pydantic.Field(description="What entities should be used")
    ] = EntityFilter.SAMPLED
    test_property_type: Annotated[
        PropertyTypeEnum,
        pydantic.Field(
            description="If target then test_properties must use target property identifiers. "
            "If observed then they must be observed property identifiers ",
        ),
    ] = "observed"
    groupby_properties: Annotated[
        list[str],
        pydantic.Field(
            description="A list of identifiers of constitutive properties to group by"
        ),
    ]
    independent_properties: Annotated[
        list[str],
        pydantic.Field(
            description="Constitutive property to treat as independent variable of each group. "
            "If there are more than one, then for each property, "
            "the others are added to group_properties to form the groups"
        ),
    ]
    test_properties: Annotated[
        list[ExpectedSeriesBehaviour],
        pydantic.Field(
            description="A list of observed properties and behaviours. "
            "For each group and each independent variable, the behaviour of the property for the group will be checked"
        ),
    ]
    failed_metric: Annotated[
        str | None,
        pydantic.Field(
            description="The target property that indicates measurement validity. Assumed to be a bool",
        ),
    ] = None
    trim_failed: Annotated[
        bool,
        pydantic.Field(
            description="If True entities that have failed_value "
            "for the failed_metric and trim from start/end of the series.",
        ),
    ] = True

    @classmethod
    def default_parameters(cls) -> "DetectAnomalousSeries":
        return cls(
            groupby_properties=[
                "model_name",
                "model_max_length",
                "number_gpus",
                "gpu_model",
            ],
            independent_properties=["batch_size"],
            test_property_type="target",
            test_properties=[
                ExpectedSeriesBehaviour(
                    property="is_valid", behaviour=SeriesBehaviourEnum.TRUE
                )
            ],
            failed_metric="is_valid",
        )


@characterize_operation(
    name="detect_anomalous_series",
    description="""
    This operation checks if the behaviour of an observed property versus
    an independent (constitutive) property is as expected.

    The behaviours that can be checked are: True (all values are 1/True); Constant (all values are the same);
    Monotonically Increasing; or Monotonically Decreasing.

    The entities in the discovery space can be divided into groups based on a user supplied set of constitutive
    properties, other than the selected independent property.
    """,
    configuration_model=DetectAnomalousSeries,
    configuration_model_default=DetectAnomalousSeries.default_parameters(),
    version="1.0",
)
def detect_anomalous_series(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    **parameters: typing.Any,  # noqa: ANN401
) -> OperationOutput:
    """
    This function checks if the behaviour of an observed property versus
    an independent (constitutive) property is as expected.

    The behaviours that can be checked are: True (all values are 1/True); Constant (all values are the same);
    Monotonically Increasing; or Monotonically Decreasing.

    The entities in the discovery space can be divided into groups based on a user supplied set of constitutive
    properties, other than the selected independent property.
    """

    def monotonically_increasing(samples: pd.Series) -> bool:
        return all(x < y for x, y in itertools.pairwise(samples))

    def monotonically_decreasing(samples: pd.Series) -> bool:
        return all(x > y for x, y in itertools.pairwise(samples))

    config = DetectAnomalousSeries.model_validate(parameters)

    if config.entity_filter == EntityFilter.SAMPLED:
        if config.test_property_type == PropertyTypeEnum.target:
            print("Using target property names")
            r = []
            for e in discoverySpace.sampledEntities():
                r.extend(
                    e.experimentSeries(
                        discoverySpace.measurementSpace.experimentReferences
                    )
                )
            df = pd.DataFrame(r)
        else:
            print("Using observed property names")
            df = discoverySpace.measuredEntitiesTable()
    elif config.entity_filter == EntityFilter.MATCHING:
        if config.test_property_type == PropertyTypeEnum.target:
            print("Using target property names")
            r = []
            for e in discoverySpace.matchingEntities():
                r.extend(
                    e.experimentSeries(
                        discoverySpace.measurementSpace.experimentReferences
                    )
                )
            df = pd.DataFrame(r)
        else:
            print("Using observed property names")
            df = discoverySpace.matchingEntitiesTable()
    else:
        raise ValueError(
            f"Unsupported entity filter mode specified {config.entity_filter}"
        )

    # Check if all properties are present
    print("Checking all groupby properties are present: ... ")
    df_columns_set = set(df.columns)
    missing_group_by_properties = df_columns_set.difference(
        set(config.groupby_properties)
    )
    if len(missing_group_by_properties) != 0:
        raise ValueError(
            f"Not all groupby properties were present. Missing: {missing_group_by_properties}"
        )
    print("All present\n")

    print("Checking all independent properties are present: ... ")
    missing_independent_properties = df_columns_set.difference(
        set(config.independent_properties)
    )
    if len(missing_independent_properties) != 0:
        raise ValueError(
            f"Not all independent properties were present. Missing: {missing_independent_properties}"
        )
    print("All present\n")

    print("Checking all test properties are present: ... ")
    for prop in config.test_properties:
        if prop.property not in df.columns:
            raise ValueError(
                f"Test property {prop.property} not found in {df.columns} - "
                f"check the property names you are using match the value of the test_property_type field. "
                f"Current setting: {config.test_property_type}"
            )
    print("All present\n")

    if config.failed_metric is not None:
        print("Checking if failed metric is present: ... ")

        if config.failed_metric not in df.columns:
            raise ValueError("failed_metric was not present in the DataFrame")

        print("Present\n")

    results = []
    for prop in config.independent_properties:
        remaining = config.independent_properties.copy()
        remaining.pop(remaining.index(prop))
        group_vars = config.groupby_properties.copy()
        group_vars.extend(remaining)

        print(f"Testing independent property: {prop}")
        print(f"Grouping by: {group_vars}")

        groups = df.groupby(group_vars)

        print(f"There are {len(group_vars)} series after grouping")

        import numpy as np

        for name, g in groups:
            print(f"\tTesting series {name}")
            for test_prop in config.test_properties:
                print(
                    f"\t\tChecking if the behaviour of property '{test_prop.property}'  is '{test_prop.behaviour.value}' "
                )
                g = g.sort_values(prop)

                if config.trim_failed and config.failed_metric is not None:
                    # Get rid of trailing zeros - should be config?
                    # Get rid of invalid points ...

                    z = np.trim_zeros(g[config.failed_metric])
                    test_group = g.loc[z.index]

                else:
                    test_group = g

                if test_prop.behaviour == SeriesBehaviourEnum.TRUE:
                    if not np.all(test_group[test_prop.property].values):
                        s = test_group[test_prop.property]
                        hole = test_group.loc[s.values == 0]
                        print("\t\tTEST FAILED.\n\t\tFailing entities are:")
                        print(f"\t{test_group.loc[hole.index]['identifier']}")
                        print("\n")
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                False,
                            ]
                        )
                    else:
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                True,
                            ]
                        )
                        print("\t\tTEST PASSED")
                elif test_prop.behaviour == SeriesBehaviourEnum.MONOTONIC_INCREASE:
                    samples = test_group[test_prop.property]
                    if not monotonically_increasing(samples):
                        print("\t\tTEST FAILED.\n\t\tThe series is")
                        print(f"\t\t{test_group[['identifier', test_prop.property]]}")
                        print("\n")
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                False,
                            ]
                        )

                    else:
                        print("\t\tTEST PASSED")
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                True,
                            ]
                        )

                elif test_prop.behaviour == SeriesBehaviourEnum.MONOTONIC_DECREASE:
                    samples = test_group[test_prop.property]
                    if not monotonically_decreasing(samples):
                        print("\tTEST FAILED.\n\t\tThe series is:")
                        print(f"\t\t{test_group[['identifier', test_prop.property]]}")
                        print("\n")
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                False,
                            ]
                        )

                    else:
                        print("\t\tTEST PASSED")
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                True,
                            ]
                        )

                elif test_prop.behaviour == SeriesBehaviourEnum.CONSTANT:
                    if len(test_group[test_prop.property].values.unique()) > 0:
                        s = test_group[test_prop.property]
                        print("\t\tTEST FAILED.\n\t\tThe series is:")
                        print(f'\t\t{test_group[["identifier", test_prop.property]]}')
                        print("\n")
                        print(
                            f"\t\tThe series contains the following values for {test_prop.property}:\n\t\t{s.values.unique()}"
                        )
                        print("\n")
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                False,
                            ]
                        )

                    else:
                        print("\t\tTEST PASSED")
                        results.append(
                            [
                                group_vars,
                                name,
                                prop,
                                test_prop.property,
                                test_prop.behaviour.value,
                                True,
                            ]
                        )

    df = pd.DataFrame(
        data=results,
        columns=[
            "groupby_properties",
            "groupby_values",
            "independent_property",
            "test_property",
            "test_behaviour",
            "passed",
        ],
    )
    df = df.sort_values(["independent_property", "test_property"])
    print(df)

    return OperationOutput(
        resources=[
            DataContainerResource(
                config=DataContainer(
                    tabularData={"results": TabularData.from_dataframe(df)}
                )
            )
        ]
    )
