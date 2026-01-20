# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import abc
import enum
import typing
import uuid
from typing import Annotated

import pydantic

from orchestrator.schema.observed_property import (
    ObservedProperty,
    ObservedPropertyValue,
)
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.virtual_property import (
    VirtualObservedProperty,
)

if typing.TYPE_CHECKING:  # pragma: nocover
    import pandas as pd


class MeasurementResultStateEnum(str, enum.Enum):
    VALID = "Valid"
    INVALID = "Invalid"


class MeasurementResult(pydantic.BaseModel):

    uid: Annotated[
        str,
        pydantic.Field(
            default_factory=lambda: str(uuid.uuid4()),
            frozen=True,
            min_length=36,
            max_length=36,
            description="A unique identifier for this MeasurementResult",
        ),
    ]

    entityIdentifier: Annotated[
        str, pydantic.Field(description="The unique identifier of the entity")
    ]

    metadata: Annotated[
        dict,
        pydantic.Field(
            default_factory=dict, description="Metadata about the MeasurementResult"
        ),
    ]

    def __eq__(self, other: "MeasurementResult") -> bool:
        return self.uid == other.uid

    @abc.abstractmethod
    def series_representation(
        self,
        output_format: typing.Literal["target", "observed"],
        virtual_target_property_identifiers: list[str] | None = None,
    ) -> "pd.Series": ...


class ValidMeasurementResult(MeasurementResult):
    """Used to record a valid measurement

    Note the experiment that made the measurements can be retrieved via the PropertyValues

    ValidMeasurementResult.measurements[0].property.experimentReference
    """

    measurements: Annotated[
        list[ObservedPropertyValue],
        pydantic.Field(
            description="A list of the observed property values measured. Cannot be empty"
        ),
    ]

    model_config = pydantic.ConfigDict(extra="forbid")

    @pydantic.field_validator("measurements")
    def validate_measurements(
        cls, value: list[ObservedPropertyValue]
    ) -> list[ObservedPropertyValue]:

        if len(value) == 0:
            raise ValueError(
                "A valid measurement result must included at least on measured value - none given"
            )

        refs = list({v.property.experimentReference for v in value})

        if len(refs) > 1:
            raise ValueError(
                f"All measurements in a results set must be made by the same experiment - "
                f"However {len(refs)} experiments detected: {refs}"
            )

        return value

    @property
    def experimentReference(self) -> ExperimentReference:

        # All the PropertyValues should have the same reference as it's checked on init
        return self.measurements[0].property.experimentReference

    def series_representation(
        self,
        output_format: typing.Literal["target", "observed"],
        virtual_target_property_identifiers: list[str] | None = None,
    ) -> "pd.Series":

        import pandas as pd

        supported_formats = {"target", "observed"}
        if output_format not in supported_formats:
            raise ValueError(
                f"The only supported series representation output formats are {supported_formats}"
            )

        rep = {
            "identifier": self.entityIdentifier,
            "experiment_id": self.experimentReference,
            "valid": True,
        }

        for measurement in self.measurements:

            key = (
                f"{measurement.property.targetProperty.identifier}"
                if output_format == "target"
                else measurement.property.identifier
            )

            rep[key] = measurement.value

        if virtual_target_property_identifiers:

            observed_properties = [
                p.property
                for p in self.measurements
                if isinstance(p.property, ObservedProperty)
            ]
            for property_identifier in virtual_target_property_identifiers:
                virtual_observed_properties = VirtualObservedProperty.from_observed_properties_matching_identifier(
                    observed_properties=observed_properties,
                    identifier=property_identifier,
                )

                if not virtual_observed_properties:
                    continue

                for (
                    virtual_observed_property
                ) in virtual_observed_properties:  # type: VirtualObservedProperty
                    if output_format == "target":
                        rep[
                            virtual_observed_property.virtualTargetPropertyIdentifier
                        ] = virtual_observed_property.aggregate_from_observed_properties(
                            self.measurements
                        )
                    else:
                        rep[virtual_observed_property.identifier] = (
                            virtual_observed_property.aggregate_from_observed_properties(
                                self.measurements
                            )
                        )

        return pd.Series(rep)


class InvalidMeasurementResult(MeasurementResult):
    """Used to record an invalid measurement"""

    experimentReference: Annotated[
        ExperimentReference,
        pydantic.Field(
            description="Reference to the experiment that attempted the measurement"
        ),
    ]
    reason: Annotated[
        str,
        pydantic.Field(
            description="A string describing why the measurement was deemed invalid"
        ),
    ]

    model_config = pydantic.ConfigDict(extra="forbid")

    def series_representation(
        self,
        output_format: typing.Literal["target", "observed"],
        virtual_target_property_identifiers: list[str] | None = None,
    ) -> "pd.Series":

        import pandas as pd

        return pd.Series(
            {
                "identifier": self.entityIdentifier,
                "experiment_id": self.experimentReference,
                "valid": False,
                "reason": self.reason,
            },
        )


class DuplicateMeasurementResultError(ValueError): ...


def measurement_result_type_discriminator(
    result: dict | ValidMeasurementResult | InvalidMeasurementResult,
) -> str:

    if isinstance(result, ValidMeasurementResult):
        return "Valid"
    if isinstance(result, InvalidMeasurementResult):
        return "Invalid"
    if isinstance(result, dict):
        if result.get("measurements", None):
            return "Valid"
        if result.get("reason", None):
            return "Invalid"

    raise ValueError(
        f"Unable to determine measurement result type for result: {result}"
    )


MeasurementResultType = Annotated[
    Annotated[ValidMeasurementResult, pydantic.Tag("Valid")]
    | Annotated[InvalidMeasurementResult, pydantic.Tag("Invalid")],
    pydantic.Discriminator(measurement_result_type_discriminator),
]
