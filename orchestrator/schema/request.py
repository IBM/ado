# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import datetime
import enum
import typing
import uuid
from typing import Annotated, Any

import pydantic
from dateutil.tz import tzlocal

from orchestrator.schema.entity import Entity
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.result import (
    InvalidMeasurementResult,
    MeasurementResultType,
    ValidMeasurementResult,
)

if typing.TYPE_CHECKING:  # pragma: nocover
    import pandas as pd
    from pydantic import ValidatorFunctionWrapHandler


class MeasurementRequestStateEnum(str, enum.Enum):
    UNKNOWN = "Unknown"
    SUCCESS = "Success"
    FAILED = "Failed"


def timestamp() -> datetime.datetime:
    """For use in MeasurementRequest timestamp creation"""

    return datetime.datetime.now(tzlocal())


class MeasurementRequest(pydantic.BaseModel, validate_assignment=True):
    """This class represents a request for a Measurement received by an Actuator

    Instances of the class should only be created by Actuators as a result of their `submit`
    method being called.

    The class allows the actuator to record when it received a request along
    with information about the internal id it assigns to the request.

    Downstream consumers can then use instances of this class to check the status of a  request,
    populate the Entity with values when the Measurement is completed and order measurements by when they were
    requested for timeseries calculations.

    """

    operation_id: Annotated[
        str,
        pydantic.Field(
            description="The id of the operation that requested this measurement"
        ),
    ]
    requestIndex: Annotated[
        int,
        pydantic.Field(
            description="An integer set by requester. This can be used to denote that this was the Nth Measurement requested by it"
        ),
    ]
    experimentReference: Annotated[
        ExperimentReference,
        pydantic.Field(description="An reference detailing the experiment to perform"),
    ]
    entities: Annotated[
        list[Entity],
        pydantic.Field(
            description="An Entity instance representing the entity being measured"
        ),
    ]
    requestid: Annotated[
        str,
        pydantic.Field(
            default_factory=lambda: str(uuid.uuid4()),
            description="The id associate with this request - created by the actuator.",
        ),
    ]
    status: Annotated[
        MeasurementRequestStateEnum,
        pydantic.Field(description="The status of the measurement"),
    ] = MeasurementRequestStateEnum.UNKNOWN
    timestamp: Annotated[datetime.datetime, pydantic.Field(default_factory=timestamp)]

    measurements: Annotated[
        tuple[MeasurementResultType, ...] | None,
        pydantic.Field(description="The results of the measurement"),
    ] = None

    metadata: Annotated[
        dict,
        pydantic.Field(
            default_factory=dict, description="Metadata about the measurement request"
        ),
    ]

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def fail_on_measurements_reassignment(
        cls, data: Any, handler: "ValidatorFunctionWrapHandler"  # noqa: ANN401
    ) -> "MeasurementRequest":

        def populate_measurement_results_in_entities(
            request: "MeasurementRequest",
        ) -> None:

            if not request.measurements:
                return

            for entity in request.entities:
                results_for_entity = [
                    result
                    for result in request.measurements
                    if result.entityIdentifier == entity.identifier
                    and isinstance(result, ValidMeasurementResult)
                ]
                for result in results_for_entity:
                    if result not in entity.measurement_results:
                        entity.add_measurement_result(result)

        # AP: 05/03/2025
        # BASIC CASE: FIRST VALIDATION OF THE MODEL
        # data will be a dictionary instance the first time this validator is called
        # as Pydantic will not have instantiated the class yet, so we can trust that
        # the handler should be called.
        if isinstance(data, dict):
            validated_model = handler(data)
            populate_measurement_results_in_entities(validated_model)
            return validated_model

        # SIMPLE CASE: THE MODEL HAS ALREADY BEEN VALIDATED BUT NO MEASUREMENTS HAVE BEEN SET
        # Here we need to check whether the 'measurements' field has already been set
        # If this is not the case, we can go ahead without any issue
        if "measurements" not in data.model_fields_set:
            validated_model = handler(data)
            populate_measurement_results_in_entities(validated_model)
            return validated_model

        # GENERIC CASE: THE MODEL HAS ALREADY BEEN VALIDATED AND MEASUREMENTS WERE SET
        # In this case we must make sure they do not change after the validation has been run.
        measurement_uids = (
            {result.uid for result in data.measurements}
            if data.measurements is not None
            else None
        )

        validated_model: MeasurementRequest = handler(data)

        # Handle case in which the user has set to None the measurements, which previously
        # weren't None
        if measurement_uids is not None and validated_model.measurements is None:
            raise ValueError(
                "Measurements cannot be re-assigned once they have been set."
            )

        new_measurement_uids = {result.uid for result in validated_model.measurements}
        if measurement_uids != new_measurement_uids:
            raise ValueError(
                f"Measurements cannot be re-assigned once they have been set. OLD: {measurement_uids}, NEW: {new_measurement_uids}"
            )

        return validated_model

    @pydantic.field_validator("measurements")
    @classmethod
    def validate_measurements(
        cls,
        value: list[ValidMeasurementResult | InvalidMeasurementResult] | None,
        values: pydantic.ValidationInfo,
    ) -> list[ValidMeasurementResult | InvalidMeasurementResult] | None:

        # None is allowed
        if value is None:
            return value

        if len(value) == 0:
            raise ValueError(
                "A MeasurementRequest must have at least 1 MeasurementResult - 0 present"
            )

        refs = list({v.experimentReference for v in value})

        if len(refs) > 1:
            raise ValueError(
                f"All MeasurementResults in a MeasurementRequest set must be made by the same experiment - "
                f"However {len(refs)} experiments detected: {refs}"
            )

        # Check if an entity is duplicated
        measured_entity_ids = [v.entityIdentifier for v in value]
        if len(measured_entity_ids) != len(set(measured_entity_ids)):
            raise ValueError(
                f"There are multiple measurement results for the same entity which is not allowed. {measured_entity_ids}"
            )

        # Check that all entities in measurements are in request.entities
        requested_entity_ids = {e.identifier for e in values.data.get("entities")}
        for e in measured_entity_ids:
            if e not in requested_entity_ids:
                raise ValueError(
                    f"MeasurementResult present for {e} but this entity is not in the request entities: {requested_entity_ids}"
                )

        # Check that all request.entities have a measurement result
        measured_entity_ids = set(measured_entity_ids)
        if measured_entity_ids != requested_entity_ids:
            raise ValueError(
                f"You must set a measurement result for all entities in the request. Missing: {requested_entity_ids - measured_entity_ids}"
            )

        return value

    def __str__(self) -> str:

        if len(self.entities) == 1:
            return "request-{}-experiment-{}-entities-{}-requester-{}-time-{}".format(  # noqa: UP032
                self.requestid,
                self.experimentReference.experimentIdentifier,
                self.entities[0],
                self.operation_id,
                self.timestamp,
            )
        return "request-{}-experiment-{}-entities-multi-{}-requester-{}-time-{}".format(  # noqa: UP032
            self.requestid,
            self.experimentReference.experimentIdentifier,
            len(self.entities),
            self.operation_id,
            self.timestamp,
        )

    def measurement_for_entity(
        self, entity_identifier: str
    ) -> ValidMeasurementResult | InvalidMeasurementResult | None:
        """Returns the measurement for the requested entity identifier.

        If no measurements have been set returns None

        Raises:
            ValueError: if entity_identifier is not the identifier of an entity in request entities
        """

        retval = None
        if self.measurements:
            if entity_identifier not in {e.identifier for e in self.entities}:
                raise ValueError(
                    f"Entity with identifier {entity_identifier} was not part of this MeasurementRequest"
                )

            mrs = [
                mr
                for mr in self.measurements
                if mr.entityIdentifier == entity_identifier
            ]

            # By construction if measurements has been set there must be one for each entity in the request
            retval = mrs[0]

        return retval

    def series_representation(
        self,
        output_format: typing.Literal["target", "observed"],
        virtual_target_property_identifiers: list[str] | None = None,
    ) -> list["pd.Series"]:

        import pandas as pd

        supported_formats = {"target", "observed"}
        if output_format not in supported_formats:
            raise ValueError(
                f"The only supported series representation output formats are {supported_formats}"
            )

        #
        result_series = [
            res.series_representation(
                output_format=output_format,
                virtual_target_property_identifiers=virtual_target_property_identifiers,
            )
            for res in self.measurements
        ]

        #
        entities_constitutive_properties_series = {
            entity.identifier: entity.seriesRepresentation(constitutiveOnly=True)
            for entity in self.entities
        }

        #
        measurement_series = []
        occurrences = {}
        for idx, s in enumerate(result_series):

            # Update occurrences counter
            result_index = s["identifier"] if s["identifier"] in occurrences else 0

            occurrences[s["identifier"]] = result_index + 1

            req = pd.Series(
                {
                    "request_id": self.requestid,
                    "request_index": self.requestIndex,
                    "entity_index": idx,
                    "result_index": result_index,
                }
            )

            # We are dealing with Series, so we want to avoid duplicate keys
            # The only keys that can clash are the ones between the entities'
            # constitutive properties and the ones coming from the MeasurementResult
            e = entities_constitutive_properties_series[s["identifier"]]
            e = e[e.index.difference(s.keys())]
            measurement_series.append(pd.concat([req, e, s]))

        return measurement_series


class ReplayedMeasurement(MeasurementRequest):
    """This class represents a completed measurement on an Entity

    Instances of the class can be used to replay an existing measurement rather than perform it again

    Downstream consumers can filter out instances of this class from MeasurementRequests if necessary
    """

    experimentReference: Annotated[
        ExperimentReference,
        pydantic.Field(
            description="A reference detailing the experiment that was performed"
        ),
    ]
    entities: Annotated[
        list[Entity],
        pydantic.Field(description="The Entity instances which are being forwarded"),
    ]
    status: Annotated[
        MeasurementRequestStateEnum,
        pydantic.Field(description="The status of the measurement"),
    ] = MeasurementRequestStateEnum.SUCCESS
    requestid: Annotated[
        str,
        pydantic.Field(
            default_factory=lambda: f"replayed-measurement-{str(uuid.uuid4())[:6]}"
        ),
    ]

    def __str__(self) -> str:

        if len(self.entities) == 1:
            return "{}-experiment-{}-entities-{}-time-{}".format(  # noqa: UP032
                self.requestid,
                self.experimentReference.experimentIdentifier,
                self.entities[0],
                self.timestamp,
            )
        return "{}-experiment-{}-entities-multi-{}-time-{}".format(  # noqa: UP032
            self.requestid,
            self.experimentReference.experimentIdentifier,
            len(self.entities),
            self.timestamp,
        )
