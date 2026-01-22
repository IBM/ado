# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import os
import uuid
from typing import Annotated

import pydantic
import ray
import yaml

from orchestrator.core.actuatorconfiguration.config import GenericActuatorParameters
from orchestrator.modules.actuators.base import ActuatorBase, DeprecatedExperimentError
from orchestrator.modules.actuators.catalog import ExperimentCatalog
from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.entity import Entity
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.reference import ExperimentReference
from orchestrator.schema.request import MeasurementRequest

from .experiment_executor import run_experiment


# In case we need configurable parameters for our actuator, we can create a class that inherits from the pydantic model GenericActuatorParameters,
# and reference it in the "parameters_class" class variable of our actuator.
#
# Users will be able to create instance of this class with specific options for the actuator (ado create actuatorconfiguration)
# These will be stored in a project in a metastore
# Users will also be able to get a default for this template using (ado template actuatorconfiguration)
#
# Details: How this model is used to store a set of parameters or to show  parameters to users
#
# When storing this pydantic model, the serialized representation is obtained using pydantic model_dump() with no options.
# When outputting, the serialized representation is obtained also with model_dump(), and for schema with  model_dump_schema().
# In general, commands that output a serialization of the object (ado template, ado get)  may include options to
# model_dump to include None values or default values.
# However, the stored representation is always that produced by model_dump() with no options.
class RoboticLabParameters(GenericActuatorParameters):
    my_parameter: Annotated[str, pydantic.Field()] = "hello world"


# An Actuator must do three things
# 1. Provide a catalog of Experiments it can execute - the catalog method
# 2. Provide a way to run those experiments asynchronously - the submit method
# 3. Provide the results of the experiments - via the Results Queue
@ray.remote
class RoboticLab(ActuatorBase):

    identifier = (
        "robotic_lab"  # The user-facing label you want this actuator to be called by
    )
    parameters_class = RoboticLabParameters  # we tell ado what our parameters class is

    @classmethod
    def catalog(
        cls, actuator_configuration: GenericActuatorParameters | None = None
    ) -> ExperimentCatalog:
        """Returns the Experiments your actuator provides"""

        # The catalog be formed in code here or read from a file containing the Experiments models
        # This shows reading from a file

        path = os.path.abspath(__file__)
        path = os.path.split(path)[0]
        with open(os.path.join(path, "experiments.yaml")) as f:
            data = yaml.safe_load(f)
            experiments = [Experiment(**data[e]) for e in data]

        return ExperimentCatalog(
            catalogIdentifier=cls.identifier,
            experiments={e: e for e in experiments},
        )

    def __init__(self, queue: MeasurementQueue, params: dict | None = None) -> None:
        """
        queue: Queue where experiment results are put for consumers
        params: This actuators configuration parameters (Note: Soon this will be a model defined by the Actuator)
        """

        # Set logging and the ivar self._stateUpdateQueue and self._parameters
        super().__init__(queue=queue, params=params)

    async def submit(
        self,
        entities: list[Entity],
        experimentReference: ExperimentReference,
        requesterid: str,
        requestIndex: int,
    ) -> list[str]:
        """Runs experimentReference on entities

        The result of running the experiment is expected to be returned asynchronously via the MeasurementQueue (self._stateUpdateQueue)
        as a MeasurementRequest instance

        :param entities: A list of Entity instances representing the entities to be measured
        :param experimentReference: An ExperimentReference defining the experiment to run on the entities. Expected to come from the actuators catalog.
        :param requesterid: Unique identifier of the requester of this measurement
        :param requestIndex: The index of this request i.e. this is the ith request by the requester in the current run

        Returns:
            A list with the ids of the experiments it submitted.
            NOTE: An actuator may not submit all entities in one batch.

        Raises:
            DeprecatedExperimentError: if the experimentReference points to an experiment that is deprecated.
        """

        # The following steps MUST be done
        # How they are implemented is up to the developer

        # 1. Create MeasurementRequest(s) instances representing the experiments to be performed
        # 2. Asynchronously launch the experiments -> placing their results in the resultsQueue
        # 3. Return the ids of the MeasurementRequests submit to the caller of this function

        # Here we give an example of creating the MeasurementRequests in this method
        # and executing asynchronously via a remote ray function

        # Create MeasurementRequest(s)
        # You can create one for all entities or one for each entity or any split
        # NOTE: ALL must have same requestIndex - this is used to maintain timeseries of requests
        request = MeasurementRequest(
            operation_id=requesterid,
            requestIndex=requestIndex,
            experimentReference=experimentReference,
            entities=entities,
            requestid=str(uuid.uuid4())[:6],  # Create UID as you like
        )

        ## Check if the requested experiment is deprecated
        experiment = self.__class__.catalog().experimentForReference(
            request.experimentReference
        )

        ## Check if the experiment has parameterization fields and create the right ParameterizedExperiment instance
        ## Note: this is necessary for getting values of optional properties that the discoverySpace overrides
        if experimentReference.parameterization:
            experiment = ParameterizedExperiment(
                parameterization=experimentReference.parameterization,
                **experiment.model_dump(),
            )

        if experiment.deprecated is True:
            raise DeprecatedExperimentError(f"Experiment {experiment} is deprecated")

        ## Execute experiment
        ## Note: Here the experiment instance is just past for convenience since we retrieved it above
        run_experiment.remote(request, experiment, self._stateUpdateQueue)

        return [request.requestid]
