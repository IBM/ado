# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

"""Serves orchestrator components"""

import asyncio
import logging
import uuid

import fastapi
import pydantic
import ray
import ray.runtime_env
import ray.serve
import ray.util.queue

import orchestrator.core.operation.config
import orchestrator.metastore.project
import orchestrator.modules.actuators.base
import orchestrator.modules.actuators.catalog
import orchestrator.modules.actuators.measurement_queue
import orchestrator.modules.actuators.registry
import orchestrator.modules.operators.orchestrate
import orchestrator.schema.entity
import orchestrator.schema.experiment
import orchestrator.schema.request
from orchestrator.utilities.logging import configure_logging

app = fastapi.FastAPI()


class ExecutionInfo(pydantic.BaseModel):
    status: str
    output: str | None = None


# GET /actuators -> return list of actuator names
# GET /actuators/{actuator} -> return model of actuator
# POST /actuators -> Create an actuator (disabled)
# POST /actuators/{actuator} -> execute an experiment with the actuator
# GET /actuators/{actuator}/experiments -> return list of experiments
# --- Should have a way to get entity model from actuator ----
# GET /actuators/{actuator}/experiments/{experiment} -> return specific experiments
# GET /operators/{operator} -> return model of operator
# --- How to get model of parameters ? Route or parameter
# POST /operators/{operator} -> create operation, return operation id
# GET /spaces -> Return spaces
# GET /spaces/{space} -> Return model of space: query string for details
# POST /spaces -> Create a space
# GET /operations -> Return list of operations
# GET /operations/{operation} -> Return model of  operation
# GET /operations/{operation}/related -> Return related resources to operation


#
# GET /{resource type}/{resource id}/related -> for all


@ray.serve.deployment
@ray.serve.ingress(app)
class ActuatorIngress:

    def __init__(self):

        configure_logging()

        self._updateQueue = (
            orchestrator.modules.actuators.measurement_queue.MeasurementQueue.get_measurement_queue()
        )
        self._completedRequests = {}
        self._submittedRequests = {}
        self.log = logging.getLogger("OrchServer")

        self._executions = {}

        asyncio.get_event_loop().create_task(self.monitorUpdates())

    # This is for requests not connected to a state
    # If they are connected to a state then the state has to update
    async def monitorUpdates(self):

        while True:
            try:

                # Get new updates
                try:
                    self.log.debug("Awaiting update queue get")
                    update = await self._updateQueue.get_async(
                        block=True, timeout=30
                    )  # type: orchestrator.schema.request.MeasurementRequest
                except ray.util.queue.Empty:
                    self.log.info(
                        "Did not get an update after 30 secs - will continue waiting"
                    )
                else:
                    self._completedRequests[update.requestid] = update

                # Clean submitted queue - we can't assume the update that came in will be in the submitted dict!
                # This is because the calculation is asynchronous to the submission
                # Thus extremely quick calculations may complete before the request id is returned to the
                # submission co-routine
                # Instead we have to periodically clean the submitted queue
                for key in self._completedRequests.keys():
                    if key in self._submittedRequests.keys():
                        self._submittedRequests.pop(key)
            except Exception as error:
                self.log.warning(
                    "Unexpected exception in monitor loop: {}".format(error)
                )
                self.log.warning("Assuming transient - will continue")
                await asyncio.sleep(1)

    @app.get("/actuators")
    async def getActuators(self) -> list:

        r = orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        return list(r.actuatorIdentifierMap.keys())

    async def getActuator(
        self, actuator: str, params: dict
    ) -> orchestrator.modules.actuators.base.ActuatorBase:

        r = orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()

        print(actuator)
        print(r.actuatorIdentifierMap.keys())
        try:
            actuatorActor = r.actuatorForIdentifier(actuatorid=actuator)
        except orchestrator.modules.actuators.registry.UnknownActuatorError:
            raise fastapi.HTTPException(
                status_code=404, detail=f"Unknown actuator {actuator}"
            )

        return actuatorActor

    @app.get("/actuators/{actuator}")
    async def getActuatorRequest(
        self, actuator: str, request: fastapi.Request
    ) -> list[str]:

        actuator = await self.getActuator(actuator, params=dict(request.query_params))
        catalog = await actuator.catalog()
        return [f"{e.reference}" for e in catalog.experiments]

    @app.get(
        "/actuators/{actuator}/experiments",
        response_model_exclude_defaults=True,
        response_model_exclude_unset=True,
    )
    async def getActuatorExperiments(
        self, actuator: str, request: fastapi.Request
    ) -> list[orchestrator.schema.entity.Experiment]:

        r = orchestrator.modules.actuators.registry.ActuatorRegistry.globalRegistry()
        experiments = []
        for (
            c
        ) in (
            r.catalogs
        ):  # type: orchestrator.modules.actuators.catalog.ExperimentCatalog
            experiments.extend(
                [e for e in c.experiments if e.actuatorIdentifier == actuator]
            )

        return experiments

    @app.post("/actuators/{actuator}")
    async def submitRequest(
        self,
        request: fastapi.Request,
        measurementRequest: orchestrator.schema.request.MeasurementRequest,
        actuator,
    ) -> str:

        actuator = await self.getActuator(
            actuator=measurementRequest.experimentReference.actuatorIdentifier,
            params=dict(request.query_params),
        )
        # noinspection PyUnresolvedReferences
        retval = await actuator.submit.remote(
            entities=measurementRequest.entities,
            experimentReference=measurementRequest.experimentReference,
            requesterid=measurementRequest.operation_id,
            requestIndex=measurementRequest.requestIndex,
        )

        self._submittedRequests[retval] = measurementRequest
        measurementRequest.requestid = retval

        return retval

    @app.get(
        "/measurementrequests/{requestid}",
        response_model=orchestrator.schema.request.MeasurementRequest,
    )
    async def getRequest(
        self, requestid: str
    ) -> orchestrator.schema.request.MeasurementRequest:

        request = self._completedRequests.get(requestid, None)
        if request is None:
            request = self._submittedRequests.get(requestid, None)
            if request is None:
                raise fastapi.HTTPException(
                    status_code=404, detail=f"Unknown request id {requestid}"
                )

        return request

    @app.get("/measurementrequests")
    async def getRequests(
        self, submitted: bool = True, completed: bool = True
    ) -> list[str]:

        requestids = []
        self.log.debug(f"Submitted ids {self._submittedRequests.keys()}")
        self.log.debug(f"Completed ids {self._completedRequests.keys()}")
        if submitted is True:
            requestids.extend(list(self._submittedRequests.keys()))

        if completed is True:
            requestids.extend(list(self._completedRequests.keys()))

        self.log.debug(f"requestids is {requestids}")
        # Due to asynchronicity the same id may be in submitted and completed - remove duplicates
        return list(set(requestids))

    @app.post("/executions")
    async def executeProject(
        self,
        request: fastapi.Request,
        configuration: orchestrator.core.operation.config.DiscoveryOperationResourceConfiguration,
    ) -> str:

        args = dict(request.query_params)
        queue = ray.util.queue.Queue()
        execid = str(uuid.uuid4())[:6]

        # AP: adapt parameters for new orchestrate params
        entities_output_file = args.get("write_entities")
        project_context = orchestrator.metastore.project.ProjectContext()

        task = asyncio.create_task(
            orchestrator.modules.operators.orchestrate.orchestrate(
                base_operation_configuration=configuration,
                project_context=project_context,
                discovery_space_configuration=configuration.discoverySpace,
                discovery_space_identifier=None,
                entities_output_file=entities_output_file,
                queue=queue,
                execid=execid,
            )
        )
        self._executions[execid] = task

        return execid

    def _get_execution(self, execid: str) -> asyncio.Task:
        task = self._executions.get(execid, None)
        if task is None:
            raise fastapi.HTTPException(
                status_code=404, detail=f"no known execution with id {execid}"
            )
        return task

    @app.get("/executions/{execid}")
    async def getExecutionStatus(self, execid: str) -> ExecutionInfo:
        task = self._get_execution(execid)
        try:
            exception = task.exception()
            if exception is None:
                result = ExecutionInfo(
                    status="success",
                    output=str(task.result()),
                )
            else:
                result = ExecutionInfo(
                    status="error",
                    output=str(exception),
                )
                self.log.exception(task)
        except asyncio.CancelledError:
            result = ExecutionInfo(status="cancelled")
        except asyncio.InvalidStateError:
            result = ExecutionInfo(status="running")
        return result

    @app.delete("/executions/{execid}", status_code=204)
    async def removeExecution(self, execid: str):
        task = self._get_execution(execid)
        if not task.done():
            raise fastapi.HTTPException(
                status_code=409, detail="cannot delete running execution"
            )
        del self._executions[execid]


actuator_deployment = ActuatorIngress.bind()
