"""
"ado/api/rest.py"

This module defines the FastAPI application and the Ray Serve deployment
for the ADO (Accelerated Discovery Orchestrator) REST API.

The module performs the following tasks:

* Creates a FastAPI instance and includes the API routers for the
latest and v0 endpoints.
* Configures coloured logging via :func:`ado.utilities.logging.configure_logging`.
* Instantiates Ray actors to keep a global state for MeasurementRequests and actuators.
* Exposes the API as a Ray Serve deployment, so it can be served in a Ray Serve cluster.

The code uses the ``serve`` decorator from Ray Serve to expose the
``FastAPI`` instance to the Ray Serve traffic routing system.
"""

# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from fastapi import FastAPI
from ray import serve

from ado.api.routers.latest import latest
from ado.api.routers.v0 import v0
from ado.api.state.actuators import ActuatorDictionaryActor
from ado.api.state.queue import QueueMonitorActor
from ado.utilities.logging import configure_logging

app = FastAPI()

# Add routers for API endpoints.
app.include_router(latest.router, prefix="/api")
app.include_router(v0.router, prefix="/api")


@serve.deployment
@serve.ingress(app)
class AdoRESTApi:
    def __init__(self) -> None:
        """Initialise the REST API deployment.

        The constructor configures coloured logging, then creates and
        keeps references to the :class:`~ado.api.state.queue.QueueMonitorActor`
        and :class:`~ado.api.state.actuator_actors.ActuatorDictionaryActor`.

        Keeping the actor references as instance attributes prevents
        Ray from garbage-collecting them while the deployment remains
        active.
        """
        configure_logging()

        self.queue_monitor = QueueMonitorActor.options(
            name="QueueMonitorActor", namespace="api"
        ).remote()

        self.actuator_dictionary_actor = ActuatorDictionaryActor.options(
            name="ActuatorDictionaryActor", namespace="api"
        ).remote()


# Bind the deployment for use by Ray Serve.
ado_rest_api = AdoRESTApi.bind()
