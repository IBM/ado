# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT
import asyncio

from fastapi import FastAPI
from ray import serve

from orchestrator.api.routers.latest import latest
from orchestrator.api.routers.v0 import v0
from orchestrator.api.state.queue import watch_queue
from orchestrator.utilities.logging import configure_logging

app = FastAPI()

app.include_router(latest.router, prefix="/api")
app.include_router(v0.router, prefix="/api")


@serve.deployment
@serve.ingress(app)
class AdoRESTApi:

    def __init__(self):
        configure_logging()

        # AP 16/09/2025: keep reference to the task
        # due to https://github.com/python/cpython/issues/88831
        self.task_reference = asyncio.get_event_loop().create_task(watch_queue())


ado_rest_api = AdoRESTApi.bind()
