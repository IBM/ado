# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

from fastapi import FastAPI
from ray import serve

from orchestrator.api.routers.v0 import actuators, experiments

app = FastAPI()

# latest
app.include_router(actuators.router, prefix="/api/latest")
app.include_router(experiments.router, prefix="/api/latest")

# v0
app.include_router(actuators.router, prefix="/api/v0")
app.include_router(experiments.router, prefix="/api/v0")


@serve.deployment
@serve.ingress(app)
class AdoRESTApi: ...


ado_rest_api = AdoRESTApi.bind()
