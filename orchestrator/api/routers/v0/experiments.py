# # Copyright (c) IBM Corporation
# # SPDX-License-Identifier: MIT
#
# import itertools
#
# from fastapi import APIRouter, status
#
# # from orchestrator.api.state.queue import shared_queue
# from orchestrator.modules.actuators.registry import ActuatorRegistry
# from orchestrator.schema.experiment import Experiment
#
# router = APIRouter(
#     prefix="/experiments",
#     tags=["experiments"],
#     responses={status.HTTP_404_NOT_FOUND: {"description": "Not found"}},
# )
#
#
# @router.get("", tags=["experiments"])
# async def get_experiments() -> list[Experiment]:
#     actuator_registry = ActuatorRegistry()
#     return itertools.chain.from_iterable(
#         [
#             actuator_registry.catalogForActuatorIdentifier(actuator_id).experiments
#             for actuator_id in actuator_registry.actuatorIdentifierMap
#         ]
#     )
