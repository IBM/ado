# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging

import ray

from orchestrator.modules.operators.console_output import RichConsoleSpinnerMessage

logger = logging.getLogger(__name__)


class DeploymentWaiter:
    def __init__(self, identifier: str):
        self.identifier = identifier
        self.wait_event = asyncio.Event()


class DeploymentConflictManager:
    def __init__(self):
        self.deployments_to_wait_for: dict[str, DeploymentWaiter] = {}
        self.model_already_downloaded: set[str] = set()

    def maybe_add_deployment(self, model: str, identifier: str) -> bool:
        if (
            model not in self.model_already_downloaded
            and model not in self.deployments_to_wait_for
        ):
            self.deployments_to_wait_for[model] = DeploymentWaiter(
                identifier=identifier
            )
            return True
        return False

    async def wait(self, request_id: str, identifier: str, model: str) -> None:
        waiter = self.deployments_to_wait_for.get(model, None)
        # making sure a deployment does not wait for itself to be READY
        if waiter is not None and waiter.identifier != identifier:
            console = ray.get_actor(name="RichConsoleQueue")
            while True:
                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request_id,
                        label=f"({request_id}) Waiting for conflicting K8s deployment ({waiter.identifier}) to be started",
                        state="start",
                    )
                )
                await waiter.wait_event.wait()
                # If after we got awaken the model is not among the downloaded models, it means that
                # something has gone wrong, such as the deployment we were waiting for has failed.
                # If am the first to wake up let me add myself as the deployment to be waited for and stop waiting.
                if (
                    model not in self.model_already_downloaded
                    and not self.maybe_add_deployment(
                        identifier=identifier, model=model
                    )
                ):
                    # If I am not the first to wake up, I get the new waiter object and continue waiting
                    waiter = self.deployments_to_wait_for.get(model, None)
                    continue

                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request_id,
                        label=f"({request_id}) Done waiting for conflicting K8s deployment",
                        state="stop",
                    )
                )
                break

    def signal(self, identifier: str, model: str, error: bool = False) -> None:
        if model in self.deployments_to_wait_for:
            waiter = self.deployments_to_wait_for.pop(model)
            assert (
                waiter.identifier == identifier
            ), f"This environment deployment ({identifier}) shouldn't have been created because it is conflicting with deployment {waiter.identifier}"
            if not error:
                self.model_already_downloaded.add(model)
            waiter.wait_event.set()
