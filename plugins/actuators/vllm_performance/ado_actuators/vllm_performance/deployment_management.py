# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import asyncio
import logging

import ray

from orchestrator.modules.operators.console_output import RichConsoleSpinnerMessage

logger = logging.getLogger(__name__)


class DeploymentWaiter:
    def __init__(self, k8s_name: str) -> None:
        self.k8s_name = k8s_name
        self.model_downloaded_event = asyncio.Event()


class DeploymentConflictManager:
    def __init__(self) -> None:
        self.deployments_to_wait_for: dict[str, DeploymentWaiter] = {}
        self.model_already_downloaded: set[str] = set()

    def maybe_add_deployment(self, model: str, k8s_name: str) -> bool:
        if (
            model in self.model_already_downloaded
            or model in self.deployments_to_wait_for
        ):
            return False

        self.deployments_to_wait_for[model] = DeploymentWaiter(k8s_name=k8s_name)
        return True

    async def wait(self, request_id: str, k8s_name: str, model: str) -> None:
        waiter = self.deployments_to_wait_for.get(model, None)
        # making sure a deployment does not wait for itself to be READY
        if waiter is not None and waiter.k8s_name != k8s_name:
            console = ray.get_actor(name="RichConsoleQueue")
            while True:
                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request_id,
                        label=f"({request_id}) Waiting on deployment ({waiter.k8s_name}) to download the model required for this experiment",
                        state="start",
                    )
                )
                await waiter.model_downloaded_event.wait()
                # If after we got awoken the model is not among the downloaded models, it means that
                # something has gone wrong, such as the deployment we were waiting for has failed.
                # If am the first to wake up let me add myself as the deployment to be waited for and stop waiting.
                if (
                    model not in self.model_already_downloaded
                    and not self.maybe_add_deployment(k8s_name=k8s_name, model=model)
                ):
                    # If I am not the first to wake up, I get the new waiter object and continue waiting
                    waiter = self.deployments_to_wait_for.get(model)
                    continue

                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request_id,
                        label=f"({request_id}) Done waiting for conflicting K8s deployment",
                        state="stop",
                    )
                )
                break

    def signal(self, k8s_name: str, model: str, error: bool = False) -> None:
        if model not in self.deployments_to_wait_for:
            return

        waiter = self.deployments_to_wait_for.pop(model)
        if waiter.k8s_name != k8s_name:
            raise ValueError(
                f"This environment deployment ({k8s_name}) shouldn't have been created because it is conflicting with deployment {waiter.k8s_name}"
            )
        if not error:
            self.model_already_downloaded.add(model)
        waiter.model_downloaded_event.set()
