# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import logging
import math
import subprocess
import sys
import time

import ray
from ado_actuators.vllm_performance.actuator_parameters import (
    VLLMPerformanceTestParameters,
)
from ado_actuators.vllm_performance.env_manager import (
    EnvironmentState,
)
from ado_actuators.vllm_performance.k8.create_environment import (
    create_test_environment,
)
from ado_actuators.vllm_performance.k8.yaml_support.build_components import (
    VLLMDtype,
)
from ado_actuators.vllm_performance.vllm_performance_test.execute_benchmark import (
    execute_benchmark,
)
from ray.actor import ActorHandle

from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.request import MeasurementRequest
from orchestrator.utilities.support import (
    compute_measurement_status,
    create_measurement_result,
    dict_to_measurements,
    get_experiment_input_values,
)

logger = logging.getLogger(__name__)


def _build_entity_env(values: dict[str, str]) -> str:
    """
    This is the list of entity parameters that define the environment:
        * model name
        * image name
        * number of gpus
        * gpu type
        * number of cpus
        * memory
        * max batch tokens
        * max number of sequences
        * gpu memory utilization
        * data type
        * cpu offload
    Build entity based environment parameters
    :param values: experiment values
    :return: definition
    """
    env_values = {
        "model": values.get("model"),
        "image": values.get("image"),
        "n_gpus": values.get("n_gpus"),
        "gpu_type": values.get("gpu_type"),
        "n_cpus": values.get("n_cpus"),
        "memory": values.get("memory"),
        "max_batch_tokens": values.get("max_batch_tokens"),
        "gpu_memory_utilization": values.get("gpu_memory_utilization"),
        "dtype": values.get("dtype"),
        "cpu_offload": values.get("cpu_offload"),
        "max_num_seq": values.get("max_num_seq"),
    }
    return json.dumps(env_values)


def _create_environment(
    values: dict[str, str],
    actuator: VLLMPerformanceTestParameters,
    node_selector: dict[str, str],
    env_manager: ActorHandle,
    check_interval: int = 5,
    timeout: int = 1200,
) -> tuple[str, str, str]:
    """
     Create environment
     :param values: experiment values
     :param actuator: actuator parameters
     :param node_selector: node selector
     :param env_manager: environment manager
     :param check_interval: wait interval
     :param timeout: timeout
    :return: kubernetes environment name
    """
    # get model for experiment
    model = values.get("model")
    # create environment definition
    definition = _build_entity_env(values=values)
    while True:
        env = ray.get(
            env_manager.get_environment.remote(
                model=model, definition=definition, increment_usage=True
            )
        )
        if env is not None:
            break
        time.sleep(check_interval)
    error = None
    print(f"Environment state {env.state}, name {env.k8_name}, definition {definition}")
    start = time.time()
    match env.state:
        case EnvironmentState.NONE:
            # Environment does not exist, create it
            print(f"Environment {env.k8_name} does not exist. Creating it")
            tmout = 1
            for attempt in range(3):
                try:
                    create_test_environment(
                        k8_name=env.k8_name,
                        model=model,
                        in_cluster=actuator.in_cluster,
                        verify_ssl=actuator.verify_ssl,
                        image=values.get("image"),
                        image_secret=actuator.image_secret,
                        deployment_template=actuator.deployment_template,
                        service_template=actuator.service_template,
                        pvc_template=actuator.pvc_template,
                        n_gpus=int(values.get("n_gpus")),
                        gpu_type=values.get("gpu_type"),
                        node_selector=node_selector,
                        n_cpus=int(values.get("n_cpus")),
                        memory=values.get("memory"),
                        max_batch_tokens=int(values.get("max_batch_tokens")),
                        gpu_memory_utilization=float(
                            values.get("gpu_memory_utilization")
                        ),
                        dtype=VLLMDtype(values.get("dtype", "auto")),
                        cpu_offload=int(values.get("cpu_offload")),
                        max_num_seq=int(values.get("max_num_seq")),
                        hf_token=actuator.hf_token,
                        reuse_service=False,
                        reuse_deployment=False,
                        pvc_name=actuator.pvc_template,
                        namespace=actuator.namespace,
                    )
                    # Update manager
                    env_manager.done_creating.remote(definition=definition)
                    error = None
                    break
                except Exception as e:
                    logger.error(
                        f"Attempt {attempt}. Failed to create test environment {e}"
                    )
                    error = f"Failed to create test environment {e}"
                    time.sleep(tmout)
                    tmout *= 2
            if error is None:
                print(
                    f"Created test environment {env.k8_name} in {time.time() - start} sec"
                )
            else:
                print(f"Failed to create test environment {env.k8_name} exiting")
                sys.exit(1)
        case EnvironmentState.CREATING:
            # Someone is creating environment, wait till its ready
            print(
                f"Environment {env.k8_name} is being created. Waiting for it to be ready."
            )
            n_checks = math.ceil(timeout / check_interval)
            for _ in range(n_checks):
                time.sleep(check_interval)
                env = ray.get(
                    env_manager.get_environment.remote(
                        model=model, definition=definition
                    )
                )
                if env.state == EnvironmentState.READY:
                    break
            if env.state != EnvironmentState.READY:
                # timed out waiting for environment creation
                error = "Timed out waiting for environment to get ready"
            print("Environment is created, using it")
        case _:
            # environment exists, use it
            print(f"Environment {env.k8_name} already exists. Reusing it")
    return env.k8_name, error, definition


@ray.remote
def run_resource_and_workload_experiment(
    request: MeasurementRequest,
    experiment: Experiment | ParameterizedExperiment,
    state_update_queue: MeasurementQueue,
    actuator_parameters: VLLMPerformanceTestParameters,
    node_selector: dict[str, str],
    env_manager: ActorHandle,
    local_port: int,
):
    """
    Runs an experiment on a specific compute resource and inference workload configuration.

    This requires spinning up a vLLM instance with the given compute resources

    :param request: measurement request
    :param experiment: definition of experiment
    :param state_update_queue: update queue
    :param actuator_parameters: actuator parameters
    :param node_selector: node selector
    :param env_manager: environment manager
    :param local_port: local port to use
    :return:
    """

    # This function
    # 1. Performs the measurement represented by MeasurementRequest
    # 2. Updates MeasurementRequest with the results of the measurement and status
    # 3. Puts it in the stateUpdateQueue

    print(
        f"number of entities {len(request.entities)}, actuator parameters {actuator_parameters}, node selector {node_selector}"
    )

    # placeholder for measurements
    measurements = []
    current_port = local_port - 1
    # For every entity
    for entity in request.entities:
        #
        # For each Entity you need to retrieve the values required to run the requested experiment:
        # - One set of values will be retrieved from the Entity
        # - If the experiment was parameterizable another set may be retrieved from the Experiment

        values = get_experiment_input_values(experiment=experiment, entity=entity)
        print(
            f"Values for entity {entity.identifier} and experiment {experiment.identifier} "
            f"experiment type is {type(experiment)} are {json.dumps(values)}"
        )
        current_port += 1
        # create environment
        if not actuator_parameters.in_cluster:
            print("We are running locally connecting to remote cluster")
            print("please make sure that you have executed `oc login`")
            print(
                "We are using ports from 10000 and above to communicate with the cluster, "
                "please make sure that it is not in use"
            )
        measured_values = []
        k8_name, error, definition = _create_environment(
            values=values,
            actuator=actuator_parameters,
            node_selector=node_selector,
            env_manager=env_manager,
        )
        if error is None:
            print("test environment is created")
            pf = None
            # compute base url
            if actuator_parameters.in_cluster:
                # we are running in cluster, connect to service directly
                base_url = f"http://{k8_name}.{actuator_parameters.namespace}.svc.cluster.local:80"
            else:
                # we are running locally. need to do port-forward and connect to the local one
                pf_command = f"kubectl port-forward svc/{k8_name} -n {actuator_parameters.namespace} {current_port}:80"
                try:
                    pf = subprocess.Popen(pf_command, shell=True)
                    print("port forwarding set")
                    # make sure that port forwarding is up
                    time.sleep(5)
                except Exception as e:
                    print(f"failed to start port forward to service {k8_name} - {e}")
                    error = f"failed to start port forward to service {k8_name} - {e}"
                base_url = f"http://localhost:{current_port}"
            if error is None:
                request_rate = int(values.get("request_rate"))
                if request_rate < 0:
                    request_rate = None
                max_concurrency = int(values.get("max_concurrency"))
                if max_concurrency < 0:
                    max_concurrency = None
                start = time.time()
                result = None
                try:
                    result = execute_benchmark(
                        base_url=base_url,
                        model=values.get("model"),
                        data_set=values.get("dataset"),
                        interpreter=actuator_parameters.interpreter,
                        num_prompts=int(values.get("num_prompts")),
                        request_rate=request_rate,
                        max_concurrency=max_concurrency,
                        hf_token=actuator_parameters.hf_token,
                        benchmark_retries=actuator_parameters.benchmark_retries,
                        retries_timeout=actuator_parameters.retries_timeout,
                    )
                    print(f"benchmark executed in {time.time() - start} sec")
                except Exception as e:
                    logger.error(f"Failed to execute VLLM performance test {e}")
                    error = f"Failed to execute VLLM performance test {e}"
                finally:
                    if pf is not None:
                        pf.kill()
                    env_manager.done_using.remote(definition=definition)
                if error is None:
                    measured_values = dict_to_measurements(
                        results=result, experiment=experiment
                    )
                    logger.debug(f"measured values {measured_values}")

        # Create a MeasurementResult to hold the results
        # This is used to
        # (a) Separate results from multiple entities
        # (b) Distinguish Valid and Invalid measurements -> especially in latter case to provide info on failure reasons

        print("creating measurement")
        measurements.append(
            create_measurement_result(
                identifier=entity.identifier,
                measurements=measured_values,
                error=error,
                reference=request.experimentReference,
            )
        )

    # For multi entity experiments if ONE entity had ValidResults the status must be SUCCESS
    print("done processing entities")
    if len(measurements) > 0:
        request.measurements = measurements
    request.status = compute_measurement_status(measurements=measurements)
    print(f"request status is {request.status}. pushing to update queue")
    # Push the request to the state updates queue
    state_update_queue.put(request, block=False)


@ray.remote
def run_workload_experiment(
    request: MeasurementRequest,
    experiment: Experiment | ParameterizedExperiment,
    state_update_queue: MeasurementQueue,
    actuator_parameters: VLLMPerformanceTestParameters,
):
    """
    Runs an experiment with a specific inference workload configuration on a given endpoint.

    The compute resource associated with the end-point is not known.

    :param request: measurement request
    :param experiment: definition of experiment
    :param state_update_queue: update queue
    :param actuator_parameters: actuator parameters
    :return:
    """

    # This function
    # 1. Performs the measurement represented by MeasurementRequest
    # 2. Updates MeasurementRequest with the results of the measurement and status
    # 3. Puts it in the stateUpdateQueue

    # placeholder for measurements
    measurements = []
    # For every entity
    for entity in request.entities:
        #
        # For each Entity you need to retrieve the values required to run the requested experiment:
        # - One set of values will be retrieved from the Entity
        # - If the experiment was parameterizable another set may be retrieved from the Experiment

        values = get_experiment_input_values(experiment=experiment, entity=entity)
        print(
            f"Values for entity {entity.identifier} and experiment {experiment.identifier} "
            f"experiment type is {type(experiment)} are {json.dumps(values)}"
        )

        request_rate = int(values.get("request_rate"))
        if request_rate < 0:
            request_rate = None
        max_concurrency = int(values.get("max_concurrency"))
        if max_concurrency < 0:
            max_concurrency = None
        start = time.time()
        result = None
        error = None
        measured_values = []
        try:
            result = execute_benchmark(
                base_url=values.get("endpoint"),
                model=values.get("model"),
                data_set=values.get("dataset"),
                interpreter=actuator_parameters.interpreter,
                num_prompts=int(values.get("num_prompts")),
                request_rate=request_rate,
                max_concurrency=max_concurrency,
                hf_token=actuator_parameters.hf_token,
                benchmark_retries=actuator_parameters.benchmark_retries,
                retries_timeout=actuator_parameters.retries_timeout,
            )
            print(f"benchmark executed in {time.time() - start} sec")
        except Exception as e:
            logger.error(f"Failed to execute VLLM performance test {e}")
            error = f"Failed to execute VLLM performance test {e}"

        if error is None:
            measured_values = dict_to_measurements(
                results=result, experiment=experiment
            )
            logger.debug(f"measured values {measured_values}")

        # Create a MeasurementResult to hold the results
        # This is used to
        # (a) Separate results from multiple entities
        # (b) Distinguish Valid and Invalid measurements -> especially in latter case to provide info on failure reasons

        print("creating measurement")
        measurements.append(
            create_measurement_result(
                identifier=entity.identifier,
                measurements=measured_values,
                error=error,
                reference=request.experimentReference,
            )
        )

    # For multi entity experiments if ONE entity had ValidResults the status must be SUCCESS
    print("done processing entities")
    if len(measurements) > 0:
        request.measurements = measurements
    request.status = compute_measurement_status(measurements=measurements)
    print(f"request status is {request.status}. pushing to update queue")
    # Push the request to the state updates queue
    state_update_queue.put(request, block=False)
