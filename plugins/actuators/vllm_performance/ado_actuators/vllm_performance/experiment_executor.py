# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import logging
import math
import subprocess
import time

import ray
from ado_actuators.vllm_performance.actuator_parameters import (
    VLLMPerformanceTestParameters,
)
from ado_actuators.vllm_performance.env_manager import (
    Environment,
    EnvironmentManager,
    EnvironmentState,
)
from ado_actuators.vllm_performance.k8s.create_environment import (
    create_test_environment,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    VLLMDtype,
)
from ado_actuators.vllm_performance.vllm_performance_test.execute_benchmark import (
    VLLMBenchmarkError,
    execute_random_benchmark,
)
from ray.actor import ActorHandle

from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.request import MeasurementRequest
from orchestrator.utilities.support import (
    compute_measurement_status,
    create_measurement_result,
    dict_to_measurements,
)

logger = logging.getLogger(__name__)


class K8EnvironmentCreationError(Exception):
    """Error raised when K8 environment cannot be created for some reason"""


class K8ConnectionError(Exception):
    """Error raised when there is an issue connecting to K8s or a service its hosting"""


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
    env_manager: ActorHandle[EnvironmentManager],
    check_interval: int = 5,
    timeout: int = 1200,
) -> tuple[str, str]:
    """
     Create environment

     Important: This function will block until env_manager.get_environment
     returns an environment.
     The env_manager will not return an environment until there is one free
     to be used

     :param values: experiment values
     :param actuator: actuator parameters
     :param node_selector: node selector
     :param env_manager: environment manager
     :param check_interval: wait interval
     :param timeout: timeout
    :return: kubernetes environment name

    :raises K8EnvironmentCreationError if there was an issue
    - If the creation step fails after three attempts
    - If after creation the environment was not in ready state after timeout seconds (1200 default)

    """
    # get model for experiment
    model = values.get("model")

    # create environment definition
    definition = _build_entity_env(values=values)
    while True:
        env: Environment = ray.get(
            env_manager.get_environment.remote(
                model=model, definition=definition, increment_usage=True
            )
        )
        if env is not None:
            break
        time.sleep(check_interval)

    error = None
    logger.debug(
        f"Environment state {env.state}, name {env.k8s_name}, definition {definition}"
    )

    start = time.time()

    # We retrieve the PVC name from the actor because it is one to be shared for the whole experiment
    pvc_name = ray.get(env_manager.get_experiment_pvc_name.remote())

    match env.state:
        case EnvironmentState.NONE:
            # Environment does not exist, create it
            logger.debug(f"Environment {env.k8s_name} does not exist. Creating it")
            tmout = 1
            for attempt in range(3):
                try:
                    create_test_environment(
                        k8s_name=env.k8s_name,
                        model=model,
                        in_cluster=actuator.in_cluster,
                        verify_ssl=actuator.verify_ssl,
                        image=values.get("image"),
                        image_secret=actuator.image_secret,
                        deployment_template=actuator.deployment_template,
                        service_template=actuator.service_template,
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
                        namespace=actuator.namespace,
                        pvc_name=pvc_name,
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

            # Check if error after three attempts
            if error is None:
                logger.info(
                    f"Created test environment {env.k8s_name} in {time.time() - start} sec"
                )
            else:
                raise K8EnvironmentCreationError(
                    f"Failed to create test environment {env.k8s_name}: {error}"
                )

        case EnvironmentState.CREATING:
            # Someone is creating environment, wait till its ready
            logger.info(
                f"Environment {env.k8s_name} is being created. Waiting for it to be ready."
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
                error = (
                    f"Timed out waiting for environment to get ready. Timeout {timeout}"
                )
                raise K8EnvironmentCreationError(
                    f"Failed to create test environment {env.k8s_name}: {error}"
                )

            logger.debug("Environment is created, using it")
        case _:
            # environment exists, use it
            logger.debug(f"Environment {env.k8s_name} already exists. Reusing it")

    return env.k8s_name, definition


def _connect_to_vllm_server(
    k8s_name: str,
    actuator_parameters: VLLMPerformanceTestParameters,
    port: int,
) -> tuple[str, subprocess.Popen | None]:
    """Returns the URL of the vLLM inference server

    Creates a port forward for the inference server if test
    is not running on the cluster with the service

    Parameters:
        k8s_name: The name of the vLLM service
        actuator_parameters: VLLMPerformanceTestParameters instance containing
            namespace and test location (in_cluster or not) information

    Returns:
        A tuple containing
        - The URL of the created vLLM server
        - If a port-forward is created the POpen object for the port-forward
          Otherwise None

    Raise:
        K8ConnectionError if a port-forward could not be created
    """

    # create environment
    if not actuator_parameters.in_cluster:
        logger.info("We are running locally connecting to remote cluster")
        logger.info("please make sure that you have executed `oc login`")
        logger.info(
            "We are using ports from 10000 and above to communicate with the cluster, "
            "please make sure that it is not in use"
        )

    if actuator_parameters.in_cluster:
        # we are running in cluster, connect to service directly
        base_url = (
            f"http://{k8s_name}.{actuator_parameters.namespace}.svc.cluster.local:80"
        )
        pf = None
    else:
        # we are running locally. need to do port-forward and connect to the local one
        pf_command = f"kubectl port-forward svc/{k8s_name} -n {actuator_parameters.namespace} {port}:80"
        try:
            pf = subprocess.Popen(pf_command, shell=True)
            # make sure that port forwarding is up
            time.sleep(5)
        except Exception as e:
            logger.warning(f"failed to start port forward to service {k8s_name} - {e}")
            raise K8ConnectionError(
                f"failed to start port forward to service {k8s_name} - {e}"
            )

        base_url = f"http://localhost:{port}"

    return base_url, pf


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

    logger.debug(
        f"number of entities {len(request.entities)}, actuator parameters {actuator_parameters}, node selector {node_selector}"
    )

    # placeholder for measurements
    measurements = []
    current_port = local_port - 1
    # For every entity
    for entity in request.entities:

        try:
            values = experiment.propertyValuesFromEntity(entity=entity)

            logger.info(f"Creating K8s environment for {entity.identifier}")

            # Will raise an K8EnvironmentCreationError if the environment could not be created
            k8s_name, definition = _create_environment(
                values=values,
                actuator=actuator_parameters,
                node_selector=node_selector,
                env_manager=env_manager,
            )

            # Will raise an K8ConnectionError if a port-forward was required
            # but could not be created
            current_port += 1
            base_url, port_forward = _connect_to_vllm_server(
                k8s_name, actuator_parameters, current_port
            )

            logger.info(f"Will use vllm server at {base_url}")

            request_rate = int(values.get("request_rate"))
            if request_rate < 0:
                request_rate = None
            max_concurrency = int(values.get("max_concurrency"))
            if max_concurrency < 0:
                max_concurrency = None
            start = time.time()
            result = execute_random_benchmark(
                base_url=base_url,
                model=values.get("model"),
                interpreter=actuator_parameters.interpreter,
                num_prompts=int(values.get("num_prompts")),
                request_rate=request_rate,
                max_concurrency=max_concurrency,
                hf_token=actuator_parameters.hf_token,
                benchmark_retries=actuator_parameters.benchmark_retries,
                retries_timeout=actuator_parameters.retries_timeout,
                number_input_tokens=int(values.get("number_input_tokens")),
                max_output_tokens=int(values.get("max_output_tokens")),
                burstiness=float(values.get("burstiness")),
            )

            logger.debug(f"benchmark executed in {time.time() - start} sec")
            if port_forward is not None:
                port_forward.kill()
            env_manager.done_using.remote(definition=definition)
        except (
            K8EnvironmentCreationError,
            K8ConnectionError,
            VLLMBenchmarkError,
        ) as error:
            logger.error(f"Error running tests for entity {entity.identifier}: {error}")
            measurements.append(
                create_measurement_result(
                    identifier=entity.identifier,
                    measurements=[],
                    error=str(error),
                    reference=request.experimentReference,
                )
            )
        except Exception as error:
            logger.critical(f"Unexpected error for entity {entity.identifier}: {error}")
            measurements.append(
                create_measurement_result(
                    identifier=entity.identifier,
                    measurements=[],
                    error=f"Unexpected error in experiment execution: {error}",
                    reference=request.experimentReference,
                )
            )
        else:
            measured_values = dict_to_measurements(
                results=result, experiment=experiment
            )
            measurements.append(
                create_measurement_result(
                    identifier=entity.identifier,
                    measurements=measured_values,
                    error=None,
                    reference=request.experimentReference,
                )
            )

    # For multi entity experiments if ONE entity had ValidResults the status must be SUCCESS
    if len(measurements) > 0:
        request.measurements = measurements
    request.status = compute_measurement_status(measurements=measurements)
    logger.debug(f"request status is {request.status}. pushing to update queue")
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
        measured_values = []
        error = None
        try:
            values = experiment.propertyValuesFromEntity(entity=entity)
            logger.debug(
                f"Values for entity {entity.identifier} and experiment {experiment.identifier} "
                f"experiment type is {type(experiment)} are {json.dumps(values)}"
            )

            request_rate = int(values.get("request_rate"))
            if request_rate < 0:
                request_rate = None
            max_concurrency = int(values.get("max_concurrency"))
            if max_concurrency < 0:
                max_concurrency = None

            # Will raise VLLMBenchmarkError if there is a problem
            result = execute_random_benchmark(
                base_url=values.get("endpoint"),
                model=values.get("model"),
                interpreter=actuator_parameters.interpreter,
                num_prompts=int(values.get("num_prompts")),
                request_rate=request_rate,
                max_concurrency=max_concurrency,
                hf_token=actuator_parameters.hf_token,
                benchmark_retries=actuator_parameters.benchmark_retries,
                retries_timeout=actuator_parameters.retries_timeout,
                number_input_tokens=int(values.get("number_input_tokens")),
                max_output_tokens=int(values.get("max_output_tokens")),
                burstiness=float(values.get("burstiness")),
            )
        except VLLMBenchmarkError as e:
            error = f"Encountered benchmark error when testing entity {entity.identifier}: {e}"
            logger.error(error)
        except Exception as e:
            error = f"Unexpected error for entity {entity.identifier}: {e}"
            logger.error(error)
        else:
            measured_values = dict_to_measurements(
                results=result, experiment=experiment
            )
            logger.debug(f"measured values {measured_values}")
        finally:
            measurements.append(
                create_measurement_result(
                    identifier=entity.identifier,
                    measurements=measured_values,
                    error=error,
                    reference=request.experimentReference,
                )
            )

    # For multi entity experiments if ONE entity had ValidResults the status must be SUCCESS
    if len(measurements) > 0:
        request.measurements = measurements
    request.status = compute_measurement_status(measurements=measurements)
    logger.debug(f"request status is {request.status}. pushing to update queue")
    # Push the request to the state updates queue
    state_update_queue.put(request, block=False)
