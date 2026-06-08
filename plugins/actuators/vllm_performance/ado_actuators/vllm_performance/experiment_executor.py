# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import json
import logging
import subprocess
import time
import traceback

import ray
from ado_actuators.vllm_performance.actuator_parameters import (
    VLLMPerformanceTestParameters,
)
from ado_actuators.vllm_performance.cache_utils import CacheKeyBuilder
from ado_actuators.vllm_performance.env_manager import (
    Environment,
    EnvironmentManager,
    EnvironmentState,
)
from ado_actuators.vllm_performance.k8s import (
    K8sConnectionError,
    K8sEnvironmentCreationError,
)
from ado_actuators.vllm_performance.k8s.create_environment import (
    create_test_environment,
)
from ado_actuators.vllm_performance.k8s.yaml_support.build_components import (
    VLLMDtype,
)
from ado_actuators.vllm_performance.version_utils import VLLMVersionChecker
from ado_actuators.vllm_performance.vllm_performance_test.benchmark_models import (
    BenchmarkParameters,
    BenchmarkResult,
)
from ado_actuators.vllm_performance.vllm_performance_test.execute_benchmark import (
    VLLMBenchmarkError,
    execute_geospatial_benchmark,
    execute_random_benchmark,
)
from ado_actuators.vllm_performance.vllm_performance_test.execute_guidellm_benchmark import (
    execute_guidellm_benchmark,
    execute_guidellm_geospatial_benchmark,
)
from ray.actor import ActorHandle

from orchestrator.modules.actuators.measurement_queue import MeasurementQueue
from orchestrator.modules.operators.console_output import RichConsoleSpinnerMessage
from orchestrator.schema.experiment import Experiment, ParameterizedExperiment
from orchestrator.schema.request import MeasurementRequest
from orchestrator.utilities.support import (
    compute_measurement_status,
    create_measurement_result,
)

logger = logging.getLogger(__name__)


def _create_environment(
    values: dict[str, str],
    actuator: VLLMPerformanceTestParameters,
    node_selector: dict[str, str],
    request_id: str,
    env_manager: ActorHandle[EnvironmentManager],
    experiment: Experiment | ParameterizedExperiment,
    check_interval: int = 5,
    timeout: int = 1200,
) -> tuple[str, str]:
    """Create environment with version-aware threadpool support.

    Blocks until env_manager returns an available environment.
    Raises K8sEnvironmentCreationError if creation fails after 3 attempts or timeout.
    """
    from orchestrator.modules.operators.console_output import (
        RichConsoleSpinnerMessage,
    )

    console = ray.get_actor(name="RichConsoleQueue")
    environment_usage = ray.get(env_manager.environment_usage.remote())

    # get model for experiment
    model = values.get("model")

    # create environment definition
    definition = CacheKeyBuilder.build_env_definition(values=values)
    console.put.remote(
        message=RichConsoleSpinnerMessage(
            id=request_id,
            label=f"({request_id}) Waiting for deployment environment slot to be available - total slots {environment_usage.get('max')}",
            state="start",
        )
    )
    while True:
        try:
            env: Environment = ray.get(
                env_manager.get_environment.remote(model=model, definition=definition)
            )
        except Exception as e:
            raise e
        if env is not None:
            console.put.remote(
                message=RichConsoleSpinnerMessage(
                    id=request_id,
                    label=f"{request_id} Got environment slot {env.k8s_name}",
                    state="stop",
                )
            )
            break

        ray.get(env_manager.wait_for_env.remote())

    error = None
    logger.debug(
        f"Environment state {env.state}, name {env.k8s_name}, definition {definition}"
    )

    start = time.time()

    # We retrieve the PVC name from the actor because it is one to be shared for the whole experiment
    pvc_name = ray.get(env_manager.get_experiment_pvc_name.remote())
    otlp_traces_endpoint = ray.get(env_manager.get_otlp_traces_endpoint.remote())

    match env.state:
        case EnvironmentState.NONE:
            # Environment does not exist, create it
            logger.debug(f"Environment {env.k8s_name} does not exist. Creating it")
            tmout = 1
            ray.get(
                env_manager.wait_deployment_before_starting.remote(
                    env=env, request_id=request_id
                )
            )

            for attempt in range(3):
                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request_id,
                        label=f"({request_id}) Creating vLLM deployment {env.k8s_name} (attempt {attempt + 1}/3)...",
                        state="start",
                    )
                )
                try:
                    image_value = values.get("image", "")
                    threadpool_requested = int(values.get("threadpool", 1))
                    enable_threadpool = VLLMVersionChecker.supports_threadpool(
                        image_value, threadpool_requested
                    )
                    threadpool_value = 1 if enable_threadpool else 0

                    if isinstance(image_value, list):
                        image_name = image_value[0] if len(image_value) > 0 else ""
                    else:
                        image_name = image_value

                    create_test_environment(
                        k8s_name=env.k8s_name,
                        model=model,
                        in_cluster=actuator.in_cluster,
                        verify_ssl=actuator.verify_ssl,
                        image=image_name,
                        image_pull_secret_name=actuator.image_pull_secret_name,
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
                        namespace=actuator.namespace,
                        pvc_name=pvc_name,
                        skip_tokenizer_init=values.get("skip_tokenizer_init", 0) == 1,
                        enforce_eager=values.get("enforce_eager", 0) == 1,
                        io_processor_plugin=values.get("io_processor_plugin"),
                        otlp_traces_endpoint=otlp_traces_endpoint,
                        threadpool=threadpool_value,
                        renderer_num_workers=int(values.get("renderer_num_workers")),
                        check_interval=check_interval,
                        timeout=timeout,
                    )
                    env_manager.done_creating.remote(identifier=env.k8s_name)
                    error = None
                    break
                except Exception as e:
                    logger.error(
                        f"Attempt {attempt}. Failed to create test environment {e}"
                    )
                    logger.error(traceback.format_exception(e))
                    error = f"Failed to create test environment {e}"
                    time.sleep(tmout)
                    tmout *= 2

            if error is None:
                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request_id,
                        label=f"({request_id})  Created vLLM deployment {env.k8s_name}",
                        state="stop",
                    )
                )
                logger.info(
                    f"Created test environment {env.k8s_name} in {time.time() - start} sec"
                )
            else:
                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request_id,
                        label=f"({request_id}) Failed to create {env.k8s_name}. Aborting.",
                        state="stop",
                    )
                )

                ray.get(
                    env_manager.cleanup_failed_deployment.remote(
                        identifier=env.k8s_name
                    )
                )

                raise K8sEnvironmentCreationError(
                    f"Failed to create test environment {env.k8s_name}: {error}"
                )

    return env.k8s_name, definition


def _connect_to_vllm_server(
    k8s_name: str,
    actuator_parameters: VLLMPerformanceTestParameters,
    port: int,
) -> tuple[str, subprocess.Popen | None]:
    """Returns vLLM server URL and optional port-forward process.

    Creates port-forward if not running in-cluster.
    Raises K8sConnectionError if port-forward fails.
    """
    if not actuator_parameters.in_cluster:
        logger.info("We are running locally connecting to remote cluster")
        logger.info("please make sure that you have executed `oc login`")
        logger.info(
            "We are using ports from 10000 and above to communicate with the cluster, "
            "please make sure that it is not in use"
        )

    if actuator_parameters.in_cluster:
        base_url = (
            f"http://{k8s_name}.{actuator_parameters.namespace}.svc.cluster.local:80"
        )
        pf = None
    else:
        pf_command_args = [
            "kubectl",
            "port-forward",
            f"svc/{k8s_name}",
            "-n",
            f"{actuator_parameters.namespace}",
            f"{port}:80",
        ]
        try:
            pf = subprocess.Popen(  # noqa: S603 - namespace is sanitized to be RFC1123
                pf_command_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)
            if pf.returncode:
                raise K8sConnectionError(
                    f"failed to start port forward to service {k8s_name} - port-forward command exited for unknown reason. Check logs."
                )
        except Exception as e:
            logger.warning(f"failed to start port forward to service {k8s_name} - {e}")
            raise K8sConnectionError(
                f"failed to start port forward to service {k8s_name} - {e}"
            ) from e

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
) -> None:
    """Run experiment on specific compute resource and workload configuration."""
    measurements = []
    current_port = local_port - 1
    console = ray.get_actor(name="RichConsoleQueue")

    for entity in request.entities:
        port_forward = None
        definition = None
        started_benchmarking = False
        try:
            values = experiment.propertyValuesFromEntity(entity=entity)

            cache_key = CacheKeyBuilder.build(values)
            logger.info("cache_key: %s", cache_key)

            cached_result = ray.get(
                env_manager.get_cached_measurement.remote(cache_key)
            )
            if cached_result is not None:
                logger.info(
                    f"Reusing cached measurement for entity {entity.identifier} "
                    f"(identical environment and benchmark parameters)"
                )
                measurements.append(
                    create_measurement_result(
                        identifier=entity.identifier,
                        measurements=cached_result.measurements,
                        error=cached_result.error,
                        reference=request.experimentReference,
                    )
                )
                continue

            logger.info(f"Creating K8s environment for {entity.identifier}")

            k8s_name, definition = _create_environment(
                values=values,
                actuator=actuator_parameters,
                node_selector=node_selector,
                env_manager=env_manager,
                experiment=experiment,
                request_id=request.requestid,
            )

            current_port += 1
            base_url, port_forward = _connect_to_vllm_server(
                k8s_name, actuator_parameters, current_port
            )

            logger.info(f"Will use vllm server at {base_url}")

            benchmark_parameters = BenchmarkParameters.model_validate(values)
            benchmark_parameters.endpoint = base_url

            started_benchmarking = True
            console.put.remote(
                message=RichConsoleSpinnerMessage(
                    id=request.requestid,
                    label=f"({request.requestid}) Executing benchmark",
                    state="start",
                )
            )
            logger.info(f"Executing experiment: {experiment.identifier}")
            result: BenchmarkResult
            if experiment.identifier in [
                "test-geospatial-deployment-v1",
                "test-geospatial-deployment-custom-dataset-v1",
                "test-geospatial-endpoint-custom-dataset-v1",
            ]:
                logger.info("Using geospatial benchmark for deployment")
                result = execute_geospatial_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    burstiness=benchmark_parameters.burstiness,
                    dataset=benchmark_parameters.dataset,
                )
            elif experiment.identifier in [
                "test-geospatial-deployment-guidellm-v1",
                "test-geospatial-deployment-guidellm-custom-dataset-v1",
            ]:
                logger.info("Using GuideLLM geospatial benchmark for deployment")
                result = execute_guidellm_geospatial_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    dataset=benchmark_parameters.dataset,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    burstiness=benchmark_parameters.burstiness,
                )
            elif experiment.identifier == "test-deployment-guidellm-v1":
                logger.info("Using GuideLLM benchmark for deployment")
                result = execute_guidellm_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    number_input_tokens=benchmark_parameters.number_input_tokens,
                    max_output_tokens=benchmark_parameters.max_output_tokens,
                    dataset=benchmark_parameters.dataset,
                    burstiness=benchmark_parameters.burstiness,
                )
            else:
                logger.info("Using vLLM random benchmark for deployment")
                result = execute_random_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    number_input_tokens=benchmark_parameters.number_input_tokens,
                    max_output_tokens=benchmark_parameters.max_output_tokens,
                    burstiness=benchmark_parameters.burstiness,
                    dataset=benchmark_parameters.dataset,
                )

        except (
            K8sEnvironmentCreationError,
            K8sConnectionError,
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
            measured_values = result.to_observed_property_values(experiment=experiment)
            measurement_result = create_measurement_result(
                identifier=entity.identifier,
                measurements=measured_values,
                error=None,
                reference=request.experimentReference,
            )
            measurements.append(measurement_result)
            env_manager.cache_measurement.remote(cache_key, measured_values, None)
        finally:
            if started_benchmarking:
                console.put.remote(
                    message=RichConsoleSpinnerMessage(
                        id=request.requestid,
                        label=f"({request.requestid}) Completed benchmark",
                        state="stop",
                    )
                )
            if port_forward is not None:
                port_forward.kill()
            if definition is not None:
                env_manager.done_using.remote(identifier=k8s_name)

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
) -> None:
    """Run experiment with specific workload configuration on given endpoint."""
    measurements = []
    for entity in request.entities:
        measured_values = []
        error = None
        try:
            values = experiment.propertyValuesFromEntity(entity=entity)
            logger.debug(
                f"Values for entity {entity.identifier} and experiment {experiment.identifier} "
                f"experiment type is {type(experiment)} are {json.dumps(values)}"
            )

            benchmark_parameters = BenchmarkParameters.model_validate(values)

            logger.info(f"Executing experiment: {experiment.identifier}")
            result: BenchmarkResult
            if experiment.identifier in [
                "test-geospatial-endpoint-v1",
                "test-geospatial-endpoint-custom-dataset-v1",
            ]:
                logger.info("Using geospatial benchmark for endpoint")
                result = execute_geospatial_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    burstiness=benchmark_parameters.burstiness,
                    dataset=benchmark_parameters.dataset,
                )
            elif experiment.identifier in [
                "test-geospatial-endpoint-guidellm-v1",
                "test-geospatial-endpoint-guidellm-custom-dataset-v1",
            ]:
                logger.info("Using GuideLLM geospatial benchmark for endpoint")
                result = execute_guidellm_geospatial_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    dataset=benchmark_parameters.dataset,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    burstiness=benchmark_parameters.burstiness,
                )
            elif experiment.identifier == "test-endpoint-guidellm-v1":
                logger.info("Using GuideLLM benchmark for endpoint")
                result = execute_guidellm_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    number_input_tokens=benchmark_parameters.number_input_tokens,
                    max_output_tokens=benchmark_parameters.max_output_tokens,
                    dataset=benchmark_parameters.dataset,
                    burstiness=benchmark_parameters.burstiness,
                )
            else:
                logger.info("Using vLLM random benchmark for endpoint")
                result = execute_random_benchmark(
                    base_url=benchmark_parameters.endpoint,
                    model=benchmark_parameters.model,
                    num_prompts=benchmark_parameters.num_prompts,
                    request_rate=benchmark_parameters.request_rate,
                    max_concurrency=benchmark_parameters.max_concurrency,
                    hf_token=actuator_parameters.hf_token,
                    benchmark_retries=actuator_parameters.benchmark_retries,
                    retries_timeout=actuator_parameters.retries_timeout,
                    number_input_tokens=benchmark_parameters.number_input_tokens,
                    max_output_tokens=benchmark_parameters.max_output_tokens,
                    burstiness=benchmark_parameters.burstiness,
                    dataset=benchmark_parameters.dataset,
                )
        except VLLMBenchmarkError as e:
            error = f"Encountered benchmark error when testing entity {entity.identifier}: {e}"
            logger.error(error)
        except Exception as e:
            error = f"Unexpected error for entity {entity.identifier}: {e}"
            logger.error(error)
        else:
            measured_values = result.to_observed_property_values(experiment=experiment)
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

    if len(measurements) > 0:
        request.measurements = measurements
    request.status = compute_measurement_status(measurements=measurements)
    logger.debug(f"request status is {request.status}. pushing to update queue")
    # Push the request to the state updates queue
    state_update_queue.put(request, block=False)
