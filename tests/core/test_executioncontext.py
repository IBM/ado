# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pydantic
import pytest
import yaml

from orchestrator.core.remotecontext.config import (
    ClusterExecutionType,
    JobExecutionType,
    PackageConfiguration,
    PortForwardConfiguration,
    RemoteExecutionContext,
)
from orchestrator.utilities.output import pydantic_model_as_yaml

# ---------------------------------------------------------------------------
# PortForwardConfiguration
# ---------------------------------------------------------------------------


def test_port_forward_configuration_defaults() -> None:
    """Default localPort is 8265."""
    pf = PortForwardConfiguration(namespace="my-ns", serviceName="my-ray-svc")
    assert pf.namespace == "my-ns"
    assert pf.serviceName == "my-ray-svc"
    assert pf.localPort == 8265


def test_port_forward_configuration_custom_port() -> None:
    pf = PortForwardConfiguration(namespace="ns", serviceName="svc", localPort=9000)
    assert pf.localPort == 9000


def test_port_forward_configuration_lifecycle() -> None:
    """Create → dump → reload produces identical object."""
    pf = PortForwardConfiguration(namespace="ns", serviceName="svc", localPort=9999)
    reloaded = PortForwardConfiguration.model_validate(pf.model_dump())
    assert reloaded == pf


def test_port_forward_configuration_rejects_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        PortForwardConfiguration(namespace="ns", serviceName="svc", unexpected="x")


# ---------------------------------------------------------------------------
# ClusterExecutionType
# ---------------------------------------------------------------------------


def test_cluster_execution_type_minimal() -> None:
    """type defaults to 'cluster' and portForward defaults to None."""
    cluster = ClusterExecutionType(clusterUrl="http://localhost:8265")
    assert cluster.type == "cluster"
    assert str(cluster.clusterUrl) == "http://localhost:8265/"
    assert cluster.portForward is None


def test_cluster_execution_type_with_port_forward() -> None:
    cluster = ClusterExecutionType(
        clusterUrl="http://localhost:8265",
        portForward=PortForwardConfiguration(namespace="my-ns", serviceName="my-svc"),
    )
    assert cluster.portForward is not None
    assert cluster.portForward.namespace == "my-ns"


def test_cluster_execution_type_lifecycle_without_port_forward() -> None:
    cluster = ClusterExecutionType(clusterUrl="http://ray.example.com:8265")
    reloaded = ClusterExecutionType.model_validate(cluster.model_dump())
    assert reloaded == cluster


def test_cluster_execution_type_lifecycle_with_port_forward() -> None:
    cluster = ClusterExecutionType(
        clusterUrl="http://localhost:8265",
        portForward=PortForwardConfiguration(
            namespace="prod-ns", serviceName="ray-head", localPort=9265
        ),
    )
    reloaded = ClusterExecutionType.model_validate(cluster.model_dump())
    assert reloaded == cluster
    assert reloaded.portForward.localPort == 9265


def test_cluster_execution_type_rejects_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        ClusterExecutionType(clusterUrl="http://localhost:8265", unknown="x")


def test_cluster_execution_type_invalid_url() -> None:
    """clusterUrl must be a valid URL with scheme and host."""
    with pytest.raises(pydantic.ValidationError, match="valid URL"):
        ClusterExecutionType(clusterUrl="not-a-url")


def test_cluster_execution_type_url_without_host() -> None:
    with pytest.raises(pydantic.ValidationError):
        ClusterExecutionType(clusterUrl="http://")


# ---------------------------------------------------------------------------
# JobExecutionType
# ---------------------------------------------------------------------------


def test_job_execution_type_defaults() -> None:
    """type defaults to 'job'."""
    job = JobExecutionType()
    assert job.type == "job"


def test_job_execution_type_lifecycle() -> None:
    job = JobExecutionType()
    reloaded = JobExecutionType.model_validate(job.model_dump())
    assert reloaded == job


def test_job_execution_type_rejects_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        JobExecutionType(unexpected="x")


# ---------------------------------------------------------------------------
# PackageConfiguration
# ---------------------------------------------------------------------------


def test_package_configuration_defaults() -> None:
    """Both lists default to empty."""
    pkg = PackageConfiguration()
    assert pkg.fromPyPI == []
    assert pkg.fromSource == []


def test_package_configuration_with_values() -> None:
    pkg = PackageConfiguration(
        fromPyPI=["ado-core", "ado-ray-tune"],
        fromSource=["plugins/actuators/vllm_performance"],
    )
    assert len(pkg.fromPyPI) == 2
    assert len(pkg.fromSource) == 1


def test_package_configuration_lifecycle() -> None:
    pkg = PackageConfiguration(
        fromPyPI=["ado-core"],
        fromSource=["plugins/operators/ray_tune"],
    )
    reloaded = PackageConfiguration.model_validate(pkg.model_dump())
    assert reloaded == pkg


# ---------------------------------------------------------------------------
# RemoteExecutionContext discriminated union routing
# ---------------------------------------------------------------------------


def test_remote_execution_context_cluster_type() -> None:
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265")
    )
    assert isinstance(ctx.executionType, ClusterExecutionType)
    assert ctx.wait is True
    assert ctx.envVars == {}
    assert ctx.packages.fromPyPI == []


def test_remote_execution_context_job_type() -> None:
    ctx = RemoteExecutionContext(executionType=JobExecutionType())
    assert isinstance(ctx.executionType, JobExecutionType)


def test_remote_execution_context_discriminated_via_dict_cluster() -> None:
    """Verify dict-based construction routes to ClusterExecutionType via discriminator."""
    data = {
        "executionType": {
            "type": "cluster",
            "clusterUrl": "http://localhost:8265",
        }
    }
    ctx = RemoteExecutionContext.model_validate(data)
    assert isinstance(ctx.executionType, ClusterExecutionType)


def test_remote_execution_context_discriminated_via_dict_job() -> None:
    data = {"executionType": {"type": "job"}}
    ctx = RemoteExecutionContext.model_validate(data)
    assert isinstance(ctx.executionType, JobExecutionType)


def test_remote_execution_context_invalid_type_discriminator() -> None:
    with pytest.raises(pydantic.ValidationError):
        RemoteExecutionContext.model_validate(
            {"executionType": {"type": "unknown_type"}}
        )


# ---------------------------------------------------------------------------
# RemoteExecutionContext full lifecycle (YAML round-trip)
# ---------------------------------------------------------------------------


def test_remote_execution_context_lifecycle_cluster_no_port_forward() -> None:
    """Create → JSON dump → reload produces identical object."""
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://ray.example.com:8265"),
        packages=PackageConfiguration(
            fromPyPI=["ado-core", "ado-ray-tune"],
            fromSource=["plugins/actuators/vllm_performance"],
        ),
        wait=False,
        envVars={"PYTHONUNBUFFERED": "x", "OMP_NUM_THREADS": "1"},
    )
    dumped = ctx.model_dump()
    reloaded = RemoteExecutionContext.model_validate(dumped)
    assert reloaded == ctx
    assert reloaded.wait is False
    assert reloaded.envVars["OMP_NUM_THREADS"] == "1"


def test_remote_execution_context_lifecycle_cluster_with_port_forward() -> None:
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(
            clusterUrl="http://localhost:8265",
            portForward=PortForwardConfiguration(
                namespace="prod-ns",
                serviceName="my-ray-cluster-head-svc",
                localPort=8265,
            ),
        ),
        packages=PackageConfiguration(fromPyPI=["ado-core"]),
    )
    dumped = ctx.model_dump()
    reloaded = RemoteExecutionContext.model_validate(dumped)
    assert reloaded == ctx
    assert reloaded.executionType.portForward.namespace == "prod-ns"


def test_remote_execution_context_yaml_round_trip() -> None:
    """Verify the model survives a YAML serialization round-trip (as users would use it)."""
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(
            clusterUrl="http://localhost:8265",
            portForward=PortForwardConfiguration(
                namespace="my-ns", serviceName="ray-svc"
            ),
        ),
        packages=PackageConfiguration(
            fromPyPI=["ado-core"],
            fromSource=["plugins/actuators/vllm_performance"],
        ),
        wait=True,
        envVars={"PYTHONUNBUFFERED": "x"},
    )
    yaml_str = pydantic_model_as_yaml(ctx)
    reloaded = RemoteExecutionContext.model_validate(yaml.safe_load(yaml_str))
    assert reloaded == ctx


def test_remote_execution_context_rejects_extra_fields() -> None:
    with pytest.raises(pydantic.ValidationError):
        RemoteExecutionContext.model_validate(
            {
                "executionType": {
                    "type": "cluster",
                    "clusterUrl": "http://localhost:8265",
                },
                "unexpectedField": "value",
            }
        )
