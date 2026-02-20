# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

from typing import Annotated, Literal

import pydantic


class PortForwardConfiguration(pydantic.BaseModel):
    """Configuration for setting up a port-forward to a Ray cluster on OpenShift/Kubernetes.

    When present in a ClusterExecutionType, ado will start the port-forward
    automatically before submitting the Ray job.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    namespace: Annotated[
        str,
        pydantic.Field(
            description="The OpenShift/Kubernetes namespace of the Ray cluster"
        ),
    ]
    serviceName: Annotated[
        str,
        pydantic.Field(description="The name of the Ray cluster service to forward to"),
    ]
    localPort: Annotated[
        int,
        pydantic.Field(
            description="The local port to bind for the port-forward",
            gt=0,
            le=65535,
        ),
    ] = 8265


class ClusterExecutionType(pydantic.BaseModel):
    """Execution type for submitting jobs to an existing Ray cluster.

    The clusterUrl is always required. If portForward is provided, ado will
    automatically start a port-forward to make the cluster reachable at that URL
    before submitting.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    type: Annotated[
        Literal["cluster"],
        pydantic.Field(description="Discriminator for the cluster execution type"),
    ] = "cluster"

    clusterUrl: Annotated[
        str,
        pydantic.Field(
            description=(
                "URL of the Ray cluster dashboard. This is either an open route URL "
                "or an in-cluster URL. When portForward is provided, this must be "
                "reachable via the forwarded local port (e.g. http://localhost:8265)."
            )
        ),
    ]

    portForward: Annotated[
        PortForwardConfiguration | None,
        pydantic.Field(
            description=(
                "If provided, ado will start a port-forward to the cluster before "
                "submitting the Ray job and tear it down afterwards. "
                "Required when the cluster is only reachable via port-forward "
                "(e.g. on OpenShift without an open route)."
            )
        ),
    ] = None

    @pydantic.field_validator("clusterUrl", mode="before")
    @classmethod
    def validate_cluster_url(cls, value: object) -> object:
        """Validate that clusterUrl is a well-formed URL with a scheme and host.

        Raises
        ------
        ValueError
            If the value is not a string or does not have a recognisable URL
            scheme and host component.
        """
        if not isinstance(value, str):
            return value
        try:
            parsed = pydantic.AnyUrl(value)
        except Exception as exc:
            raise ValueError(
                f"clusterUrl '{value}' is not a valid URL. "
                "Expected a full URL including scheme, e.g. http://localhost:8265"
            ) from exc
        if not parsed.host:
            raise ValueError(
                f"clusterUrl '{value}' must include a host, "
                "e.g. http://localhost:8265"
            )
        return value


class JobExecutionType(pydantic.BaseModel):
    """Execution type for submitting a KubeRay job (planned, not yet implemented)."""

    model_config = pydantic.ConfigDict(extra="forbid")

    type: Annotated[
        Literal["job"],
        pydantic.Field(description="Discriminator for the KubeRay job execution type"),
    ] = "job"


ExecutionTypeUnion = Annotated[
    Annotated[ClusterExecutionType, pydantic.Tag("cluster")]
    | Annotated[JobExecutionType, pydantic.Tag("job")],
    pydantic.Field(discriminator="type"),
]


class PackageConfiguration(pydantic.BaseModel):
    """Configuration for Python packages to install in the Ray job environment."""

    model_config = pydantic.ConfigDict(extra="forbid")

    fromPyPI: Annotated[
        list[str],
        pydantic.Field(
            description="PyPI package names (or version-pinned specs) to install in the Ray job",
            default_factory=list,
        ),
    ]

    fromSource: Annotated[
        list[str],
        pydantic.Field(
            description=(
                "Paths to in-tree plugin directories to build as wheels and send with the job. "
                "Paths are relative to the ado repository root."
            ),
            default_factory=list,
        ),
    ]


class ExecutionContext(pydantic.BaseModel):
    """Configuration for executing ado commands on a remote Ray cluster.

    Captures all information required to dispatch an ado command to a remote
    cluster via ``ray job submit``.

    Example usage::

        ado --execution-context exc_context.yaml create operation -f operation.yaml

    The project context must use a non-SQLite (remote) metastore when an
    execution context is provided.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    executionType: Annotated[
        ExecutionTypeUnion,
        pydantic.Field(description="How the remote execution should be performed"),
    ]

    packages: Annotated[
        PackageConfiguration,
        pydantic.Field(
            description="Python packages to install in the Ray job runtime environment",
            default_factory=PackageConfiguration,
        ),
    ]

    wait: Annotated[
        bool,
        pydantic.Field(
            description=(
                "Whether to remain attached to the Ray job until it completes. "
                "If False, the job is submitted with --no-wait and ado exits immediately."
            )
        ),
    ] = True

    envVars: Annotated[
        dict[str, str],
        pydantic.Field(
            description="Environment variables to set in the Ray job runtime environment",
            default_factory=dict,
        ),
    ]
