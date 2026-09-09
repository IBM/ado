# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import warnings
from pathlib import PurePath
from typing import Annotated, Literal

import pydantic
from typing_extensions import Self

from ado.utilities.pydantic import validate_rfc_1123


class PortForwardConfiguration(pydantic.BaseModel):
    """Configuration for setting up a port-forward to a Ray cluster on OpenShift/Kubernetes.

    When present in a ClusterExecutionType, ado will start the port-forward
    automatically before submitting the Ray job.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    namespace: Annotated[
        str,
        pydantic.AfterValidator(validate_rfc_1123),
        pydantic.Field(
            description="The OpenShift/Kubernetes namespace of the Ray cluster"
        ),
    ]
    serviceName: Annotated[
        str,
        pydantic.AfterValidator(validate_rfc_1123),
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
        pydantic.HttpUrl,
        pydantic.UrlConstraints(host_required=True, default_port=8265),
        pydantic.Field(
            description=(
                "URL of the Ray cluster dashboard (host required, default port 8265). "
                "This is either an open route URL or an in-cluster URL. "
                "When portForward is provided, this must be "
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


# Ray RuntimeEnvironmentConfiguration defaults
# Used in next class
RAY_DEFAULT_SETUP_TIMEOUT_SECONDS = 600
RAY_DEFAULT_EAGER_INSTALL = True


class RuntimeEnvironmentConfiguration(pydantic.BaseModel):
    """Ray ``runtime_env.config`` options for remote job submission.

    Maps to Ray's ``RuntimeEnvConfig`` (see Ray handling-dependencies docs).
    Field defaults match Ray's defaults and are always written when ``runtimeEnv``
    is present on the execution context.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    setupTimeoutSeconds: Annotated[
        int,
        pydantic.Field(
            description=(
                "Maximum seconds to create the job runtime environment on a worker. "
                "Use -1 to disable the timeout."
            ),
        ),
    ] = RAY_DEFAULT_SETUP_TIMEOUT_SECONDS

    eagerInstall: Annotated[
        bool,
        pydantic.Field(
            description=(
                "If true, install the job runtime environment on nodes when the job "
                "starts. If false, install lazily when the first task runs."
            ),
        ),
    ] = RAY_DEFAULT_EAGER_INSTALL

    @pydantic.field_validator("setupTimeoutSeconds")
    @classmethod
    def validate_setup_timeout_seconds(cls, value: int) -> int:
        """Validate setup timeout matches Ray rules."""
        if value == -1:
            return value
        if value <= 0:
            raise ValueError(
                "setupTimeoutSeconds must be greater than zero or -1 to disable timeout"
            )
        return value


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


class RemoteExecutionContext(pydantic.BaseModel):
    """Configuration for executing ado commands on a remote Ray cluster.

    Captures all information required to dispatch an ado command to a remote
    cluster via ``ray job submit``.

    Example usage::

        ado --remote remote_context.yaml create operation -f operation.yaml

    The project context must use a non-SQLite (remote) metastore when a
    remote execution context is provided.
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

    runtimeEnv: Annotated[
        RuntimeEnvironmentConfiguration | None,
        pydantic.Field(
            description=(
                "Optional Ray runtime environment configuration (setup timeout, "
                "eager install). Written to the ``config`` section of runtime_env.yaml."
            ),
        ),
    ] = None

    additionalFiles: Annotated[
        list[str],
        pydantic.Field(
            description=(
                "Additional files or directories to send with the Ray job. "
                "Paths may be absolute or relative to the directory where "
                "``ado --remote`` is executed. "
                "Symbolic links are created in the Ray working directory to "
                "avoid unnecessary copies."
            ),
            default_factory=list,
        ),
    ]

    @pydantic.model_validator(mode="after")
    def validate_and_normalize_wheels(self) -> Self:
        """Validate wheel entries in packages.fromPyPI and additionalFiles.

        Ensures that:
        1. Entries in additionalFiles do not have duplicate basenames.
        2. Wheel entries in packages.fromPyPI do not use relative subpaths.
        3. Wheel entries prefixed with ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/ emit a
           UserWarning and are rewritten to the bare wheel filename.
        4. Local wheel entries not in additionalFiles emit a warning.
        """
        prefix = "${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/"

        # Check duplicate basenames in additionalFiles
        seen_additional_basenames: set[str] = set()
        for add_file in self.additionalFiles:
            name = PurePath(add_file).name
            if name in seen_additional_basenames:
                raise ValueError(
                    f"Conflicting duplicate basename in additionalFiles: '{name}' "
                    f"from entry '{add_file}'."
                )
            seen_additional_basenames.add(name)

        normalized_from_pypi: list[str] = []
        for entry in self.packages.fromPyPI:
            if entry.lower().endswith(".whl"):
                # If prefixed with ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/, warn and strip
                if entry.startswith(prefix):
                    warnings.warn(
                        f"Prefix '{prefix}' in packages.fromPyPI for '{entry}' is "
                        "unnecessary and will be automatically managed by ado. "
                        "Removing prefix. For wheels from other local directories, "
                        "include them in additionalFiles.",
                        UserWarning,
                        stacklevel=2,
                    )
                    wheel_name = entry[len(prefix) :]
                else:
                    wheel_name = entry

                pure_path = PurePath(wheel_name)
                # Check if it is an absolute path
                if pure_path.is_absolute():
                    normalized_from_pypi.append(wheel_name)
                    continue

                # If relative path has directories, warn to use bare wheel name + additionalFiles
                if pure_path.parent != PurePath("."):
                    warnings.warn(
                        f"Wheel '{entry}' in packages.fromPyPI contains a relative "
                        "directory path. Consider specifying a bare wheel name "
                        f"'{pure_path.name}' and listing '{entry}' in additionalFiles instead.",
                        UserWarning,
                        stacklevel=2,
                    )
                elif wheel_name not in seen_additional_basenames:
                    # Bare wheel: warn if not present in additionalFiles
                    warnings.warn(
                        f"Wheel '{wheel_name}' in packages.fromPyPI is not listed in "
                        "additionalFiles. If it is located in the working directory, it will "
                        "be used; otherwise, include it in additionalFiles.",
                        UserWarning,
                        stacklevel=2,
                    )

                normalized_from_pypi.append(wheel_name)
            else:
                normalized_from_pypi.append(entry)

        self.packages.fromPyPI = normalized_from_pypi
        return self
