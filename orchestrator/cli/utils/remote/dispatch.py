# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import contextlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from urllib.parse import urlparse

import yaml
from rich.status import Status

from orchestrator.cli.models.remote_submission import (
    SUBMISSION_FILE_COPY_FLAGS,
    SUBMISSION_STRIP_FLAGS,
    RemoteSubmissionFlagMatch,
    RemoteSubmissionFlagSpec,
)
from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_REMOTE_PORT_FORWARD,
    ADO_SPINNER_REMOTE_PREPARING_FILES,
)
from orchestrator.cli.utils.remote.arg_parser import (
    rewrite_flag_values,
    strip_flags,
)
from orchestrator.core.remotecontext.config import (
    ClusterExecutionType,
    JobExecutionType,
    PortForwardConfiguration,
    RemoteExecutionContext,
)
from orchestrator.metastore.project import ProjectContext

log = logging.getLogger(__name__)

# Port-forward tool preference order
_PORT_FORWARD_TOOLS = ["oc", "kubectl"]

# Substring written to stdout by oc/kubectl when the tunnel is bound and ready
_PORT_FORWARD_READY_PATTERN = b"Forwarding from"

# How long to wait for the port-forward tunnel to become ready
_PORT_FORWARD_READY_TIMEOUT_S = 30.0
_PORT_FORWARD_READY_POLL_S = 0.1


def _find_port_forward_tool() -> str:
    """Return the first available port-forward CLI tool (``oc`` or ``kubectl``).

    Raises:
        RuntimeError: If neither ``oc`` nor ``kubectl`` is available on PATH.
    """
    for tool in _PORT_FORWARD_TOOLS:
        if shutil.which(tool):
            return tool
    raise RuntimeError(
        f"Neither {' nor '.join(_PORT_FORWARD_TOOLS)} was found on PATH. "
        "Install one of them to use port-forward with --remote."
    )


def _decode_joined_bytes(chunks: list[bytes]) -> str:
    """Decode collected byte chunks for display in error messages."""
    return b"".join(chunks).decode(errors="replace")


def _start_stream_drain_thread(
    stream: object,
    *,
    on_line: Callable[[bytes], None] | None = None,
    sink: list[bytes] | None = None,
) -> threading.Thread:
    """Start a daemon thread that drains a subprocess stream."""

    def _drain() -> None:
        if stream is None:  # pragma: no cover
            return
        for line in stream:
            if sink is not None:
                sink.append(line)
            if on_line is not None:
                on_line(line)

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    return thread


def _wait_for_port_forward_ready(
    proc: subprocess.Popen[bytes],
    ready: threading.Event,
) -> int | None:
    """Wait for ready signal while failing fast if the process exits."""
    deadline = time.monotonic() + _PORT_FORWARD_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if ready.wait(timeout=_PORT_FORWARD_READY_POLL_S):
            return None
        if (exit_code := proc.poll()) is not None:
            return exit_code
    return proc.poll()


@contextlib.contextmanager
def _port_forward_context(
    pf_config: PortForwardConfiguration,
    cluster_url: str,
) -> Generator[None, None, None]:
    """Context manager that starts a port-forward subprocess and tears it down on exit.

    Waits until ``oc``/``kubectl`` writes ``"Forwarding from"`` to its stdout,
    which is the authoritative signal that the tunnel is bound and ready to
    accept connections.  This avoids the race condition of a fixed-duration
    sleep where the tunnel may not yet be bound when the caller proceeds.

    Args:
        pf_config: Port-forward configuration from the remote execution context.
        cluster_url: The Ray cluster URL (used to extract the target service port).
    """
    parsed = urlparse(cluster_url)
    service_port = parsed.port or 8265

    tool = _find_port_forward_tool()
    cmd = [
        tool,
        "port-forward",
        "--namespace",
        pf_config.namespace,
        f"svc/{pf_config.serviceName}",
        f"{pf_config.localPort}:{service_port}",
    ]
    log.info("Starting port-forward: %s", shlex.join(cmd))
    proc = subprocess.Popen(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # A background thread drains stdout and signals when the tunnel is bound.
    # Draining is necessary even after the ready signal to prevent the pipe
    # buffer from filling and blocking the oc/kubectl process.
    ready = threading.Event()
    stderr_chunks: list[bytes] = []

    stdout_thread = _start_stream_drain_thread(
        proc.stdout,
        on_line=lambda line: (
            ready.set() if _PORT_FORWARD_READY_PATTERN in line else None
        ),
    )
    stderr_thread = _start_stream_drain_thread(proc.stderr, sink=stderr_chunks)

    if (exit_code := _wait_for_port_forward_ready(proc, ready)) is not None:
        stderr_thread.join(timeout=1)
        raise RuntimeError(
            "Port-forward process exited before becoming ready.\n"
            f"Command: {shlex.join(cmd)}\n"
            f"exit_code: {exit_code}\n"
            f"stderr: {_decode_joined_bytes(stderr_chunks)}"
        )

    if not ready.is_set():
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stderr_thread.join(timeout=1)
        raise RuntimeError(
            f"Port-forward did not become ready within {_PORT_FORWARD_READY_TIMEOUT_S}s.\n"
            f"Command: {shlex.join(cmd)}\n"
            f"stderr: {_decode_joined_bytes(stderr_chunks)}"
        )

    log.debug("Port-forward ready")
    try:
        yield
    finally:
        log.info("Tearing down port-forward (pid=%d)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stdout_thread.join(timeout=1)
        if stdout_thread.is_alive():
            log.warning("stdout thread did not terminate within timeout")
        stderr_thread.join(timeout=1)
        if stderr_thread.is_alive():
            log.warning("stderr thread did not terminate within timeout")


def _copy_files_and_rewrite_args(
    args: list[str],
    working_dir: Path,
) -> list[str]:
    """Copy files referenced by flags into *working_dir* and rewrite paths.

    Returns a new argument list with file paths replaced by basenames.

    Args:
        args: Argument list potentially containing file flags.
        working_dir: Destination directory for copied files.

    Returns:
        A new argument list with file paths replaced by basenames.

    Raises:
        ValueError: If two file arguments share the same basename.
        FileNotFoundError: If a ``--with KEY=VALUE`` value looks like a path but doesn't exist.
    """
    copied_basenames: set[str] = set()

    def rewrite_file_value(
        flag_match: RemoteSubmissionFlagMatch,
        flag_spec: RemoteSubmissionFlagSpec,
    ) -> str:
        """Rewrite a file flag value by copying file and returning basename."""
        # This should never be None for flags with hasValue=True, but handle it
        if flag_match.value is None:
            raise ValueError(f"Flag {flag_match.name} has no value")

        # Special handling for --with KEY=VALUE
        if flag_spec.valueType == "key_value":
            return _rewrite_with_value(flag_match.value, working_dir, copied_basenames)

        # Regular file path handling
        src = Path(flag_match.value).resolve()
        _copy_file_checked(src, working_dir, copied_basenames)
        return src.name

    return rewrite_flag_values(args, SUBMISSION_FILE_COPY_FLAGS, rewrite_file_value)


def _rewrite_with_value(kv: str, working_dir: Path, seen: set[str]) -> str:
    """Rewrite the value part of a ``--with KEY=VALUE`` argument if it is a file.

    Mirrors the path-detection logic in ``ado create``: a value is treated as
    a file path when it contains a ``.`` or path separator.  If it resolves to
    an existing file it is copied to *working_dir* and the argument is rewritten
    to ``KEY=basename``.  If it looks like a path but the file is not found, a
    ``FileNotFoundError`` is raised so the problem is caught locally rather than
    on the remote cluster.  Plain resource identifiers are returned unchanged.

    Args:
        kv: The ``KEY=VALUE`` string from the ``--with`` argument.
        working_dir: Destination directory for copied files.
        seen: Set of basenames already copied; updated in-place on copy.

    Returns:
        The (possibly rewritten) ``KEY=VALUE`` string.

    Raises:
        FileNotFoundError: If the value looks like a path but does not exist.
        ValueError: If the file's basename collides with another already-copied file.
    """
    if "=" not in kv:
        # Malformed --with value — pass through unchanged and let ado validate it
        return kv

    key, value = kv.split("=", 1)
    value_looks_like_path = "." in value or os.sep in value
    value_as_path = Path(value)

    if value_as_path.exists() and value_as_path.is_file():
        src = value_as_path.resolve()
        _copy_file_checked(src, working_dir, seen)
        return f"{key}={src.name}"

    if value_looks_like_path:
        raise FileNotFoundError(
            f"--with {kv}: '{value}' looks like a file path but does not exist."
        )

    # Plain resource identifier — leave unchanged
    return kv


def _copy_file_checked(src: Path, working_dir: Path, seen: set[str]) -> None:
    """Copy *src* into *working_dir*, raising ``ValueError`` on basename collision.

    Args:
        src: Resolved source file path.
        working_dir: Destination directory.
        seen: Set of basenames already copied; updated in-place.

    Raises:
        ValueError: If *src*'s basename already exists in *seen* or is empty.
    """
    if not src.name:
        raise ValueError(
            f"Invalid file path: '{src}' resolves to a path with an empty basename. "
            "Please provide a valid file path."
        )
    if src.name in seen:
        raise ValueError(
            f"File basename collision: two file arguments resolve to the "
            f"same filename '{src.name}'. Rename one of the files before "
            "submitting remotely."
        )
    shutil.copy2(src, working_dir / src.name)
    seen.add(src.name)


def _symlink_additional_files(
    files: list[str],
    cwd: Path,
    working_dir: Path,
    seen: set[str],
) -> None:
    """Create symlinks in *working_dir* for each path listed in *files*.

    Both regular files and directories are supported.  Relative paths are
    resolved with respect to *cwd* (the directory where ``ado --remote`` was
    invoked).  Symbolic links are used rather than copies to avoid duplicating
    potentially large files or directory trees.

    Args:
        files: Paths from the ``additionalFiles`` config field.
        cwd: Directory used to resolve relative paths.
        working_dir: Ray working directory in which symlinks are created.
        seen: Set of basenames already present in *working_dir*; updated in-place.

    Raises:
        FileNotFoundError: If a listed path does not exist.
        ValueError: If a path's basename collides with an entry already present
            in *working_dir*.
    """
    for file_str in files:
        path = Path(file_str)
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"additionalFiles entry '{file_str}' does not exist: {path}"
            )
        if path.name in seen:
            raise ValueError(
                f"File basename collision: additionalFiles entry '{file_str}' "
                f"has basename '{path.name}' which conflicts with another entry "
                "already included in the Ray working directory. "
                "Rename the file or directory to avoid the conflict."
            )

        (working_dir / path.name).symlink_to(path)
        seen.add(path.name)
        log.debug("Symlinked additional path %s -> %s", working_dir / path.name, path)


def _build_source_wheels(
    from_source: list[str],
    working_dir: Path,
    repo_root: Path,
) -> list[str]:
    """Build wheels for in-tree plugins and copy them into *working_dir*.

    Each entry in *from_source* is a path relative to *repo_root* (or absolute).
    Runs ``uv build --wheel -o dist --clear`` in each plugin directory.

    Args:
        from_source: Plugin directory paths to build.
        working_dir: Destination for built ``.whl`` files.
        repo_root: Base directory used to resolve relative paths in *from_source*.

    Returns:
        Basenames of the wheel files copied to *working_dir*.
    """
    wheel_names: list[str] = []
    for plugin_path_str in from_source:
        plugin_path = Path(plugin_path_str)
        if not plugin_path.is_absolute():
            plugin_path = (repo_root / plugin_path).resolve()

        if not plugin_path.is_dir():
            raise FileNotFoundError(
                f"fromSource plugin path does not exist: {plugin_path}"
            )

        dist_dir = plugin_path / "dist"
        log.info("Building wheel for plugin at %s", plugin_path)
        result = subprocess.run(  # noqa: S603
            ["uv", "build", "--wheel", "-o", str(dist_dir), "--clear"],  # noqa: S607
            cwd=str(plugin_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"uv build failed for {plugin_path}:\n{result.stderr}")

        wheels = list(dist_dir.glob("*.whl"))
        if not wheels:
            raise RuntimeError(f"No wheel produced by uv build in {dist_dir}")

        for whl in wheels:
            dest = working_dir / whl.name
            shutil.copy2(whl, dest)
            wheel_names.append(whl.name)
            log.info("Copied wheel %s to %s", whl.name, working_dir)

    return wheel_names


def _write_runtime_env(
    remote_context: RemoteExecutionContext,
    wheel_names: list[str],
    dest: Path,
) -> None:
    """Write the Ray runtime environment YAML to *dest*.

    Combines PyPI packages, wheel references, and environment variables from
    *remote_context* into a ``runtime_env.yaml`` compatible with
    ``ray job submit --runtime-env``.

    Args:
        remote_context: The remote execution context describing packages and env vars.
        wheel_names: Basenames of wheel files present in the Ray working dir.
        dest: Path to write the generated ``runtime_env.yaml``.
    """
    uv_packages: list[str] = list(remote_context.packages.fromPyPI)
    uv_packages.extend(
        f"${{RAY_RUNTIME_ENV_CREATE_WORKING_DIR}}/{wheel_name}"
        for wheel_name in wheel_names
    )

    runtime_env: dict[str, list[str] | dict[str, str]] = {}
    if uv_packages:
        runtime_env["uv"] = uv_packages

    if remote_context.envVars:
        runtime_env["env_vars"] = dict(remote_context.envVars)

    dest.write_text(yaml.dump(runtime_env, default_flow_style=False))
    log.debug("Wrote runtime_env.yaml to %s", dest)


def _run_ray_submit(
    cluster_exec: ClusterExecutionType,
    remote_context: RemoteExecutionContext,
    working_dir: Path,
    runtime_env_path: Path,
    remote_ado_args: list[str],
) -> int:
    """Construct and run the ``ray job submit`` command.

    Args:
        cluster_exec: Cluster execution type (contains ``clusterUrl``).
        remote_context: Full remote execution context (used for ``wait`` flag).
        working_dir: Working directory to send with the job.
        runtime_env_path: Path to the generated ``runtime_env.yaml``.
        remote_ado_args: The rewritten ado argument list for use inside the Ray job.

    Returns:
        The exit code of ``ray job submit``.
    """
    cmd = ["ray", "job", "submit"]

    if not remote_context.wait:
        cmd.append("--no-wait")

    cmd += [
        "--address",
        cluster_exec.clusterUrl.unicode_string(),
        "--working-dir",
        str(working_dir),
        "--runtime-env",
        str(runtime_env_path),
        "-v",
        "--",
        "ado",
        *remote_ado_args,
    ]

    log.info("Running: %s", shlex.join(cmd))
    result = subprocess.run(cmd)  # noqa: S603
    if result.returncode != 0:
        log.error(
            "ray job submit exited with code %d. "
            "Check the output above for details.",
            result.returncode,
        )
    return result.returncode


def _dispatch_to_cluster(
    cluster_exec: ClusterExecutionType,
    remote_context: RemoteExecutionContext,
    project_context: ProjectContext,
    ado_args: list[str],
    working_dir: Path,
    repo_root: Path,
    cwd: Path,
) -> int:
    """Build the working directory and run ``ray job submit``.

    Args:
        cluster_exec: Resolved cluster execution type.
        remote_context: Full remote execution context.
        project_context: The ProjectContext instance to serialize into the working directory.
        ado_args: Full ado argument list (without ``--remote``).
        working_dir: Temporary directory to use as the Ray working directory.
        repo_root: Repository root for resolving relative ``fromSource`` paths.
        cwd: Directory used to resolve relative paths in ``additionalFiles``.

    Returns:
        Exit code of ``ray job submit``.
    """
    pf = cluster_exec.portForward
    pf_ctx = (
        _port_forward_context(pf, cluster_exec.clusterUrl.unicode_string())
        if pf is not None
        else contextlib.nullcontext()
    )

    # Use an ExitStack so that:
    #  - the Status spinner can be stopped before ray job submit produces output,
    #  - while the port-forward context manager remains active for the duration
    #    of the ray job submit call.
    with contextlib.ExitStack() as stack:
        status = stack.enter_context(Status(ADO_SPINNER_REMOTE_PREPARING_FILES))

        # 1. Serialize project context to working directory
        context_filename = f"{project_context.project}.yaml"
        context_file_path = working_dir / context_filename
        context_file_path.write_text(
            yaml.dump(project_context.model_dump(), default_flow_style=False)
        )

        # 2. Copy any -f / --with files into the working directory and rewrite paths
        rewritten_args = _copy_files_and_rewrite_args(ado_args, working_dir)
        remote_ado_args = ["-c", context_filename, *rewritten_args]

        # 3. Symlink any additionalFiles into the working directory.
        #    Collect basenames already present to detect collisions.
        seen_basenames: set[str] = {f.name for f in working_dir.iterdir()}
        _symlink_additional_files(
            remote_context.additionalFiles,
            cwd,
            working_dir,
            seen_basenames,
        )

        # 4. Build wheels for fromSource plugins
        wheel_names = _build_source_wheels(
            remote_context.packages.fromSource,
            working_dir,
            repo_root,
        )

        # 5. Generate runtime_env.yaml
        runtime_env_path = working_dir / "runtime_env.yaml"
        _write_runtime_env(remote_context, wheel_names, runtime_env_path)

        # 6. Establish port-forward (blocks until tunnel is bound and ready)
        if pf is not None:
            status.update(ADO_SPINNER_REMOTE_PORT_FORWARD)
        stack.enter_context(pf_ctx)

        # Stop the spinner before ray job submit starts producing its own output
        status.stop()

        return _run_ray_submit(
            cluster_exec=cluster_exec,
            remote_context=remote_context,
            working_dir=working_dir,
            runtime_env_path=runtime_env_path,
            remote_ado_args=remote_ado_args,
        )


def dispatch(
    remote_context: RemoteExecutionContext,
    project_context: ProjectContext,
    argv: list[str],
    repo_root: Path | None = None,
    cwd: Path | None = None,
) -> int:
    """Dispatch an ado command to a remote Ray cluster via ``ray job submit``.

    Handles building of temporary working directories, wheel compilation for
    in-tree plugins, ``runtime_env.yaml`` generation, optional port-forwarding,
    and the ``ray job submit`` invocation itself.

    Args:
        remote_context: Loaded and validated remote execution context.
        project_context: The ProjectContext instance to serialize and send to the remote cluster.
            This will be written to a file in the working directory and referenced
            via ``-c`` in the remote ado command.
        argv: The full ado argument list (sys.argv[1:]). This function will strip
            ``--remote``, ``--override-ado-app-dir``, and other
            submission-specific flags before processing.
        repo_root: Root of the ado repository, used to resolve relative ``fromSource``
            paths.  Defaults to ``Path.cwd()``.
        cwd: Directory used to resolve relative paths in ``additionalFiles``.
            Defaults to ``Path.cwd()``.

    Returns:
        The exit code of the ``ray job submit`` subprocess.

    Raises:
        NotImplementedError: If ``executionType`` is ``JobExecutionType`` (KubeRay - not yet
            implemented).
        RuntimeError: On port-forward startup failures or wheel build errors.
        ValueError: If two file arguments share the same basename.
        FileNotFoundError: If a ``--with`` value looks like a path but does not exist,
            or if an ``additionalFiles`` entry does not exist.
    """
    if isinstance(remote_context.executionType, JobExecutionType):
        raise NotImplementedError(
            "KubeRay job execution (executionType: job) is not yet implemented."
        )

    cluster_exec: ClusterExecutionType = remote_context.executionType
    resolved_repo_root = repo_root or Path.cwd()
    resolved_cwd = cwd or Path.cwd()

    # Reconstruct the ado argument list without submission-specific flags
    ado_args = strip_flags(argv, SUBMISSION_STRIP_FLAGS)

    with tempfile.TemporaryDirectory(prefix="ado-remote-") as tmp_dir:
        return _dispatch_to_cluster(
            cluster_exec=cluster_exec,
            remote_context=remote_context,
            project_context=project_context,
            ado_args=ado_args,
            working_dir=Path(tmp_dir),
            repo_root=resolved_repo_root,
            cwd=resolved_cwd,
        )
