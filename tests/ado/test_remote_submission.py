# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for remote submission utilities and the --remote CLI option."""

import importlib
import pathlib
import subprocess
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from orchestrator.cli.core.cli import app as ado
from orchestrator.cli.models.remote_submission import (
    CONTEXT_FLAG,
    SUBMISSION_STRIP_FLAGS,
)
from orchestrator.cli.utils.remote import strip_flags
from orchestrator.cli.utils.remote.dispatch import (
    _copy_files_and_rewrite_args,
    _port_forward_context,
    _symlink_additional_files,
    _write_runtime_env,
    dispatch,
)
from orchestrator.core.remotecontext.config import (
    ClusterExecutionType,
    JobExecutionType,
    PackageConfiguration,
    PortForwardConfiguration,
    RemoteExecutionContext,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.output import pydantic_model_as_yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_remote_context() -> RemoteExecutionContext:
    """Minimal cluster RemoteExecutionContext without port-forward."""
    return RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        packages=PackageConfiguration(fromPyPI=["ado-core"]),
        wait=True,
        envVars={"PYTHONUNBUFFERED": "x"},
    )


@pytest.fixture
def cluster_remote_context_with_port_forward() -> RemoteExecutionContext:
    return RemoteExecutionContext(
        executionType=ClusterExecutionType(
            clusterUrl="http://localhost:8265",
            portForward=PortForwardConfiguration(
                namespace="my-ns",
                serviceName="my-ray-svc",
                localPort=8265,
            ),
        ),
    )


@pytest.fixture
def remote_context_file(
    tmp_path: pathlib.Path,
    cluster_remote_context: RemoteExecutionContext,
) -> pathlib.Path:
    """Write a cluster RemoteExecutionContext to a temp YAML file."""
    path = tmp_path / "remote_context.yaml"
    path.write_text(pydantic_model_as_yaml(cluster_remote_context))
    return path


@pytest.fixture
def mysql_context_yaml_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """A minimal MySQL context YAML file (non-SQLite)."""
    content = {
        "project": "test-project",
        "metadataStore": {
            "scheme": "mysql+pymysql",
            "host": "db.example.com",
            "port": 3306,
            "user": "admin",
            "password": "secret",
            "database": "test-project",
        },
    }
    path = tmp_path / "mysql_context.yaml"
    path.write_text(yaml.dump(content))
    return path


@pytest.fixture
def mysql_project_context(mysql_context_yaml_file: pathlib.Path) -> ProjectContext:
    """Load a ProjectContext instance from the MySQL context YAML file."""
    return ProjectContext.model_validate(
        yaml.safe_load(mysql_context_yaml_file.read_text())
    )


@pytest.fixture
def sqlite_context_yaml_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """A SQLite context YAML file."""
    project_id = "sqlite-test"
    content = {
        "project": project_id,
        "metadataStore": {
            "scheme": "sqlite",
            "database": project_id,
            "path": f"{project_id}.db",
        },
    }
    path = tmp_path / "sqlite_context.yaml"
    path.write_text(yaml.dump(content))
    return path


# ---------------------------------------------------------------------------
# strip_flags with SUBMISSION_STRIP_FLAGS (replaces remove_execution_context_from_argv)
# ---------------------------------------------------------------------------


def test_strip_remote_flags_long_form() -> None:
    argv = ["-c", "ctx.yaml", "--remote", "remote.yaml", "create", "operation"]
    result = strip_flags(argv, SUBMISSION_STRIP_FLAGS)
    assert result == ["create", "operation"]


def test_strip_remote_flags_equals_form() -> None:
    argv = ["--remote=remote.yaml", "get", "space"]
    result = strip_flags(argv, SUBMISSION_STRIP_FLAGS)
    assert result == ["get", "space"]


def test_strip_remote_flags_not_present() -> None:
    argv = ["-c", "ctx.yaml", "get", "space"]
    result = strip_flags(argv, SUBMISSION_STRIP_FLAGS)
    assert result == ["get", "space"]


def test_strip_remote_flags_strips_override_ado_app_dir() -> None:
    """--override-ado-app-dir is a local-only flag and must not be forwarded."""
    argv = [
        "--override-ado-app-dir",
        "/tmp/test",
        "--remote",
        "remote.yaml",
        "get",
        "space",
    ]
    result = strip_flags(argv, SUBMISSION_STRIP_FLAGS)
    assert result == ["get", "space"]


# ---------------------------------------------------------------------------
# strip_flags with context flags (replaces _strip_context_flag)
# ---------------------------------------------------------------------------


def test_strip_context_flags_short_form() -> None:
    args = ["-c", "ctx.yaml", "create", "operation", "-f", "op.yaml"]
    result = strip_flags(args, [CONTEXT_FLAG])
    assert result == ["create", "operation", "-f", "op.yaml"]


def test_strip_context_flags_long_form() -> None:
    args = ["--context", "ctx.yaml", "get", "space"]
    result = strip_flags(args, [CONTEXT_FLAG])
    assert result == ["get", "space"]


def test_strip_context_flags_not_present() -> None:
    args = ["create", "operation", "-f", "op.yaml"]
    result = strip_flags(args, [CONTEXT_FLAG])
    assert result == ["create", "operation", "-f", "op.yaml"]


# ---------------------------------------------------------------------------
# _copy_files_and_rewrite_args
# ---------------------------------------------------------------------------


def test_copy_files_and_rewrite_args_short_flag(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    op_file = src / "operation.yaml"
    op_file.write_text("kind: operation")

    dest = tmp_path / "working"
    dest.mkdir()

    args = ["create", "operation", "-f", str(op_file)]
    result = _copy_files_and_rewrite_args(args, dest)

    assert result == ["create", "operation", "-f", "operation.yaml"]
    assert (dest / "operation.yaml").read_text() == "kind: operation"


def test_copy_files_and_rewrite_args_long_flag(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "op.yaml"
    src.write_text("kind: operation")
    dest = tmp_path / "working"
    dest.mkdir()

    args = ["create", "operation", "--file", str(src)]
    result = _copy_files_and_rewrite_args(args, dest)

    assert result == ["create", "operation", "--file", "op.yaml"]


def test_copy_files_and_rewrite_args_equals_form(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "op.yaml"
    src.write_text("kind: operation")
    dest = tmp_path / "working"
    dest.mkdir()

    args = [f"--file={src}"]
    result = _copy_files_and_rewrite_args(args, dest)

    assert result == ["--file=op.yaml"]


def test_copy_files_and_rewrite_args_no_files(tmp_path: pathlib.Path) -> None:
    dest = tmp_path / "working"
    dest.mkdir()
    args = ["get", "space"]
    result = _copy_files_and_rewrite_args(args, dest)
    assert result == ["get", "space"]


def test_copy_files_and_rewrite_args_basename_collision(tmp_path: pathlib.Path) -> None:
    """Two -f files with the same basename should raise ValueError."""
    src_a = tmp_path / "a" / "operation.yaml"
    src_a.parent.mkdir()
    src_a.write_text("kind: a")

    src_b = tmp_path / "b" / "operation.yaml"
    src_b.parent.mkdir()
    src_b.write_text("kind: b")

    dest = tmp_path / "working"
    dest.mkdir()

    args = ["-f", str(src_a), "-f", str(src_b)]
    with pytest.raises(ValueError, match="basename collision"):
        _copy_files_and_rewrite_args(args, dest)


def test_copy_files_and_rewrite_args_with_file_path(tmp_path: pathlib.Path) -> None:
    """--with space=path/to/space.yaml copies the file and rewrites to basename."""
    space_file = tmp_path / "my_space.yaml"
    space_file.write_text("kind: space")
    dest = tmp_path / "working"
    dest.mkdir()

    args = ["create", "operation", "--with", f"space={space_file}"]
    result = _copy_files_and_rewrite_args(args, dest)

    assert result == ["create", "operation", "--with", "space=my_space.yaml"]
    assert (dest / "my_space.yaml").read_text() == "kind: space"


def test_copy_files_and_rewrite_args_with_resource_id(tmp_path: pathlib.Path) -> None:
    """--with space=some-resource-id leaves the argument unchanged."""
    dest = tmp_path / "working"
    dest.mkdir()

    args = ["create", "operation", "--with", "space=my-space-identifier"]
    result = _copy_files_and_rewrite_args(args, dest)

    assert result == ["create", "operation", "--with", "space=my-space-identifier"]


def test_copy_files_and_rewrite_args_with_equals_form(tmp_path: pathlib.Path) -> None:
    """--with=space=path/to/space.yaml (equals form) copies the file."""
    space_file = tmp_path / "space.yaml"
    space_file.write_text("kind: space")
    dest = tmp_path / "working"
    dest.mkdir()

    args = [f"--with=space={space_file}"]
    result = _copy_files_and_rewrite_args(args, dest)

    assert result == ["--with=space=space.yaml"]


def test_copy_files_and_rewrite_args_with_missing_file(tmp_path: pathlib.Path) -> None:
    """--with value that looks like a path but doesn't exist raises FileNotFoundError."""
    dest = tmp_path / "working"
    dest.mkdir()

    args = ["--with", "space=nonexistent.yaml"]
    with pytest.raises(FileNotFoundError, match=r"nonexistent\.yaml"):
        _copy_files_and_rewrite_args(args, dest)


def test_copy_files_and_rewrite_args_with_collision_across_flags(
    tmp_path: pathlib.Path,
) -> None:
    """A --with file whose basename collides with a -f file raises ValueError."""
    op_file = tmp_path / "a" / "resource.yaml"
    op_file.parent.mkdir()
    op_file.write_text("kind: operation")

    space_file = tmp_path / "b" / "resource.yaml"
    space_file.parent.mkdir()
    space_file.write_text("kind: space")

    dest = tmp_path / "working"
    dest.mkdir()

    args = ["-f", str(op_file), "--with", f"space={space_file}"]
    with pytest.raises(ValueError, match="basename collision"):
        _copy_files_and_rewrite_args(args, dest)


# ---------------------------------------------------------------------------
# _symlink_additional_files
# ---------------------------------------------------------------------------


def test_symlink_additional_files_absolute_path(tmp_path: pathlib.Path) -> None:
    """Absolute path is symlinked into the working directory."""
    src = tmp_path / "extra.py"
    src.write_text("# extra")
    working_dir = tmp_path / "working"
    working_dir.mkdir()

    _symlink_additional_files([str(src)], tmp_path, working_dir, set())

    link = working_dir / "extra.py"
    assert link.is_symlink()
    assert link.resolve() == src.resolve()
    assert link.read_text() == "# extra"


def test_symlink_additional_files_relative_path(tmp_path: pathlib.Path) -> None:
    """Relative path is resolved against cwd and symlinked."""
    src = tmp_path / "helper.py"
    src.write_text("# helper")
    working_dir = tmp_path / "working"
    working_dir.mkdir()

    _symlink_additional_files(["helper.py"], tmp_path, working_dir, set())

    link = working_dir / "helper.py"
    assert link.is_symlink()
    assert link.read_text() == "# helper"


def test_symlink_additional_files_updates_seen(tmp_path: pathlib.Path) -> None:
    """The seen set is updated with the symlinked file's basename."""
    src = tmp_path / "data.csv"
    src.write_text("a,b")
    working_dir = tmp_path / "working"
    working_dir.mkdir()

    seen: set[str] = set()
    _symlink_additional_files([str(src)], tmp_path, working_dir, seen)

    assert "data.csv" in seen


def test_symlink_additional_files_missing_file(tmp_path: pathlib.Path) -> None:
    """Non-existent path raises FileNotFoundError."""
    working_dir = tmp_path / "working"
    working_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _symlink_additional_files(["nonexistent.py"], tmp_path, working_dir, set())


def test_symlink_additional_files_directory(tmp_path: pathlib.Path) -> None:
    """A directory path is symlinked into the working directory."""
    subdir = tmp_path / "mydir"
    subdir.mkdir()
    (subdir / "script.py").write_text("# script")
    working_dir = tmp_path / "working"
    working_dir.mkdir()

    _symlink_additional_files([str(subdir)], tmp_path, working_dir, set())

    link = working_dir / "mydir"
    assert link.is_symlink()
    assert link.resolve() == subdir.resolve()
    assert (link / "script.py").read_text() == "# script"


def test_symlink_additional_files_basename_collision_with_seen(
    tmp_path: pathlib.Path,
) -> None:
    """Collision with an existing basename in seen raises ValueError."""
    src = tmp_path / "file.py"
    src.write_text("# x")
    working_dir = tmp_path / "working"
    working_dir.mkdir()

    with pytest.raises(ValueError, match="basename collision"):
        _symlink_additional_files([str(src)], tmp_path, working_dir, {"file.py"})


def test_symlink_additional_files_basename_collision_within_list(
    tmp_path: pathlib.Path,
) -> None:
    """Two additionalFiles entries sharing a basename raises ValueError."""
    src_a = tmp_path / "a" / "shared.py"
    src_a.parent.mkdir()
    src_a.write_text("# a")
    src_b = tmp_path / "b" / "shared.py"
    src_b.parent.mkdir()
    src_b.write_text("# b")
    working_dir = tmp_path / "working"
    working_dir.mkdir()

    with pytest.raises(ValueError, match="basename collision"):
        _symlink_additional_files(
            [str(src_a), str(src_b)], tmp_path, working_dir, set()
        )


def test_symlink_additional_files_empty_list(tmp_path: pathlib.Path) -> None:
    """Empty additionalFiles list is a no-op."""
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    seen: set[str] = set()
    _symlink_additional_files([], tmp_path, working_dir, seen)
    assert list(working_dir.iterdir()) == []
    assert seen == set()


# ---------------------------------------------------------------------------
# dispatch() — additionalFiles integration
# ---------------------------------------------------------------------------


def test_dispatcher_additional_files_symlinked(
    tmp_path: pathlib.Path,
    cluster_remote_context: RemoteExecutionContext,
    mysql_project_context: ProjectContext,
    mysql_context_yaml_file: pathlib.Path,
) -> None:
    """additionalFiles entries are symlinked into the Ray working directory."""
    extra_file = tmp_path / "my_function.py"
    extra_file.write_text("def fn(): pass")

    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        additionalFiles=[str(extra_file)],
    )

    found_symlink = False
    symlink_target_correct = False

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        nonlocal found_symlink, symlink_target_correct
        idx = cmd.index("--working-dir")
        working_dir = pathlib.Path(cmd[idx + 1])
        link = working_dir / "my_function.py"
        found_symlink = link.exists()
        symlink_target_correct = (
            link.is_symlink() and link.resolve() == extra_file.resolve()
        )
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        exit_code = dispatch(
            remote_context=ctx,
            project_context=mysql_project_context,
            argv=["-c", str(mysql_context_yaml_file), "get", "space"],
            cwd=tmp_path,
        )

    assert exit_code == 0
    assert found_symlink, "additional file was not found in working directory"
    assert symlink_target_correct, "additional file is not a symlink to the source"


def test_dispatcher_additional_files_relative_path(
    tmp_path: pathlib.Path,
    mysql_project_context: ProjectContext,
    mysql_context_yaml_file: pathlib.Path,
) -> None:
    """Relative additionalFiles paths are resolved relative to cwd."""
    extra_file = tmp_path / "config.yaml"
    extra_file.write_text("key: value")

    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        additionalFiles=["config.yaml"],
    )

    found = False

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        nonlocal found
        idx = cmd.index("--working-dir")
        working_dir = pathlib.Path(cmd[idx + 1])
        found = (working_dir / "config.yaml").exists()
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        dispatch(
            remote_context=ctx,
            project_context=mysql_project_context,
            argv=["get", "space"],
            cwd=tmp_path,
        )

    assert found, "relative additionalFiles path was not resolved and symlinked"


def test_dispatcher_additional_files_collision_with_copied_file(
    tmp_path: pathlib.Path,
    mysql_project_context: ProjectContext,
    mysql_context_yaml_file: pathlib.Path,
) -> None:
    """additionalFiles basename collision with a -f file raises ValueError."""
    shared_name = "operation.yaml"
    op_file = tmp_path / "a" / shared_name
    op_file.parent.mkdir()
    op_file.write_text("kind: operation")

    additional_file = tmp_path / "b" / shared_name
    additional_file.parent.mkdir()
    additional_file.write_text("kind: other")

    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        additionalFiles=[str(additional_file)],
    )

    with (
        pytest.raises(ValueError, match="basename collision"),
        patch("subprocess.run", return_value=MagicMock(returncode=0)),
    ):
        dispatch(
            remote_context=ctx,
            project_context=mysql_project_context,
            argv=["-f", str(op_file), "create", "operation"],
            cwd=tmp_path,
        )


# ---------------------------------------------------------------------------
# _write_runtime_env
# ---------------------------------------------------------------------------


def test_write_runtime_env_pypi_only(
    tmp_path: pathlib.Path,
    cluster_remote_context: RemoteExecutionContext,
) -> None:
    dest = tmp_path / "runtime_env.yaml"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    _write_runtime_env(cluster_remote_context, [], dest, tmp_path, working_dir, set())

    loaded = yaml.safe_load(dest.read_text())
    assert loaded["uv"] == ["ado-core"]
    assert loaded["env_vars"] == {"PYTHONUNBUFFERED": "x"}


def test_write_runtime_env_with_wheels(
    tmp_path: pathlib.Path,
) -> None:
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        packages=PackageConfiguration(fromPyPI=["ado-core"]),
        envVars={},
    )
    dest = tmp_path / "runtime_env.yaml"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    _write_runtime_env(
        ctx, ["my_plugin-1.0-py3-none-any.whl"], dest, tmp_path, working_dir, set()
    )

    loaded = yaml.safe_load(dest.read_text())
    assert "ado-core" in loaded["uv"]
    assert any("my_plugin" in p for p in loaded["uv"])
    assert (
        "${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/my_plugin-1.0-py3-none-any.whl"
        in loaded["uv"]
    )


def test_write_runtime_env_no_packages(tmp_path: pathlib.Path) -> None:
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
    )
    dest = tmp_path / "runtime_env.yaml"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    _write_runtime_env(ctx, [], dest, tmp_path, working_dir, set())

    loaded = yaml.safe_load(dest.read_text())
    # No uv key when there are no packages
    assert "uv" not in loaded
    assert "env_vars" not in loaded


def test_write_runtime_env_env_vars_only(tmp_path: pathlib.Path) -> None:
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        envVars={"MY_VAR": "1"},
    )
    dest = tmp_path / "runtime_env.yaml"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    _write_runtime_env(ctx, [], dest, tmp_path, working_dir, set())

    loaded = yaml.safe_load(dest.read_text())
    assert "uv" not in loaded
    assert loaded["env_vars"] == {"MY_VAR": "1"}


def test_write_runtime_env_local_wheel_in_pypi(tmp_path: pathlib.Path) -> None:
    """Local .whl paths in fromPyPI are copied to working_dir and rewritten."""
    local_whl = tmp_path / "my_local-1.0-py3-none-any.whl"
    local_whl.write_bytes(b"fake wheel")

    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        packages=PackageConfiguration(fromPyPI=[str(local_whl), "ado-core"]),
    )
    dest = tmp_path / "runtime_env.yaml"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    _write_runtime_env(ctx, [], dest, tmp_path, working_dir, set())

    assert (working_dir / local_whl.name).exists()
    loaded = yaml.safe_load(dest.read_text())
    assert "ado-core" in loaded["uv"]
    assert f"${{RAY_RUNTIME_ENV_CREATE_WORKING_DIR}}/{local_whl.name}" in loaded["uv"]
    assert str(local_whl) not in loaded["uv"]


# ---------------------------------------------------------------------------
# dispatch() — ray job submit command assembly
# ---------------------------------------------------------------------------


class _FakePortForwardProcess:
    """Minimal fake subprocess for testing port-forward lifecycle."""

    def __init__(
        self,
        stdout_data: bytes,
        stderr_data: bytes,
        poll_value: int | None = None,
    ) -> None:
        self.stdout = BytesIO(stdout_data)
        self.stderr = BytesIO(stderr_data)
        self._poll_value = poll_value
        self.terminated = False
        self.killed = False
        self.pid = 12345

    def poll(self) -> int | None:
        return self._poll_value

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


def test_port_forward_context_ready_and_teardown() -> None:
    """Port-forward should become ready and terminate on context exit."""

    # For patching
    remote_dispatch_module = importlib.import_module(
        "orchestrator.cli.utils.remote.dispatch"
    )

    fake_proc = _FakePortForwardProcess(
        stdout_data=b"Forwarding from 127.0.0.1:8265 -> 8265\n",
        stderr_data=b"",
        poll_value=None,
    )

    with (
        patch.object(
            remote_dispatch_module,
            "_find_port_forward_tool",
            return_value="kubectl",
        ),
        patch.object(
            remote_dispatch_module.subprocess, "Popen", return_value=fake_proc
        ),
        _port_forward_context(
            PortForwardConfiguration(
                namespace="my-ns",
                serviceName="my-svc",
                localPort=8265,
            ),
            "http://localhost:8265",
        ),
    ):
        pass

    assert fake_proc.terminated is True


def test_port_forward_context_fails_fast_on_early_exit() -> None:
    """Early process exit should fail before readiness timeout."""

    remote_dispatch_module = importlib.import_module(
        "orchestrator.cli.utils.remote.dispatch"
    )

    fake_proc = _FakePortForwardProcess(
        stdout_data=b"",
        stderr_data=b"service not found\n",
        poll_value=1,
    )

    with (
        patch.object(
            remote_dispatch_module,
            "_find_port_forward_tool",
            return_value="kubectl",
        ),
        patch.object(
            remote_dispatch_module.subprocess, "Popen", return_value=fake_proc
        ),
        pytest.raises(RuntimeError, match="exited before becoming ready"),
        _port_forward_context(
            PortForwardConfiguration(
                namespace="my-ns",
                serviceName="missing-service",
                localPort=8265,
            ),
            "http://localhost:8265",
        ),
    ):
        pass


def test_dispatcher_assembles_ray_job_submit_command(
    tmp_path: pathlib.Path,
    cluster_remote_context: RemoteExecutionContext,
    mysql_context_yaml_file: pathlib.Path,
    mysql_project_context: ProjectContext,
) -> None:
    """Verify ray job submit is called with expected arguments."""
    op_file = tmp_path / "operation.yaml"
    op_file.write_text("kind: operation")

    ado_args = [
        "-c",
        str(mysql_context_yaml_file),
        "create",
        "operation",
        "-f",
        str(op_file),
    ]

    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured_cmd.append(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        exit_code = dispatch(
            remote_context=cluster_remote_context,
            project_context=mysql_project_context,
            argv=ado_args,
        )

    assert exit_code == 0
    assert len(captured_cmd) == 1
    cmd = captured_cmd[0]

    assert cmd[0] == "ray"
    assert cmd[1] == "job"
    assert cmd[2] == "submit"
    assert "--address" in cmd
    assert "http://localhost:8265/" in cmd
    assert "--working-dir" in cmd
    assert "--runtime-env" in cmd
    assert "--" in cmd
    assert "ado" in cmd

    # The remote ado command should reference context by fixed filename
    ado_part = cmd[cmd.index("ado") :]
    assert "-c" in ado_part
    assert f"{mysql_project_context.project}.yaml" in ado_part

    # --no-wait should NOT be present since wait=True
    assert "--no-wait" not in cmd


def test_dispatcher_no_wait(
    tmp_path: pathlib.Path,
    mysql_context_yaml_file: pathlib.Path,
    mysql_project_context: ProjectContext,
) -> None:
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        wait=False,
    )
    captured_cmd: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured_cmd.append(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        dispatch(
            remote_context=ctx,
            project_context=mysql_project_context,
            argv=["get", "space"],
        )

    assert "--no-wait" in captured_cmd[0]


def test_dispatcher_propagates_exit_code(
    tmp_path: pathlib.Path,
    mysql_context_yaml_file: pathlib.Path,
    cluster_remote_context: RemoteExecutionContext,
    mysql_project_context: ProjectContext,
) -> None:
    with patch(
        "subprocess.run",
        return_value=MagicMock(returncode=2),
    ):
        exit_code = dispatch(
            remote_context=cluster_remote_context,
            project_context=mysql_project_context,
            argv=["get", "space"],
        )
    assert exit_code == 2


def test_dispatcher_job_type_raises_not_implemented(
    tmp_path: pathlib.Path,
    mysql_context_yaml_file: pathlib.Path,
    mysql_project_context: ProjectContext,
) -> None:
    ctx = RemoteExecutionContext(executionType=JobExecutionType())
    with pytest.raises(NotImplementedError, match="KubeRay"):
        dispatch(
            remote_context=ctx,
            project_context=mysql_project_context,
            argv=["get", "space"],
        )


def test_dispatcher_copies_context_and_op_file(
    tmp_path: pathlib.Path,
    mysql_context_yaml_file: pathlib.Path,
    cluster_remote_context: RemoteExecutionContext,
    mysql_project_context: ProjectContext,
) -> None:
    """Context file and -f files are copied; paths in command are basenames only.

    The working dir is a TemporaryDirectory cleaned up after dispatch, so we
    inspect files from inside the fake_run callback while the dir still exists.
    """
    op_file = tmp_path / "my_operation.yaml"
    op_file.write_text("kind: operation")

    found_context = False
    found_op = False
    found_runtime_env = False

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        nonlocal found_context, found_op, found_runtime_env
        idx = cmd.index("--working-dir")
        working_dir = pathlib.Path(cmd[idx + 1])
        found_context = (working_dir / f"{mysql_project_context.project}.yaml").exists()
        found_op = (working_dir / "my_operation.yaml").exists()
        found_runtime_env = (working_dir / "runtime_env.yaml").exists()
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        dispatch(
            remote_context=cluster_remote_context,
            project_context=mysql_project_context,
            argv=[
                "-c",
                str(mysql_context_yaml_file),
                "create",
                "operation",
                "-f",
                str(op_file),
            ],
        )

    assert found_context, "context yaml was not serialized to working dir"
    assert found_op, "operation yaml was not copied to working dir"
    assert found_runtime_env, "runtime_env.yaml was not generated in working dir"


def test_dispatcher_runtime_env_contents(
    tmp_path: pathlib.Path,
    mysql_context_yaml_file: pathlib.Path,
    mysql_project_context: ProjectContext,
) -> None:
    """Verify runtime_env.yaml has the expected PyPI packages."""
    ctx = RemoteExecutionContext(
        executionType=ClusterExecutionType(clusterUrl="http://localhost:8265"),
        packages=PackageConfiguration(fromPyPI=["ado-core", "ado-ray-tune"]),
        envVars={"OMP_NUM_THREADS": "1"},
    )

    inspected_runtime_env: list[dict] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        idx = cmd.index("--runtime-env")
        runtime_env_path = pathlib.Path(cmd[idx + 1])
        inspected_runtime_env.append(yaml.safe_load(runtime_env_path.read_text()))
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        dispatch(
            remote_context=ctx,
            project_context=mysql_project_context,
            argv=["get", "space"],
        )

    assert len(inspected_runtime_env) == 1
    env = inspected_runtime_env[0]
    assert "ado-core" in env["uv"]
    assert "ado-ray-tune" in env["uv"]
    assert env["env_vars"]["OMP_NUM_THREADS"] == "1"


# ---------------------------------------------------------------------------
# CLI integration: --remote flag
# ---------------------------------------------------------------------------


def test_cli_remote_sqlite_guard(
    tmp_path: pathlib.Path,
    sqlite_context_yaml_file: pathlib.Path,
    remote_context_file: pathlib.Path,
) -> None:
    """ado should fail with a clear error when using --remote with SQLite."""
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "-c",
            str(sqlite_context_yaml_file),
            "--remote",
            str(remote_context_file),
            "get",
            "space",
        ],
    )
    assert result.exit_code == 1
    assert "SQLite" in result.output


def test_cli_remote_missing_file(
    tmp_path: pathlib.Path,
) -> None:
    """ado should fail gracefully if the remote execution context file doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "--remote",
            str(tmp_path / "nonexistent.yaml"),
            "get",
            "space",
        ],
    )
    assert result.exit_code == 1


def test_cli_remote_invalid_yaml(
    tmp_path: pathlib.Path,
    mysql_context_yaml_file: pathlib.Path,
) -> None:
    """ado should fail with a clear error when the remote execution context YAML is invalid."""
    invalid_file = tmp_path / "bad_remote_ctx.yaml"
    invalid_file.write_text("not_a_valid_field: true\nexecutionType: {type: cluster}")

    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "-c",
            str(mysql_context_yaml_file),
            "--remote",
            str(invalid_file),
            "get",
            "space",
        ],
    )
    assert result.exit_code == 1
    assert "not valid" in result.output


def test_cli_execution_context_dispatches_remotely(
    tmp_path: pathlib.Path,
    mysql_context_yaml_file: pathlib.Path,
    remote_context_file: pathlib.Path,
) -> None:
    """When --remote is valid and context is non-SQLite, ray job submit is called."""
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        captured.append(cmd)
        return MagicMock(returncode=0, spec=subprocess.CompletedProcess)

    runner = CliRunner()
    with patch("subprocess.run", side_effect=fake_run):
        result = runner.invoke(
            ado,
            [
                "--override-ado-app-dir",
                str(tmp_path),
                "-c",
                str(mysql_context_yaml_file),
                "--remote",
                str(remote_context_file),
                "get",
                "space",
            ],
        )

    # Should have exited with the ray job submit exit code (0)
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0][0] == "ray"
    assert "submit" in captured[0]


def test_cli_execution_context_auto_sqlite_guard(
    tmp_path: pathlib.Path,
    remote_context_file: pathlib.Path,
) -> None:
    """When no context is manually set, ado auto-creates a local SQLite context.

    Using --remote in this scenario must fail with the SQLite guard
    message (the same as when a SQLite context is explicitly provided).
    """
    runner = CliRunner()
    result = runner.invoke(
        ado,
        [
            "--override-ado-app-dir",
            str(tmp_path),
            "--remote",
            str(remote_context_file),
            "get",
            "space",
        ],
    )
    assert result.exit_code == 1
    assert "SQLite" in result.output


# ---------------------------------------------------------------------------
# Tests for generic arg_parser functions
# ---------------------------------------------------------------------------


def test_strip_flags_generic() -> None:
    """Test generic strip_flags function."""
    from orchestrator.cli.models.remote_submission import (
        CONTEXT_FLAG,
        REMOTE_FLAG,
    )
    from orchestrator.cli.utils.remote.arg_parser import strip_flags

    argv = ["-c", "ctx.yaml", "--remote", "exec.yaml", "create", "op"]

    # Strip only remote context
    result = strip_flags(argv, [REMOTE_FLAG])
    assert result == ["-c", "ctx.yaml", "create", "op"]

    # Strip only context
    result = strip_flags(argv, [CONTEXT_FLAG])
    assert result == ["--remote", "exec.yaml", "create", "op"]

    # Strip both
    result = strip_flags(argv, [REMOTE_FLAG, CONTEXT_FLAG])
    assert result == ["create", "op"]


def test_strip_flags_preserves_order() -> None:
    """Verify that strip_flags maintains exact argument order."""
    from orchestrator.cli.models.remote_submission import FILE_FLAG
    from orchestrator.cli.utils.remote.arg_parser import strip_flags

    argv = ["cmd", "-f", "a.yaml", "arg1", "-f", "b.yaml", "arg2"]
    result = strip_flags(argv, [FILE_FLAG])
    assert result == ["cmd", "arg1", "arg2"]


def test_rewrite_flag_values_generic() -> None:
    """Test generic rewrite_flag_values function."""
    from orchestrator.cli.models.remote_submission import (
        FILE_FLAG,
        RemoteSubmissionFlagMatch,
        RemoteSubmissionFlagSpec,
    )
    from orchestrator.cli.utils.remote.arg_parser import rewrite_flag_values

    def to_uppercase(
        occ: RemoteSubmissionFlagMatch, flag_def: RemoteSubmissionFlagSpec
    ) -> str:
        return occ.value.upper() if occ.value else ""

    argv = ["-f", "file.yaml", "create"]
    result = rewrite_flag_values(argv, [FILE_FLAG], to_uppercase)
    assert result == ["-f", "FILE.YAML", "create"]


def test_extensibility_new_flag() -> None:
    """Test that adding a new flag definition works without code changes."""
    from orchestrator.cli.models.remote_submission import RemoteSubmissionFlagSpec
    from orchestrator.cli.utils.remote.arg_parser import strip_flags

    # Define a new flag
    NEW_FLAG = RemoteSubmissionFlagSpec(
        names=frozenset({"--new-flag"}),
        hasValue=True,
    )

    argv = ["--new-flag", "value", "create", "op"]
    result = strip_flags(argv, [NEW_FLAG])
    assert result == ["create", "op"]


# ---------------------------------------------------------------------------
# Tests for edge cases
# ---------------------------------------------------------------------------


def test_flag_at_end_without_value() -> None:
    """Flag expecting value at end of argv should raise ValueError."""
    from orchestrator.cli.models.remote_submission import FILE_FLAG
    from orchestrator.cli.utils.remote.arg_parser import parse_argv_with_positions

    argv = ["create", "op", "-f"]
    with pytest.raises(ValueError, match="expects a value but is at end"):
        parse_argv_with_positions(argv, [FILE_FLAG])


def test_value_starting_with_dash() -> None:
    """Values starting with dash should be treated as values, not flags."""
    from orchestrator.cli.models.remote_submission import WITH_FLAG
    from orchestrator.cli.utils.remote.arg_parser import parse_argv_with_positions

    argv = ["--with", "key=-123"]
    parsed = parse_argv_with_positions(argv, [WITH_FLAG])
    assert len(parsed.handled_flags) == 1
    assert parsed.handled_flags[0].value == "key=-123"


def test_empty_argv() -> None:
    """Empty argv should be handled gracefully."""
    from orchestrator.cli.models.remote_submission import REMOTE_FLAG
    from orchestrator.cli.utils.remote.arg_parser import strip_flags

    assert strip_flags([], [REMOTE_FLAG]) == []


def test_multiple_occurrences_same_flag() -> None:
    """Multiple occurrences of same flag should all be processed."""
    from orchestrator.cli.models.remote_submission import FILE_FLAG
    from orchestrator.cli.utils.remote.arg_parser import parse_argv_with_positions

    argv = ["-f", "a.yaml", "cmd", "-f", "b.yaml"]
    parsed = parse_argv_with_positions(argv, [FILE_FLAG])
    assert len(parsed.handled_flags) == 2
    assert parsed.handled_flags[0].value == "a.yaml"
    assert parsed.handled_flags[1].value == "b.yaml"


def test_flag_spec_matches() -> None:
    """Test RemoteSubmissionFlagSpec.matches() method."""
    from orchestrator.cli.models.remote_submission import RemoteSubmissionFlagSpec

    flag = RemoteSubmissionFlagSpec(names=frozenset({"-f", "--file"}), hasValue=True)
    assert flag.matches("-f")
    assert flag.matches("--file")
    assert flag.matches("--file=value")
    assert not flag.matches("-x")


def test_flag_spec_extract_value_from_equals_form() -> None:
    """Test RemoteSubmissionFlagSpec.extract_value_from_equals_form() method."""
    from orchestrator.cli.models.remote_submission import RemoteSubmissionFlagSpec

    flag = RemoteSubmissionFlagSpec(names=frozenset({"-f", "--file"}), hasValue=True)
    assert flag.extract_value_from_equals_form("--file=test.yaml") == "test.yaml"
    assert flag.extract_value_from_equals_form("-f=test.yaml") == "test.yaml"
    assert flag.extract_value_from_equals_form("--file") is None
    assert flag.extract_value_from_equals_form("-f") is None


def test_flag_spec_get_canonical_name() -> None:
    """Test RemoteSubmissionFlagSpec.get_canonical_name() method."""
    from orchestrator.cli.models.remote_submission import RemoteSubmissionFlagSpec

    flag = RemoteSubmissionFlagSpec(names=frozenset({"-f", "--file"}), hasValue=True)
    assert flag.get_canonical_name() == "--file"


def test_parse_argv_with_positions_equals_form() -> None:
    """Test parsing flags in --flag=value form."""
    from orchestrator.cli.models.remote_submission import FILE_FLAG
    from orchestrator.cli.utils.remote.arg_parser import parse_argv_with_positions

    argv = ["--file=test.yaml", "create", "op"]
    parsed = parse_argv_with_positions(argv, [FILE_FLAG])
    assert len(parsed.handled_flags) == 1
    assert parsed.handled_flags[0].name == "--file"
    assert parsed.handled_flags[0].value == "test.yaml"
    assert parsed.handled_flags[0].is_equals_form is True
    assert parsed.passthrough_args == [(1, "create"), (2, "op")]


def test_reconstruct_argv_maintains_order() -> None:
    """Test that reconstruct_argv maintains original argument order."""
    from orchestrator.cli.models.remote_submission import CONTEXT_FLAG, FILE_FLAG
    from orchestrator.cli.utils.remote.arg_parser import parse_argv_with_positions

    argv = ["-c", "ctx.yaml", "create", "op", "-f", "op.yaml"]
    parsed = parse_argv_with_positions(argv, [FILE_FLAG, CONTEXT_FLAG])
    reconstructed = parsed.reconstruct_argv()
    assert reconstructed == argv
