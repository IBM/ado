# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for _build_ray_runtime_env_with_extra in actuator.py."""

from unittest.mock import MagicMock, patch

from ado_actuators.vllm_performance.actuator import _build_ray_runtime_env_with_extra

# Patch targets:
# RAY_CTX — where extract_package_specs_from_job_env calls Ray.
# PKG_VER  — the already-imported `version` name inside the actuator module.
RAY_CTX = "ado.utilities.environment.ray.get_runtime_context"
PKG_VER = "ado_actuators.vllm_performance.actuator.version"


def _mock_runtime_context(uv_packages: list[str]) -> MagicMock:
    """Return a mock Ray runtime context with the given uv package list."""
    ctx = MagicMock()
    ctx.runtime_env = {"uv": {"packages": uv_packages}}
    return ctx


class TestBuildRayRuntimeEnvWithExtra:
    """Unit tests for _build_ray_runtime_env_with_extra."""

    def test_all_packages_from_job_env(self) -> None:
        """When all five packages are in the job env, their specs are forwarded."""
        job_packages = [
            "ado-core==1.0",
            "ado-vllm-performance==2.0",
            "ray==2.9.0",
            "vllm==0.12.0",
            "guidellm==0.5.0",
        ]
        with patch(RAY_CTX, return_value=_mock_runtime_context(job_packages)):
            result = _build_ray_runtime_env_with_extra("vllm")

        deps = result["uv"]
        assert any(d.startswith("ado-core") and "==1.0" in d for d in deps)
        assert "ado-vllm-performance[vllm]==2.0" in deps
        assert any(d.startswith("ray") and "==2.9.0" in d for d in deps)
        assert any(d.startswith("vllm") and "==0.12.0" in d for d in deps)
        assert any(d.startswith("guidellm") and "==0.5.0" in d for d in deps)

    def test_benchmark_tool_extra_always_injected(self) -> None:
        """ado-vllm-performance always carries [benchmark_tool] regardless of job env."""
        job_packages = ["ado-vllm-performance==2.0"]
        with (
            patch(RAY_CTX, return_value=_mock_runtime_context(job_packages)),
            patch(PKG_VER, side_effect=lambda n: "1.0"),
        ):
            result_vllm = _build_ray_runtime_env_with_extra("vllm")
            result_guidellm = _build_ray_runtime_env_with_extra("guidellm")

        assert any("[vllm]" in d for d in result_vllm["uv"])
        assert any("[guidellm]" in d for d in result_guidellm["uv"])

    def test_ado_core_extras_preserved_from_job_env(self) -> None:
        """ado-core extras from the job env spec are preserved."""
        job_packages = ["ado-core[foo]==1.0"]
        with (
            patch(RAY_CTX, return_value=_mock_runtime_context(job_packages)),
            patch(PKG_VER, side_effect=lambda n: "0.0"),
        ):
            result = _build_ray_runtime_env_with_extra("vllm")

        assert any("ado-core[foo]" in d for d in result["uv"])

    def test_fallback_pins_installed_version_when_not_in_job_env(self) -> None:
        """When a package is absent from the job env, the installed version is pinned."""
        installed = {
            "ado-core": "1.1",
            "ado-vllm-performance": "2.2",
            "ray": "3.0",
            "vllm": "0.9",
            "guidellm": "0.5",
        }
        with (
            patch(RAY_CTX, return_value=_mock_runtime_context([])),
            patch(PKG_VER, side_effect=lambda n: installed[n]),
        ):
            result = _build_ray_runtime_env_with_extra("vllm")

        deps = result["uv"]
        assert "ado-core==1.1" in deps
        assert "ado-vllm-performance[vllm]==2.2" in deps
        assert "ray==3.0" in deps
        assert "vllm==0.9" in deps
        assert "guidellm==0.5" in deps

    def test_package_skipped_when_not_in_job_env_and_not_installed(self) -> None:
        """A package absent from both the job env and the local install is skipped silently."""
        from importlib.metadata import PackageNotFoundError

        job_packages = [
            "ado-core==1.0",
            "ado-vllm-performance==2.0",
            "ray==2.9",
            "vllm==0.12.0",
        ]

        def fake_version(n: str) -> str:
            if n == "guidellm":
                raise PackageNotFoundError(n)
            return "0.0"

        with (
            patch(RAY_CTX, return_value=_mock_runtime_context(job_packages)),
            patch(PKG_VER, side_effect=fake_version),
        ):
            result = _build_ray_runtime_env_with_extra("vllm")

        assert not any(d.startswith("guidellm") for d in result["uv"])
