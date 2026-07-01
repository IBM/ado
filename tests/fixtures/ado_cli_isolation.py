# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Isolate ado CLI configuration per test to avoid pytest-xdist races."""

import pathlib

import pytest

from orchestrator.cli.core.config import AdoConfiguration


def apply_isolated_ado_app_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch ``AdoConfiguration.load`` to use *tmp_path* when no override is set.

    Parallel pytest-xdist workers otherwise share the real ``~/.config/ado``
    directory. Concurrent ``store()`` calls can leave ``ado_cli_config.json``
    empty while another worker reads it.
    """
    original_load = AdoConfiguration.load.__func__

    @classmethod
    def load_with_isolated_app_dir(
        cls: type[AdoConfiguration],
        *args: object,
        _override_config_dir: pathlib.Path | None = None,
        **kwargs: object,
    ) -> AdoConfiguration:
        if _override_config_dir is None:
            _override_config_dir = tmp_path
        return original_load(
            cls, *args, _override_config_dir=_override_config_dir, **kwargs
        )

    monkeypatch.setattr(AdoConfiguration, "load", load_with_isolated_app_dir)


@pytest.fixture
def isolated_ado_app_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> pathlib.Path:
    """Fixture wrapper around :func:`apply_isolated_ado_app_dir`."""
    apply_isolated_ado_app_dir(tmp_path, monkeypatch)
    return tmp_path
