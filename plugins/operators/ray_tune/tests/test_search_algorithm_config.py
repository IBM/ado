# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for converting OrchSearchAlgorithm parameters to RayTune parameters."""

import pytest
from ado_ray_tune.config import OrchSearchAlgorithm

optuna_samplers = pytest.importorskip("optuna.samplers")


@pytest.mark.parametrize(
    "params",
    [
        {"sampler": "TPESampler"},
        {"sampler": "TPESampler", "sampler_parameters": {}},
        {"sampler": "TPESampler", "sampler_parameters": {"seed": 1}},
    ],
    ids=["no-sampler-parameters", "empty-sampler-parameters", "sampler-parameters"],
)
def test_optuna_sampler_is_instantiated(params: dict) -> None:
    """The named Optuna sampler is instantiated with or without sampler_parameters."""
    search_alg = OrchSearchAlgorithm(name="optuna", params=params)

    ray_tune_parameters = search_alg.parameters_for_ray_tune()

    assert isinstance(ray_tune_parameters["sampler"], optuna_samplers.TPESampler)
    assert "sampler_parameters" not in ray_tune_parameters


def test_optuna_sampler_parameters_without_sampler_raises() -> None:
    """sampler_parameters without a sampler raises a ValueError."""
    search_alg = OrchSearchAlgorithm(
        name="optuna", params={"sampler_parameters": {"seed": 1}}
    )

    with pytest.raises(ValueError, match="no sampler specified"):
        search_alg.parameters_for_ray_tune()


def test_nevergrad_optimizer_name_is_resolved() -> None:
    """The nevergrad optimizer name is replaced by the registry entry."""
    nevergrad = pytest.importorskip("nevergrad")
    search_alg = OrchSearchAlgorithm(
        name="nevergrad",
        params={"optimizer": "NGOpt", "optimizer_kwargs": {"budget": 10}},
    )

    ray_tune_parameters = search_alg.parameters_for_ray_tune()

    assert ray_tune_parameters["optimizer"] is nevergrad.optimizers.registry["NGOpt"]
    assert ray_tune_parameters["optimizer_kwargs"] == {"budget": 10}
