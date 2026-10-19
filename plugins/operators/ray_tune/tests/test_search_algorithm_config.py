# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for converting OrchSearchAlgorithm parameters to RayTune parameters."""

import pytest
from ado_ray_tune.config import OrchSearchAlgorithm

optuna_samplers = pytest.importorskip("optuna.samplers")


def test_example_configuration_uses_optuna_tpe() -> None:
    """The ray_tune operation template uses Optuna TPE with multivariate sampling."""
    from ado_ray_tune.operator import RayTune

    search_alg = RayTune.operator_metadata().example_configuration.tuneConfig.search_alg

    assert search_alg.name == "optuna"
    assert search_alg.params["sampler"] == "TPESampler"
    assert search_alg.params["sampler_parameters"]["multivariate"] is True
    ray_tune_parameters = search_alg.parameters_for_ray_tune()
    assert isinstance(ray_tune_parameters["sampler"], optuna_samplers.TPESampler)
    assert ray_tune_parameters["sampler"]._multivariate is True


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

    with pytest.raises(
        ValueError,
        match="Optuna sampler parameters specified but no sampler specified",
    ):
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


def test_open_categorical_dimension_is_rejected() -> None:
    """ray_tune cannot build a search space from an open categorical dimension."""
    from ado_ray_tune.operator import property_domain_to_ray_distribution

    from ado.schema.domain import PropertyDomain, VariableTypeEnum

    domain = PropertyDomain(
        variableType=VariableTypeEnum.OPEN_CATEGORICAL_VARIABLE_TYPE
    )
    with pytest.raises(
        ValueError,
        match=(
            "ray_tune cannot sample an open categorical dimension. "
            "List its values to close it, or use a random_walk selector."
        ),
    ):
        property_domain_to_ray_distribution(domain)
