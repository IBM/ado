# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for TrimParameters construction."""

from trim.trim_pydantic import AutoGluonArgs, TrimParameters


def test_shared_autogluon_args_are_copied() -> None:
    """Each model stage keeps its own AutoGluon arguments.

    A single AutoGluonArgs instance passed for both stages must not stay
    shared, so a path written on one stage does not appear on the other.
    """
    args = AutoGluonArgs(
        tabularPredictorArgs={"problem_type": "regression", "verbosity": 0}
    )
    params = TrimParameters(
        targetOutput="bar",
        autoGluonArgs=args,
        finalModelAutoGluonArgs=args,
    )

    assert params.autoGluonArgs is not params.finalModelAutoGluonArgs
    assert (
        params.autoGluonArgs.tabularPredictorArgs
        is not params.finalModelAutoGluonArgs.tabularPredictorArgs
    )

    params.autoGluonArgs.tabularPredictorArgs["path"] = "trim_models"
    params.finalModelAutoGluonArgs.tabularPredictorArgs["path"] = (
        "trim_models_finalized"
    )

    assert params.autoGluonArgs.tabularPredictorArgs["path"] == "trim_models"
    assert (
        params.finalModelAutoGluonArgs.tabularPredictorArgs["path"]
        == "trim_models_finalized"
    )


def test_distinct_final_model_args_are_kept() -> None:
    """A final-model configuration that differs from the default is preserved."""
    args = AutoGluonArgs(tabularPredictorArgs={"verbosity": 0})
    final_args = AutoGluonArgs(tabularPredictorArgs={"verbosity": 2})
    params = TrimParameters(
        targetOutput="bar",
        autoGluonArgs=args,
        finalModelAutoGluonArgs=final_args,
    )

    assert params.finalModelAutoGluonArgs.tabularPredictorArgs["verbosity"] == 2
    assert params.autoGluonArgs is not params.finalModelAutoGluonArgs
