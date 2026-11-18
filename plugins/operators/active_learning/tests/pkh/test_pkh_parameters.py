# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for PKH-specific predictor configuration."""

import pytest
from active_learning.pkh.parameters import (
    ActiveLearningAutoGluonParameters,
    PKHParameters,
)


def test_random_forest_rejects_autogluon_parameters() -> None:
    """Keep backend-specific parameters consistent across serialization."""
    parameters = PKHParameters(targetOutput="pressure", numberEntities=20)
    assert PKHParameters.model_validate(parameters.model_dump()) == parameters
    with pytest.raises(ValueError, match="autoGluon parameters must not be provided"):
        PKHParameters(
            targetOutput="pressure",
            numberEntities=20,
            autoGluon=ActiveLearningAutoGluonParameters(),
        )
