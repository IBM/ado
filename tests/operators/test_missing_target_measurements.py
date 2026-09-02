# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Integration test for TrimParameters.missingTargetMeasurements.

This test verifies that TRIM, when configured with
``missingTargetMeasurements.mode = Skip``, completes successfully despite
encountering both error and missing-target measurements.
"""

import pathlib
from collections.abc import Callable

import pytest
import trim_custom_experiments.experiments  # noqa: F401 — registers controlled_error and other experiments
import yaml
from testcontainers.community.mysql import MySqlContainer

import ado.modules.operators.randomwalk  # noqa: F401
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.resource import OperationExitStateEnum
from ado.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from ado.core.samplestore.sql import SQLSampleStore
from ado.metastore.project import ProjectContext
from ado.modules.operators.collections import characterize

pytest.importorskip("autogluon")

from trim.trim_pydantic import (  # noqa: E402
    AutoGluonArgs,
    MissingTargetMeasurementMode,
    MissingTargetMeasurements,
    SamplingBudget,
    StoppingCriterion,
    TrimParameters,
)

# ---------------------------------------------------------------------------
# Discovery space configuration
# ---------------------------------------------------------------------------
_SPACE_YAML = """\
metadata:
  name: trim_controlled_error_space
entitySpace:
  - identifier: foo
    propertyDomain:
      domainRange: [0, 99]
      interval: 1
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: controlled_error
"""

# ---------------------------------------------------------------------------
# AutoGluon settings — minimal for speed
# ---------------------------------------------------------------------------
_AUTOGLUON_ARGS = AutoGluonArgs(
    fitArgs={
        "time_limit": 10,
        "presets": "medium_quality",
        "auto_stack": False,
        "excluded_model_types": ["CAT", "NN_TORCH", "FASTAI", "GBM", "XGB", "RF"],
    },
    tabularPredictorArgs={
        "problem_type": "regression",
        "verbosity": 0,
    },
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def controlled_error_space(
    mysql_test_instance: MySqlContainer,
    valid_ado_project_context: ProjectContext,
    create_sample_store: Callable[[SampleStoreConfiguration], SQLSampleStore],
    create_space: Callable[[DiscoverySpaceConfiguration, str], DiscoverySpace],
) -> DiscoverySpace:
    """Discovery space backed by the controlled_error custom experiment."""
    space_conf = DiscoverySpaceConfiguration.model_validate(yaml.safe_load(_SPACE_YAML))
    sample_store = create_sample_store(
        SampleStoreConfiguration(
            specification=SampleStoreSpecification(
                module=SampleStoreModuleConf(
                    moduleClass="SQLSampleStore",
                    moduleName="ado.core.samplestore.sql",
                ),
            )
        )
    )
    space = create_space(space_conf, sample_store.identifier)
    return DiscoverySpace.from_stored_configuration(
        project_context=valid_ado_project_context,
        space_identifier=space.uri,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.timeout(900)
def test_trim_skips_missing_target_measurements(
    controlled_error_space: DiscoverySpace,
    tmp_path: pathlib.Path,
) -> None:
    """TRIM with mode=Skip should succeed despite missing-target measurements."""
    params = TrimParameters(
        targetOutput="bar",
        missingTargetMeasurements=MissingTargetMeasurements(
            mode=MissingTargetMeasurementMode.Skip
        ),
        samplingBudget=SamplingBudget(minPoints=40, maxPoints=100),
        iterationSize=1,
        outputDirectory=str(tmp_path / "trim_models"),
        debugDirectory=str(tmp_path / "debug_output"),
        stoppingCriterion=StoppingCriterion(enabled=False),
        autoGluonArgs=_AUTOGLUON_ARGS,
        finalModelAutoGluonArgs=_AUTOGLUON_ARGS,
    )

    trim_fn = characterize.operators["trim"].function
    assert trim_fn is not None

    output = trim_fn(
        controlled_error_space,
        **params.model_dump(),
    )

    assert output.exitStatus is not None
    assert output.exitStatus.exit_state == OperationExitStateEnum.SUCCESS
