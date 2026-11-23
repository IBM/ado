# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for optional Ray initialization based on operator requiresRay."""

import os
import signal
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from unittest import mock

import pytest

import ado.core.samplestore.sql  # noqa: F401 — registers SQLSampleStore
import ado.modules.operators.randomwalk  # noqa: F401 — loads explore operators
from ado.core.discoveryspace.config import DiscoverySpaceConfiguration
from ado.core.discoveryspace.resource import DiscoverySpaceResource
from ado.core.discoveryspace.space import DiscoverySpace
from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    FunctionOperationInfo,
    GenericOperatorParameters,
    OperatorReference,
    ScriptOperatorConf,
)
from ado.core.operation.operation import OperationOutput
from ado.core.resources import ADOResourceReference, CoreResourceKinds
from ado.core.samplestore.config import (
    SampleStoreConfiguration,
    SampleStoreModuleConf,
    SampleStoreSpecification,
)
from ado.core.samplestore.utils import create_sample_store_resource
from ado.metastore.project import ProjectContext
from ado.metastore.sqlstore import SQLStore
from ado.modules.operators._cleanup import cleanup_callback_functions
from ado.modules.operators.collections import (
    characterize,
    characterize_operation,
    explore,
)
from ado.schema.experiment import Experiment
from ado.schema.measurementspace import MeasurementSpaceConfiguration
from ado.schema.property import AbstractPropertyDescriptor


def _one_space(
    discoverySpace: DiscoverySpace,
    operationInfo: FunctionOperationInfo | None = None,
    *,
    parameters: GenericOperatorParameters,
) -> OperationOutput:
    """Characterize one discovery space."""
    return OperationOutput()


@contextmanager
def _temporary_characterize(
    *,
    requires_ray: bool,
    operator_function: Callable[..., OperationOutput] | None = None,
) -> Iterator[str]:
    """Register a temporary characterize operator and remove it afterwards."""
    name = f"test-ray-op-{uuid.uuid4().hex[:8]}"
    characterize_operation(
        name=name,
        version="1.0.0",
        configuration_model=GenericOperatorParameters,
        example_configuration=GenericOperatorParameters(),
        description="temporary ray-requirement test operator",
        requires_ray=requires_ray,
    )(operator_function or _one_space)
    try:
        yield name
    finally:
        characterize.operators.pop(name, None)


def _characterize_config(operator_name: str) -> DiscoveryOperationResourceConfiguration:
    """Build a characterize operation config for *operator_name*."""
    return DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName=operator_name,
                operationType=DiscoveryOperationEnum.CHARACTERIZE,
            ),
            parameters={},
        ),
        inputs={
            "discoverySpace": ADOResourceReference(
                identifier="space-test123",
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )
        },
    )


def test_registered_explore_operators_require_ray() -> None:
    """Explore registration always sets requiresRay."""
    assert explore.operators["random_walk"].requiresRay is True


def test_trim_operator_requires_ray() -> None:
    """Trim nests random_walk explore ops, so it must opt into Ray."""
    pytest.importorskip("trim")
    import trim.operator  # noqa: F401 — registers the trim operator

    assert characterize.operators["trim"].requiresRay is True


def test_characterize_decorator_requires_ray_opt_in() -> None:
    """requires_ray=True is stored on OperatorMetadata."""
    with _temporary_characterize(requires_ray=True) as name:
        assert characterize.operators[name].requiresRay is True


def test_characterize_decorator_requires_ray_default_false() -> None:
    """requires_ray defaults to False for non-explore operators."""
    with _temporary_characterize(requires_ray=False) as name:
        assert characterize.operators[name].requiresRay is False


def test_requires_ray_follows_operator_metadata() -> None:
    """requires_ray reads requiresRay from the registered operator."""
    with _temporary_characterize(requires_ray=False) as name:
        assert _characterize_config(name).requires_ray() is False

    with _temporary_characterize(requires_ray=True) as name:
        assert _characterize_config(name).requires_ray() is True


def test_requires_ray_true_for_explore_reference() -> None:
    """Explore operator references always require Ray."""
    config = DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(
            module=OperatorReference(
                operatorName="random_walk",
                operationType=DiscoveryOperationEnum.EXPLORE,
            ),
            parameters={"numberEntities": 1, "batchSize": 1},
        ),
        inputs={
            "discoverySpace": ADOResourceReference(
                identifier="space-test123",
                kind=CoreResourceKinds.DISCOVERYSPACE,
            )
        },
    )
    assert config.requires_ray() is True


def test_requires_ray_true_for_script_operator() -> None:
    """Script operators conservatively require Ray."""
    config = DiscoveryOperationResourceConfiguration.model_construct(
        operation=DiscoveryOperationConfiguration.model_construct(
            module=ScriptOperatorConf(name="inline"),
            parameters={},
        ),
        inputs={},
    )
    assert config.requires_ray() is True


def test_requires_ray_rejects_unexpected_module_type() -> None:
    """requires_ray raises when operation.module is not a known type."""
    config = DiscoveryOperationResourceConfiguration.model_construct(
        operation=DiscoveryOperationConfiguration.model_construct(
            module=object(),  # type: ignore[arg-type]
            parameters={},
        ),
        inputs={},
    )
    with pytest.raises(TypeError, match=r"Unexpected operation\.module type"):
        config.requires_ray()


def _store_discovery_space(project_context: ProjectContext) -> tuple[SQLStore, str]:
    """Persist a sample store and a discovery space with one experiment."""
    metastore = SQLStore(project_context=project_context)
    sample_store_configuration = SampleStoreConfiguration(
        specification=SampleStoreSpecification(
            module=SampleStoreModuleConf(
                moduleClass="SQLSampleStore",
                moduleName="ado.core.samplestore.sql",
            ),
        )
    )
    sample_store_configuration.specification.storageLocation = (
        project_context.metadataStore
    )
    _resource, sample_store = create_sample_store_resource(
        sample_store_configuration, metastore
    )
    experiment = Experiment(
        actuatorIdentifier="test",
        identifier="noop-exp",
        targetProperties=[AbstractPropertyDescriptor(identifier="output")],
    )
    space = DiscoverySpaceResource(
        identifier=f"space-{uuid.uuid4().hex[:8]}",
        config=DiscoverySpaceConfiguration(
            sampleStoreIdentifier=sample_store.identifier,
            experiments=MeasurementSpaceConfiguration(experiments=[experiment]),
        ),
    )
    metastore.addResource(space)
    return metastore, space.identifier


@pytest.mark.parametrize(
    ("requires_ray", "in_ray_job"),
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_orchestrate_ray_initialization(
    requires_ray: bool,
    in_ray_job: bool,
    valid_ado_sqlite_project_context: ProjectContext,
) -> None:
    """Start Ray only when the operator requires it or this process is a Ray job.

    SIGTERM handling is installed in every case. ``ray.init`` and
    ``ray.shutdown`` are spies so the test leaves the session Ray cluster
    alone. ``time.sleep`` is patched because Ray shutdown waits for logs.
    """
    from ado.modules.operators import orchestrate as orchestrate_module

    metastore, space_identifier = _store_discovery_space(
        valid_ado_sqlite_project_context
    )
    observed: dict[str, bool] = {}

    def one_space(
        discoverySpace: DiscoverySpace,
        operationInfo: FunctionOperationInfo | None = None,
        *,
        parameters: GenericOperatorParameters,
    ) -> OperationOutput:
        """Characterize one stored discovery space."""
        del discoverySpace, operationInfo, parameters
        handler = signal.getsignal(signal.SIGTERM)
        observed["sigterm_handler_installed"] = callable(handler) and handler not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        )
        observed["orchestrate_cleanup_registered"] = (
            "orchestrate" in cleanup_callback_functions
        )
        return OperationOutput()

    environment = (
        {"RAY_JOB_CONFIG_JSON_ENV_VAR": '{"metadata": {}}'} if in_ray_job else {}
    )
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
    try:
        with _temporary_characterize(
            requires_ray=requires_ray,
            operator_function=one_space,
        ) as name:
            config = DiscoveryOperationResourceConfiguration(
                operation=DiscoveryOperationConfiguration(
                    module=OperatorReference(
                        operatorName=name,
                        operationType=DiscoveryOperationEnum.CHARACTERIZE,
                    ),
                    parameters={},
                ),
                inputs={
                    "discoverySpace": ADOResourceReference(
                        identifier=space_identifier,
                        kind=CoreResourceKinds.DISCOVERYSPACE,
                    )
                },
            )
            with (
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(orchestrate_module.ray, "init") as ray_init,
                mock.patch.object(orchestrate_module.ray, "shutdown") as ray_shutdown,
                mock.patch("time.sleep"),
            ):
                if not in_ray_job:
                    os.environ.pop("RAY_JOB_CONFIG_JSON_ENV_VAR", None)
                output = orchestrate_module.orchestrate(
                    operation_resource_configuration=config,
                    project_context=valid_ado_sqlite_project_context,
                )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

    assert isinstance(output, OperationOutput)
    assert observed["sigterm_handler_installed"] is True
    assert observed["orchestrate_cleanup_registered"] is True
    operation_identifiers = metastore.getResourceIdentifiersOfKind("operation")[
        "IDENTIFIER"
    ]
    assert len(operation_identifiers) == 1
    if requires_ray or in_ray_job:
        ray_init.assert_called_once()
        ray_shutdown.assert_called_once()
    else:
        ray_init.assert_not_called()
        ray_shutdown.assert_not_called()
