# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import datetime

import pydantic
import pytest
import yaml

from ado.core.operation.config import (
    DiscoveryOperationConfiguration,
    DiscoveryOperationEnum,
    DiscoveryOperationResourceConfiguration,
    ScriptOperatorConf,
)
from ado.core.operation.resource import (
    OperationExitStateEnum,
    OperationResource,
    OperationResourceEventEnum,
    OperationResourceStatus,
)
from ado.core.resources import ADOResourceEventEnum, CoreResourceKinds
from ado.modules.module import load_module_class_or_function


def test_discovery_operation_enum_legacy_search_redirects_to_explore() -> None:
    """Legacy 'search' value is accepted and redirected to EXPLORE via _missing_."""
    assert DiscoveryOperationEnum("search") is DiscoveryOperationEnum.EXPLORE


@pytest.fixture
def operation_result() -> dict:

    # Return the default
    return {}


@pytest.fixture
def operation_resource(
    operation_configuration: DiscoveryOperationResourceConfiguration,
) -> OperationResource:

    # This auto-generates the operation identifier
    return OperationResource(
        operatorIdentifier="test_operator",
        operationType=DiscoveryOperationEnum.EXPLORE,
        config=operation_configuration,
    )


def test_operation_resource(operation_resource: OperationResource) -> None:

    assert operation_resource.operatorIdentifier is not None
    assert operation_resource.operatorIdentifier == "test_operator"
    assert operation_resource.identifier is not None
    x = operation_resource.identifier.split("-")
    assert "-".join(x[:2]) == "operation-test_operator"

    assert operation_resource.kind == CoreResourceKinds.OPERATION
    assert operation_resource.identifier.split("-")[0] == "operation"
    assert (
        len(operation_resource.identifier)
        == len("operation") + len("test_operator") + 2 + 8
    )
    assert operation_resource.created < datetime.datetime.now(datetime.timezone.utc)
    assert isinstance(operation_resource.metadata, dict)
    assert operation_resource.config is not None
    assert isinstance(
        operation_resource.config, DiscoveryOperationResourceConfiguration
    )
    assert operation_resource.config.operation.parameters is not None
    assert len(operation_resource.status) == 1


def test_operation_resource_event_status() -> None:
    """Test we can set additional event status for operation resources"""

    # Check we can create a resource with generic field
    status = OperationResourceStatus(event=ADOResourceEventEnum.UPDATED)
    assert status.event == ADOResourceEventEnum.UPDATED
    assert status.recorded_at
    assert not status.exit_state

    # Check we can create a resource with operation field
    status = OperationResourceStatus(event=OperationResourceEventEnum.STARTED)
    assert status.event == OperationResourceEventEnum.STARTED
    assert status.recorded_at
    assert not status.exit_state
    dump = status.model_dump()

    # check deser
    deser = OperationResourceStatus.model_validate(dump)
    assert deser.event == OperationResourceEventEnum.STARTED


def test_operation_resource_exit_state() -> None:
    """Test we can set additional event status for operation resources"""

    # Check we can create an event+exit status for operation
    status = OperationResourceStatus(
        event=OperationResourceEventEnum.FINISHED,
        exit_state=OperationExitStateEnum.FAIL,
    )
    assert status.event == OperationResourceEventEnum.FINISHED
    assert status.recorded_at
    assert status.exit_state == OperationExitStateEnum.FAIL

    # Check dumping with exit_state dumps exit_state
    dump = status.model_dump()
    assert dump.get("event")
    assert dump.get("exit_state")

    # Check deser
    deser = OperationResourceStatus.model_validate(dump)
    assert deser.event == status.event
    assert deser.recorded_at == status.recorded_at
    assert deser.exit_state == status.exit_state

    # Check we can not create an exit-code status without also setting a FINISHED event
    with pytest.raises(pydantic.ValidationError):
        OperationResourceStatus(exit_state=OperationExitStateEnum.FAIL)

    # Check we can not create an exit-code status without also setting a FINISHED event
    with pytest.raises(pydantic.ValidationError):
        OperationResourceStatus(
            event=OperationResourceEventEnum.STARTED,
            exit_state=OperationExitStateEnum.FAIL,
        )

    # Check we can create a resource with operation field
    status = OperationResourceStatus(event=OperationResourceEventEnum.STARTED)
    assert status.event == OperationResourceEventEnum.STARTED
    assert status.recorded_at


def test_operation_config_file_valid(valid_operation_config_file: str) -> None:

    with open(valid_operation_config_file) as f:
        content = f.read()

    op_cfg = DiscoveryOperationResourceConfiguration.model_validate(
        yaml.safe_load(content)
    )

    try:
        module = op_cfg.module
    except AttributeError:
        pass
    else:
        moduleClass = load_module_class_or_function(module)  # type: "ado.modules.operators.base.DiscoveryOperationBase"
        meta = moduleClass.operator_metadata()
        if meta.configuration_model is not None:
            meta.configuration_model.model_validate(op_cfg.parameters)


def test_set_manual_operation_identifier(
    operation_configuration: DiscoveryOperationResourceConfiguration,
) -> None:

    test = OperationResource(
        operatorIdentifier="test",
        identifier="test-xxxdd3",
        operationType=DiscoveryOperationEnum.CHARACTERIZE,
        config=operation_configuration,
    )
    assert test.identifier == "test-xxxdd3"


def test_setting_space_id(
    operation_configuration: DiscoveryOperationResourceConfiguration,
) -> None:

    import pydantic

    # Test setting empty spaces raises an error
    with pytest.raises(pydantic.ValidationError):
        DiscoveryOperationResourceConfiguration(
            spaces=[], operation=DiscoveryOperationConfiguration()
        )

    # Test setting no space raises an error
    with pytest.raises(pydantic.ValidationError):
        DiscoveryOperationResourceConfiguration(
            operation=DiscoveryOperationConfiguration()
        )


def test_add_operation_result(
    operation_resource: OperationResource, operation_result: dict
) -> None:

    pass


def test_script_operator_conf_round_trip() -> None:
    """ScriptOperatorConf serialises and validates through operation configuration."""
    script_module = ScriptOperatorConf(
        name="grid-sweep",
        version="1.0.0",
        operationType=DiscoveryOperationEnum.EXPLORE,
    )
    assert script_module.operationType == DiscoveryOperationEnum.EXPLORE
    assert script_module.operatorIdentifier == "script-grid-sweep-1.0.0"

    operation_configuration = DiscoveryOperationConfiguration(
        module=script_module,
        parameters={"ignored": "value"},
    )
    assert operation_configuration.parameters == {}

    resource_configuration = DiscoveryOperationResourceConfiguration(
        operation=operation_configuration,
        spaces=["space-test123"],
    )

    dumped = resource_configuration.model_dump()
    restored = DiscoveryOperationResourceConfiguration.model_validate(dumped)
    assert isinstance(restored.operation.module, ScriptOperatorConf)
    assert restored.operation.module.name == "grid-sweep"
    assert restored.operation.module.version == "1.0.0"
    assert restored.operation.module.operationType == DiscoveryOperationEnum.EXPLORE
    assert restored.operation.parameters == {}


def test_script_operation_resource_identifier() -> None:
    """OperationResource built from ScriptOperatorConf uses script operator id."""
    script_module = ScriptOperatorConf(
        name="inline-script",
        operationType=DiscoveryOperationEnum.CHARACTERIZE,
    )
    operation_configuration = DiscoveryOperationResourceConfiguration(
        operation=DiscoveryOperationConfiguration(module=script_module),
        spaces=["space-test123"],
    )
    operation = OperationResource(
        operationType=script_module.operationType,
        operatorIdentifier=script_module.operatorIdentifier,
        config=operation_configuration,
    )

    assert operation.operationType == DiscoveryOperationEnum.CHARACTERIZE
    assert operation.operatorIdentifier == "script-inline-script-0.1.0"
    assert operation.identifier.startswith("operation-script-inline-script-0.1.0-")


def test_operation_resource_wrong_kind_raises_validation_error(
    operation_resource: OperationResource,
) -> None:
    """OperationResource rejects a kind value other than OPERATION."""
    data = operation_resource.model_dump()
    data["kind"] = CoreResourceKinds.DISCOVERYSPACE
    with pytest.raises(pydantic.ValidationError):
        OperationResource.model_validate(data)
