# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pytest

import orchestrator.core.samplestore.base
import orchestrator.core.samplestore.sql
import orchestrator.modules.operators.randomwalk
import orchestrator.utilities
import orchestrator.utilities.environment
import orchestrator.utilities.location
from orchestrator.metastore.project import ProjectContext
from orchestrator.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)


def test_discovery_storage_conf_dump_reload(orchestrator_project_name: str) -> None:

    conf = ProjectContext(
        project=orchestrator_project_name,
        metadataStore=orchestrator.utilities.location.SQLStoreConfiguration(
            scheme="mysql+pymysql",
            host="localhost",
            port=3306,
            user="someuser",
            password="somepass",
            database=orchestrator_project_name,
            sslVerify=False,
        ),
    )

    assert conf.metadataStore.password is not None

    # Dump and load model
    d = conf.model_dump()
    newconf = ProjectContext.model_validate(d)

    assert newconf.metadataStore.password is not None
    assert newconf.metadataStore.password == conf.metadataStore.password
    assert newconf.metadataStore.host == conf.metadataStore.host

    # Dump and load model - exclude password
    d = conf.model_dump(exclude={"metadataStore": {"password": True}})
    newconf = ProjectContext.model_validate(d)

    assert newconf.metadataStore.password is None
    assert newconf.metadataStore.password != conf.metadataStore.password
    assert newconf.metadataStore.host == conf.metadataStore.host


def test_default_plugin_configs(module_config: ModuleConf) -> None:
    if module_config.moduleType == ModuleTypeEnum.OPERATION:
        assert (
            load_module_class_or_function(module_config)
            == orchestrator.modules.operators.randomwalk.RandomWalk
        )
    elif module_config.moduleType == ModuleTypeEnum.SAMPLE_STORE:
        assert (
            load_module_class_or_function(module_config)
            == orchestrator.core.samplestore.sql.SQLSampleStore
        )

    # ACTUATOR should not have a default
    # Its moduleClass will be None and should raise TypeError on trying to load it.
    if module_config.moduleType == ModuleTypeEnum.ACTUATOR:
        with pytest.raises(TypeError):
            load_module_class_or_function(module_config)
