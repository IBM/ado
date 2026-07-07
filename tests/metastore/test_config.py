# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest

import ado.core.samplestore.base
import ado.core.samplestore.sql
import ado.modules.operators.randomwalk
import ado.utilities
import ado.utilities.environment
import ado.utilities.location
from ado.metastore.project import ProjectContext
from ado.modules.module import (
    ModuleConf,
    ModuleTypeEnum,
    load_module_class_or_function,
)


def test_discovery_storage_conf_dump_reload() -> None:

    conf = ProjectContext(
        project="project",
        metadataStore=ado.utilities.location.SQLStoreConfiguration(
            scheme="mysql+pymysql",
            host="localhost",
            port=3306,
            user="someuser",
            password="somepass",
            database="project",
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
            == ado.modules.operators.randomwalk.RandomWalk
        )
    elif module_config.moduleType == ModuleTypeEnum.SAMPLE_STORE:
        assert (
            load_module_class_or_function(module_config)
            == ado.core.samplestore.sql.SQLSampleStore
        )

    # ACTUATOR should not have a default
    # Its moduleClass will be None and should raise TypeError on trying to load it.
    if module_config.moduleType == ModuleTypeEnum.ACTUATOR:
        with pytest.raises(TypeError):
            load_module_class_or_function(module_config)
