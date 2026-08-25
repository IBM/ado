# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for ModuleConf validators"""

import logging

import pytest

from ado.core.samplestore.config import SampleStoreModuleConf
from ado.modules.module import ModuleConf, ModuleTypeEnum


def test_legacy_orchestrator_prefix_is_rewritten(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """orchestrator.* prefix is rewritten to ado.* and a warning is emitted."""
    with caplog.at_level(logging.WARNING, logger="module"):
        conf = ModuleConf(
            moduleType=ModuleTypeEnum.SAMPLE_STORE,
            moduleName="orchestrator.core.samplestore.sql",
            moduleClass="SQLSampleStore",
        )
    assert conf.moduleName == "ado.core.samplestore.sql"
    assert "orchestrator.core.samplestore.sql" in caplog.text
    assert "ado.core.samplestore.sql" in caplog.text
    assert (
        "ibm.github.io/ado/latest/user-guide/advanced/migration-ado-1x-to-2x/"
        in caplog.text
    )


def test_orchestrator_not_at_prefix_is_unchanged() -> None:
    """orchestrator. mid-string is not rewritten."""
    conf = ModuleConf(
        moduleType=ModuleTypeEnum.GENERIC,
        moduleName="ado.something.orchestrator.module",
    )
    assert conf.moduleName == "ado.something.orchestrator.module"


def test_none_triggers_default_for_sample_store_type() -> None:
    """None moduleName is defaulted to ado.core.samplestore.sql for SAMPLE_STORE."""
    conf = ModuleConf(
        moduleType=ModuleTypeEnum.SAMPLE_STORE,
        moduleClass="SQLSampleStore",
    )
    assert conf.moduleName == "ado.core.samplestore.sql"


def test_samplestore_subclass_emits_rich_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SampleStoreModuleConf emits the resource-specific rich WARN callout."""
    conf = SampleStoreModuleConf(
        moduleName="orchestrator.core.samplestore.sql",
        moduleClass="SQLSampleStore",
    )
    assert conf.moduleName == "ado.core.samplestore.sql"
    stderr = capsys.readouterr().err
    assert "WARN" in stderr
    assert "moduleName" in stderr
    assert "ado upgrade samplestores" in stderr
    assert (
        "ibm.github.io/ado/latest/user-guide/advanced/migration-ado-1x-to-2x/" in stderr
    )
