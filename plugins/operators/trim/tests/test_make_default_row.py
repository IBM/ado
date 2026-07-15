# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pandas as pd
import pytest
from trim.trim_sampler import _make_default_row

from ado.schema.entity import (
    Entity,
    entity_identifier_from_properties_and_values,
)
from ado.schema.property import ConstitutiveProperty
from ado.schema.property_value import ConstitutivePropertyValue


@pytest.fixture
def simple_entity() -> Entity:
    """Entity with two constitutive properties; identifier is auto-generated."""
    cp_a = ConstitutiveProperty(identifier="temperature")
    cp_b = ConstitutiveProperty(identifier="pressure")
    return Entity(
        constitutive_property_values=(
            ConstitutivePropertyValue(property=cp_a, value=25.0),
            ConstitutivePropertyValue(property=cp_b, value=5.0),
        ),
        measurement_results=[],
    )


def test_make_default_row_columns(simple_entity: Entity) -> None:
    """Returned DataFrame has identifier, cp columns, and target column."""
    row = _make_default_row(simple_entity, target_output="output", default_value=-1.0)
    assert list(row.columns) == ["identifier", "temperature", "pressure", "output"]


def test_make_default_row_values(simple_entity: Entity) -> None:
    """Returned DataFrame contains the correct values."""
    expected_id = entity_identifier_from_properties_and_values(
        {"temperature": 25.0, "pressure": 5.0}
    )
    row = _make_default_row(simple_entity, target_output="output", default_value=-1.0)
    assert row["identifier"].iloc[0] == expected_id
    assert row["temperature"].iloc[0] == 25.0
    assert row["pressure"].iloc[0] == 5.0
    assert row["output"].iloc[0] == -1.0


def test_make_default_row_single_row(simple_entity: Entity) -> None:
    """Returned DataFrame has exactly one row."""
    row = _make_default_row(simple_entity, target_output="output", default_value=0.0)
    assert len(row) == 1


def test_make_default_row_is_dataframe(simple_entity: Entity) -> None:
    """Return type is a pandas DataFrame."""
    row = _make_default_row(simple_entity, target_output="output", default_value=0.0)
    assert isinstance(row, pd.DataFrame)
