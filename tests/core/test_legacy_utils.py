# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

"""Tests for legacy migrator utility functions"""

from ado.utilities.dictionaries import (
    get_nested_value,
    get_parent_dict_and_key,
    has_nested_field,
    remove_nested_field,
    set_nested_value,
)


class TestGetParentDictAndKey:
    """Tests for get_parent_dict_and_key function"""

    def test_simple_path(self) -> None:
        """Test getting a simple top-level field"""
        data = {"config": {"properties": ["a", "b"]}}
        parent, field = get_parent_dict_and_key(data, "config")
        assert parent == data
        assert field == "config"

    def test_nested_path(self) -> None:
        """Test getting a nested field"""
        data = {"config": {"specification": {"module": {"moduleType": "test"}}}}
        parent, field = get_parent_dict_and_key(
            data, "config.specification.module.moduleType"
        )
        assert parent == {"moduleType": "test"}
        assert field == "moduleType"

    def test_nonexistent_path(self) -> None:
        """Test getting a path that doesn't exist"""
        data = {"config": {}}
        parent, field = get_parent_dict_and_key(data, "config.nonexistent.field")
        assert parent is None
        assert field is None

    def test_path_through_non_dict(self) -> None:
        """Test path that goes through a non-dict value"""
        data = {"config": "string_value"}
        parent, field = get_parent_dict_and_key(data, "config.field")
        assert parent is None
        assert field is None


class TestGetNestedValue:
    """Tests for get_nested_value function"""

    def test_simple_path(self) -> None:
        """Test getting a simple top-level value"""
        data = {"config": {"properties": ["a", "b"]}}
        value = get_nested_value(data, "config.properties")
        assert value == ["a", "b"]

    def test_nested_path(self) -> None:
        """Test getting a nested value"""
        data = {"config": {"specification": {"module": {"moduleType": "test"}}}}
        value = get_nested_value(data, "config.specification.module.moduleType")
        assert value == "test"

    def test_nonexistent_path(self) -> None:
        """Test getting a path that doesn't exist"""
        data = {"config": {}}
        value = get_nested_value(data, "config.nonexistent.field")
        assert value is None

    def test_path_through_non_dict(self) -> None:
        """Test path that goes through a non-dict value"""
        data = {"config": "string_value"}
        value = get_nested_value(data, "config.field")
        assert value is None

    def test_get_dict_value(self) -> None:
        """Test getting a dict value"""
        data = {"config": {"nested": {"key": "value"}}}
        value = get_nested_value(data, "config.nested")
        assert value == {"key": "value"}

    def test_get_none_value(self) -> None:
        """Test getting a field that exists but has None value"""
        data = {"config": {"test": None}}
        value = get_nested_value(data, "config.test")
        assert value is None


class TestSetNestedValue:
    """Tests for set_nested_value function"""

    def test_set_simple_value(self) -> None:
        """Test setting a simple nested value"""
        data = {"config": {}}
        result = set_nested_value(data, "config.test", "value")
        assert result is True
        assert data["config"]["test"] == "value"

    def test_set_deeply_nested_value(self) -> None:
        """Test setting a deeply nested value"""
        data = {"config": {"specification": {"module": {}}}}
        result = set_nested_value(data, "config.specification.module.type", "new_type")
        assert result is True
        assert data["config"]["specification"]["module"]["type"] == "new_type"

    def test_set_nonexistent_path(self) -> None:
        """Test setting a value on a nonexistent path"""
        data = {"config": {}}
        result = set_nested_value(data, "config.nonexistent.field", "value")
        assert result is False
        assert "nonexistent" not in data["config"]

    def test_overwrite_existing_value(self) -> None:
        """Test overwriting an existing value"""
        data = {"config": {"test": "old_value"}}
        result = set_nested_value(data, "config.test", "new_value")
        assert result is True
        assert data["config"]["test"] == "new_value"


class TestRemoveNestedField:
    """Tests for remove_nested_field function"""

    def test_remove_simple_field(self) -> None:
        """Test removing a simple field"""
        data = {"config": {"properties": ["a", "b"], "other": "value"}}
        result = remove_nested_field(data, "config.properties")
        assert result is True
        assert "properties" not in data["config"]
        assert data["config"]["other"] == "value"

    def test_remove_deeply_nested_field(self) -> None:
        """Test removing a deeply nested field"""
        data = {
            "config": {
                "specification": {"module": {"moduleType": "old", "other": "value"}}
            }
        }
        result = remove_nested_field(data, "config.specification.module.moduleType")
        assert result is True
        assert "moduleType" not in data["config"]["specification"]["module"]
        assert data["config"]["specification"]["module"]["other"] == "value"

    def test_remove_nonexistent_field(self) -> None:
        """Test removing a field that doesn't exist"""
        data = {"config": {}}
        result = remove_nested_field(data, "config.nonexistent")
        assert result is False

    def test_remove_field_idempotent(self) -> None:
        """Test that removing a field twice is safe"""
        data = {"config": {"test": "value"}}
        result1 = remove_nested_field(data, "config.test")
        assert result1 is True
        result2 = remove_nested_field(data, "config.test")
        assert result2 is False


class TestHasNestedField:
    """Tests for has_nested_field function"""

    def test_has_simple_field(self) -> None:
        """Test checking for a simple field"""
        data = {"config": {"properties": ["a", "b"]}}
        assert has_nested_field(data, "config.properties") is True

    def test_has_deeply_nested_field(self) -> None:
        """Test checking for a deeply nested field"""
        data = {"config": {"specification": {"module": {"moduleType": "test"}}}}
        assert has_nested_field(data, "config.specification.module.moduleType") is True

    def test_has_nonexistent_field(self) -> None:
        """Test checking for a field that doesn't exist"""
        data = {"config": {}}
        assert has_nested_field(data, "config.nonexistent") is False

    def test_has_field_through_non_dict(self) -> None:
        """Test checking for a field through a non-dict value"""
        data = {"config": "string_value"}
        assert has_nested_field(data, "config.field") is False


class TestIntegration:
    """Integration tests combining multiple utility functions"""

    def test_check_set_remove_workflow(self) -> None:
        """Test a complete workflow: check, set, remove"""
        data = {"config": {}}

        # Check field doesn't exist
        assert has_nested_field(data, "config.test") is False

        # Set the field
        assert set_nested_value(data, "config.test", "value") is True
        assert has_nested_field(data, "config.test") is True
        assert data["config"]["test"] == "value"

        # Remove the field
        assert remove_nested_field(data, "config.test") is True
        assert has_nested_field(data, "config.test") is False

    def test_complex_nested_structure(self) -> None:
        """Test with a complex nested structure"""
        data = {
            "metadata": {"name": "test"},
            "config": {
                "specification": {
                    "module": {"moduleType": "entity_source", "moduleName": "test"}
                }
            },
        }

        # Check existing field
        assert has_nested_field(data, "config.specification.module.moduleType") is True

        # Modify the field
        assert (
            set_nested_value(
                data, "config.specification.module.moduleType", "sample_store"
            )
            is True
        )
        assert data["config"]["specification"]["module"]["moduleType"] == "sample_store"

        # Remove another field
        assert (
            remove_nested_field(data, "config.specification.module.moduleName") is True
        )
        assert "moduleName" not in data["config"]["specification"]["module"]

        # Original structure still intact
        assert data["metadata"]["name"] == "test"


# Made with Bob
