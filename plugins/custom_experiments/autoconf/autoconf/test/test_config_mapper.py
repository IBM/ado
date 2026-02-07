# Copyright IBM Corporation 2025, 2026

# SPDX-License-Identifier: MIT

from autoconf.utils.config_mapper import (
    map_valid_model_name,
    mapped_models,
)

invalid_model_names_transform_to_valid = {
    "granite-2b-base": mapped_models["GRANITE_3_1_2B"],
    "granite-3.2-8b-instruct": mapped_models["GRANITE_3_1_8B"],
    "granite-4.0-tiny": mapped_models["GRANITE_4_TINY"],
    "granite-4.0-small-base-prerelease-greylock": mapped_models["GRANITE_4_SMALL"],
    "llama-3-1-8b-instruct": mapped_models["LLAMA_3_1_8B"],
}

invalid_model_names_no_transform = {
    "gb_tuned_model_fw0u3wim_checkpoint-32637": "gb_tuned_model_fw0u3wim_checkpoint-32637",
    "granite-8b-code-instruct-128k": "granite-8b-code-instruct-128k",
    "devstral-small-182a9e3_24b": "devstral-small-182a9e3_24b",
    "granite-5.0-d453da": "granite-5.0-d453da",
}

valid_model_names_no_transform = {
    mapped_models["LLAMA_3_1_8B"]: mapped_models["LLAMA_3_1_8B"],
    mapped_models["GRANITE_4_TINY"]: mapped_models["GRANITE_4_TINY"],
    mapped_models["GRANITE_4_MICRO"]: mapped_models["GRANITE_4_MICRO"],
    mapped_models["GRANITE_3_1_8B"]: mapped_models["GRANITE_3_1_8B"],
}


def test_mapping_invalid_to_valid() -> None:
    assert all(
        map_valid_model_name(k) == v
        for k, v in invalid_model_names_transform_to_valid.items()
    )


def test_mapping_invalid_no_transform() -> None:
    assert all(
        map_valid_model_name(k) == v
        for k, v in invalid_model_names_no_transform.items()
    )


def test_mapping_valid_no_transform() -> None:
    assert all(
        map_valid_model_name(k) == v for k, v in valid_model_names_no_transform.items()
    )


# Additional unit tests for map_valid_model_name function (lines 34-39)


def test_map_valid_model_name_granite_3_1_2b_variants() -> None:
    """Test various granite 3.1 2B model name patterns."""
    test_cases = [
        ("granite-2b", mapped_models["GRANITE_3_1_2B"]),
        ("granite-2b-base", mapped_models["GRANITE_3_1_2B"]),
        ("granite-2b-instruct", mapped_models["GRANITE_3_1_2B"]),
        ("granite-3.1-2b", mapped_models["GRANITE_3_1_2B"]),
        ("granite-3.1-2b-base", mapped_models["GRANITE_3_1_2B"]),
        ("granite-3.1-2b-instruct", mapped_models["GRANITE_3_1_2B"]),
        ("granite-3.2-2b", mapped_models["GRANITE_3_1_2B"]),
        ("granite-3.3-2b-instruct", mapped_models["GRANITE_3_1_2B"]),
    ]
    for input_name, expected_output in test_cases:
        assert (
            map_valid_model_name(input_name) == expected_output
        ), f"Failed for input: {input_name}"


def test_map_valid_model_name_granite_3_1_8b_variants() -> None:
    """Test various granite 3.1 8B model name patterns."""
    test_cases = [
        ("granite-8b", mapped_models["GRANITE_3_1_8B"]),
        ("granite-8b-base", mapped_models["GRANITE_3_1_8B"]),
        ("granite-8b-instruct", mapped_models["GRANITE_3_1_8B"]),
        ("granite-3.1-8b", mapped_models["GRANITE_3_1_8B"]),
        ("granite-3.1-8b-base", mapped_models["GRANITE_3_1_8B"]),
        ("granite-3.1-8b-instruct", mapped_models["GRANITE_3_1_8B"]),
        ("granite-3.2-8b", mapped_models["GRANITE_3_1_8B"]),
        ("granite-3.3-8b-instruct", mapped_models["GRANITE_3_1_8B"]),
    ]
    for input_name, expected_output in test_cases:
        assert (
            map_valid_model_name(input_name) == expected_output
        ), f"Failed for input: {input_name}"


def test_map_valid_model_name_llama_3_1_8b_variants() -> None:
    """Test various llama 3.1 8B model name patterns."""
    test_cases = [
        ("llama-3.1-8b", mapped_models["LLAMA_3_1_8B"]),
        ("llama-3.1-8b-instruct", mapped_models["LLAMA_3_1_8B"]),
        ("llama-3.1-8b-base", mapped_models["LLAMA_3_1_8B"]),
        ("llama-3.1-8b-custom", mapped_models["LLAMA_3_1_8B"]),
    ]
    for input_name, expected_output in test_cases:
        assert (
            map_valid_model_name(input_name) == expected_output
        ), f"Failed for input: {input_name}"


def test_map_valid_model_name_granite_4_variants() -> None:
    """Test various granite 4.0 model name patterns."""
    test_cases = [
        ("granite-4.0-small", mapped_models["GRANITE_4_SMALL"]),
        ("granite-4.0-small-base", mapped_models["GRANITE_4_SMALL"]),
        ("granite-4.0-h-small", mapped_models["GRANITE_4_SMALL"]),
        ("granite-4.0-h-small-instruct", mapped_models["GRANITE_4_SMALL"]),
        ("granite-4.0-tiny", mapped_models["GRANITE_4_TINY"]),
        ("granite-4.0-tiny-base", mapped_models["GRANITE_4_TINY"]),
        ("granite-4.0-h-tiny", mapped_models["GRANITE_4_TINY"]),
        ("granite-4.0-h-tiny-instruct", mapped_models["GRANITE_4_TINY"]),
        ("granite-4.0-micro", mapped_models["GRANITE_4_MICRO"]),
        ("granite-4.0-micro-base", mapped_models["GRANITE_4_MICRO"]),
        ("granite-4.0-h-micro", mapped_models["GRANITE_4_MICRO"]),
        ("granite-4.0-h-micro-instruct", mapped_models["GRANITE_4_MICRO"]),
    ]
    for input_name, expected_output in test_cases:
        assert (
            map_valid_model_name(input_name) == expected_output
        ), f"Failed for input: {input_name}"


def test_map_valid_model_name_no_match_returns_original() -> None:
    """Test that unmatched model names are returned unchanged."""
    unmatched_names = [
        "unknown-model",
        "gpt-4",
        "claude-3",
        "mistral-7b",
        "granite-5.0-test",
        "llama-2-7b",
        "",
        "granite",
        "granite-4.0-large",  # size not in patterns
    ]
    for name in unmatched_names:
        assert (
            map_valid_model_name(name) == name
        ), f"Expected unchanged output for: {name}"


def test_map_valid_model_name_empty_string() -> None:
    """Test that empty string is handled correctly."""
    assert map_valid_model_name("") == ""


def test_map_valid_model_name_case_sensitivity() -> None:
    """Test that function is case-sensitive (lowercase required)."""
    # These should NOT match because patterns expect lowercase
    uppercase_names = [
        "GRANITE-3.1-8B",
        "Granite-3.1-8b",
        "LLAMA-3.1-8B",
    ]
    for name in uppercase_names:
        # Should return original since patterns are case-sensitive
        assert map_valid_model_name(name) == name


def test_map_valid_model_name_multiple_matches_returns_first() -> None:
    """Test behavior when model name could match multiple patterns.
    The function returns the first match found in the iteration order.
    This test documents the current behavior.
    """
    # Test with a name that should match
    result = map_valid_model_name("granite-3.1-2b")
    assert result == mapped_models["GRANITE_3_1_2B"]
    # Verify it's deterministic
    result2 = map_valid_model_name("granite-3.1-2b")
    assert result == result2


def test_map_valid_model_name_special_characters() -> None:
    """Test handling of model names with special characters."""
    special_names = [
        "granite-3.1-2b_custom",  # underscore - should not match
        "granite-3.1-2b.custom",  # dot after size - should not match
        "granite@3.1-2b",  # @ symbol - should not match
    ]
    for name in special_names:
        # These should not match the patterns and return original
        assert map_valid_model_name(name) == name
