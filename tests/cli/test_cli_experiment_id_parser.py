# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import pytest

from ado.cli.utils.resources.experiments import parse_cli_experiment_id


def test_bare_name() -> None:
    """Bare experiment name: no actuator prefix, no version."""
    assert parse_cli_experiment_id("experiment") == (None, "experiment", None)


def test_bare_name_with_semver() -> None:
    """Bare name with SemVer version suffix."""
    assert parse_cli_experiment_id("experiment@1.0.0") == (
        None,
        "experiment",
        "1.0.0",
    )


def test_bare_name_with_major_version() -> None:
    """Bare name with @v<N> shorthand is normalised to <N>.0.0."""
    assert parse_cli_experiment_id("experiment@v1") == (None, "experiment", "1.0.0")


def test_bare_name_with_major_version_zero() -> None:
    """@v0 is a valid major-version shorthand."""
    assert parse_cli_experiment_id("experiment@v0") == (None, "experiment", "0.0.0")


def test_qualified_name() -> None:
    """Fully-qualified name: actuator prefix, no version."""
    assert parse_cli_experiment_id("actuator.experiment") == (
        "actuator",
        "experiment",
        None,
    )


def test_qualified_name_with_semver() -> None:
    """Fully-qualified name with SemVer version suffix."""
    assert parse_cli_experiment_id("actuator.experiment@1.0.0") == (
        "actuator",
        "experiment",
        "1.0.0",
    )


def test_qualified_name_with_major_version() -> None:
    """Fully-qualified name with @v<N> shorthand is normalised to <N>.0.0."""
    assert parse_cli_experiment_id("actuator.experiment@v2") == (
        "actuator",
        "experiment",
        "2.0.0",
    )


def test_hyphenated_experiment_name() -> None:
    """Hyphens in the experiment name are preserved as-is."""
    assert parse_cli_experiment_id("actuator.my-experiment") == (
        "actuator",
        "my-experiment",
        None,
    )


def test_hyphenated_experiment_name_with_version() -> None:
    """Hyphens in the experiment name are preserved when a version is also present."""
    assert parse_cli_experiment_id("actuator.my-experiment@1.2.3") == (
        "actuator",
        "my-experiment",
        "1.2.3",
    )


def test_at_before_dot_treated_as_bare_name() -> None:
    """When '@' appears before the first '.', no actuator prefix is extracted."""
    # e.g. "experiment@1.0.0" — the dot is inside the version, not a separator
    actuator_id, experiment_id, version = parse_cli_experiment_id("experiment@1.0.0")
    assert actuator_id is None
    assert experiment_id == "experiment"
    assert version == "1.0.0"


def test_invalid_version_suffix_raises() -> None:
    """A '@' suffix that is neither SemVer nor @v<N> raises ValueError."""
    with pytest.raises(ValueError, match="Invalid version suffix"):
        parse_cli_experiment_id("actuator.experiment@bad")


def test_invalid_version_suffix_bare_raises() -> None:
    """Same for a bare name with an invalid '@' suffix."""
    with pytest.raises(ValueError, match="Invalid version suffix"):
        parse_cli_experiment_id("experiment@not-a-version")
