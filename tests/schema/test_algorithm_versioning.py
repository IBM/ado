# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
"""Tests for algorithm versioning: StrictSemVerStr, semver_major, Experiment identifiers,
ExperimentReference identifiers, ExperimentCatalog behaviour, and experimentForReference with resolve.
"""

import warnings

import pytest

from ado.modules.actuators.catalog import (
    ExperimentCatalog,
)
from ado.modules.actuators.errors import (
    AmbiguousExperimentIdentifierError,
    ExperimentVersionMismatchError,
    MissingActuatorConfigurationForCatalogError,
    UnexpectedCatalogRetrievalError,
    UnknownActuatorError,
    UnknownExperimentError,
)
from ado.modules.actuators.registry import (
    ActuatorRegistry,
)
from ado.schema.domain import PropertyDomain, VariableTypeEnum
from ado.schema.experiment import Experiment, ParameterizedExperiment
from ado.schema.property import (
    AbstractPropertyDescriptor,
    ConstitutiveProperty,
    ConstitutivePropertyDescriptor,
)
from ado.schema.property_value import ConstitutivePropertyValue
from ado.schema.reference import ExperimentReference
from ado.utilities.pydantic import StrictSemVerStr, semver_major

# ─── StrictSemVerStr validation ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "version",
    ["0.0.0", "1.0.0", "2.3.1", "10.20.30", "0.1.0"],
)
def test_strict_semver_valid(version: str) -> None:
    """Valid MAJOR.MINOR.PATCH strings should be accepted."""
    from pydantic import TypeAdapter

    ta: TypeAdapter[StrictSemVerStr] = TypeAdapter(StrictSemVerStr)
    assert ta.validate_python(version) == version


@pytest.mark.parametrize(
    "version",
    [
        "1",
        "1.0",
        "1.0.0.0",
        "1.0.0-alpha",
        "1.0.0+build",
        "1.0.0-alpha.1",
        "v1.0.0",
        "01.0.0",  # leading zero in major
        "",
        "latest",
    ],
)
def test_strict_semver_invalid(version: str) -> None:
    """Strings that are not strict MAJOR.MINOR.PATCH should be rejected."""
    from pydantic import TypeAdapter, ValidationError

    ta: TypeAdapter[StrictSemVerStr] = TypeAdapter(StrictSemVerStr)
    with pytest.raises(ValidationError):
        ta.validate_python(version)


@pytest.mark.parametrize(
    ("version", "expected_major"),
    [
        ("0.0.0", 0),
        ("1.0.0", 1),
        ("2.3.1", 2),
        ("10.20.30", 10),
    ],
)
def test_semver_major(version: str, expected_major: int) -> None:
    """semver_major extracts the MAJOR component correctly."""
    assert semver_major(version) == expected_major


@pytest.mark.parametrize(
    "version",
    [
        "1",
        "1.0",
        "1.0.0.0",
        "1.0.0-alpha",
        "1.0.0+build",
        "1.0.0-alpha.1",
        "v1.0.0",
        "01.0.0",
        "",
        "latest",
    ],
)
def test_semver_major_rejects_invalid_version(version: str) -> None:
    """semver_major validates input and raises a contextual ValueError."""
    with pytest.raises(ValueError, match="not a valid strict SemVer string"):
        semver_major(version)


# ─── Experiment identifiers ───────────────────────────────────────────────────


def _make_experiment(identifier: str, version: str | None = None) -> Experiment:
    """Helper to create a minimal Experiment with optional version."""
    return Experiment(
        actuatorIdentifier="test_actuator",
        identifier=identifier,
        targetProperties=[AbstractPropertyDescriptor(identifier="output")],
        version=version,
    )


def _catalog_with_versioned_experiment(
    identifier: str = "solve_mip", version: str = "1.0.0"
) -> ExperimentCatalog:
    exp = _make_experiment(identifier, version=version)
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(exp)
    return catalog


def _catalog_with_multiple_major_versions(
    identifier: str = "solve_mip",
) -> ExperimentCatalog:
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(_make_experiment(identifier, version="1.0.0"))
        catalog.addExperiment(_make_experiment(identifier, version="2.0.0"))
    return catalog


def test_experiment_major_version_identifier_with_version() -> None:
    """major_version_identifier includes @vMAJOR when version is set."""
    exp = _make_experiment("solve_mip", version="1.2.3")
    assert exp.major_version_identifier == "solve_mip@v1"


def test_experiment_major_version_identifier_without_version() -> None:
    """major_version_identifier falls back to base identifier when version is None."""
    exp = _make_experiment("solve_mip")
    assert exp.major_version_identifier == "solve_mip"


def test_experiment_fully_qualified_identifier_with_version() -> None:
    """fully_qualified_identifier includes @MAJOR.MINOR.PATCH when version is set."""
    exp = _make_experiment("solve_mip", version="1.2.3")
    assert exp.fully_qualified_identifier == "solve_mip@1.2.3"


def test_experiment_fully_qualified_identifier_without_version() -> None:
    """fully_qualified_identifier falls back to base identifier when version is None."""
    exp = _make_experiment("solve_mip")
    assert exp.fully_qualified_identifier == "solve_mip"


def test_experiment_eq_same_major_version() -> None:
    """Experiments with the same base name and same major version are equal."""
    exp_v100 = _make_experiment("solve_mip", version="1.0.0")
    exp_v120 = _make_experiment("solve_mip", version="1.2.0")
    assert exp_v100 == exp_v120


def test_experiment_eq_different_major_version() -> None:
    """Experiments with different major versions are NOT equal."""
    exp_v1 = _make_experiment("solve_mip", version="1.0.0")
    exp_v2 = _make_experiment("solve_mip", version="2.0.0")
    assert exp_v1 != exp_v2


def test_experiment_eq_unversioned_legacy() -> None:
    """Two unversioned experiments with the same identifier and actuator are equal."""
    exp_a = _make_experiment("solve_mip")
    exp_b = _make_experiment("solve_mip")
    assert exp_a == exp_b


def test_experiment_eq_non_experiment() -> None:
    """Experiment is not equal to a non-Experiment object."""
    exp = _make_experiment("solve_mip", version="1.0.0")
    assert exp != "solve_mip"


def test_experiment_hash_same_major_version() -> None:
    """Experiments with same base name and same major version have the same hash."""
    exp_v100 = _make_experiment("solve_mip", version="1.0.0")
    exp_v120 = _make_experiment("solve_mip", version="1.2.0")
    assert hash(exp_v100) == hash(exp_v120)


def test_experiment_hash_different_major_version() -> None:
    """Experiments with different major versions have different hashes."""
    exp_v1 = _make_experiment("solve_mip", version="1.0.0")
    exp_v2 = _make_experiment("solve_mip", version="2.0.0")
    assert hash(exp_v1) != hash(exp_v2)


def test_experiment_reference_carries_version() -> None:
    """Experiment.reference includes experimentVersion from the experiment's version."""
    exp = _make_experiment("solve_mip", version="1.2.3")
    ref = exp.reference
    assert ref.experimentVersion == "1.2.3"


# ─── ParameterizedExperiment.major_version_parameterized_identifier ──────────────────────────


def _make_parameterizable_experiment(
    identifier: str, version: str | None = None
) -> Experiment:
    param_prop = ConstitutiveProperty(
        identifier="timeout",
        propertyDomain=PropertyDomain(
            variableType=VariableTypeEnum.DISCRETE_VARIABLE_TYPE, interval=1
        ),
    )
    return Experiment(
        actuatorIdentifier="test_actuator",
        identifier=identifier,
        targetProperties=[AbstractPropertyDescriptor(identifier="output")],
        optionalProperties=(param_prop,),
        defaultParameterization=(
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier="timeout"), value=60
            ),
        ),
        version=version,
    )


def test_parameterized_identifier_uses_major_version_prefix() -> None:
    """ParameterizedExperiment.major_version_parameterized_identifier encodes major version."""
    base = _make_parameterizable_experiment("solve_mip", version="1.0.0")
    parameterization = [
        ConstitutivePropertyValue(
            property=ConstitutivePropertyDescriptor(identifier="timeout"), value=120
        )
    ]
    pe = ParameterizedExperiment(parameterization=parameterization, **base.model_dump())
    assert pe.major_version_parameterized_identifier == "solve_mip@v1-timeout.120"


def test_parameterized_identifier_unversioned_no_at_sign() -> None:
    """ParameterizedExperiment.major_version_parameterized_identifier has no @version when unversioned."""
    base = _make_parameterizable_experiment("solve_mip")
    parameterization = [
        ConstitutivePropertyValue(
            property=ConstitutivePropertyDescriptor(identifier="timeout"), value=120
        )
    ]
    pe = ParameterizedExperiment(parameterization=parameterization, **base.model_dump())
    assert pe.major_version_parameterized_identifier == "solve_mip-timeout.120"


# ─── ExperimentReference identifiers ─────────────────────────────────────────


def test_reference_major_version_identifier_with_version() -> None:
    """major_version_experiment_identifier includes @vMAJOR when experimentVersion is set."""
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.3",
    )
    assert ref.major_version_experiment_identifier == "solve_mip@v1"


def test_reference_major_version_identifier_without_version() -> None:
    """major_version_experiment_identifier falls back to experimentIdentifier when unversioned."""
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    assert ref.major_version_experiment_identifier == "solve_mip"


def test_reference_fully_qualified_identifier_with_version() -> None:
    """fully_qualified_experiment_identifier returns base@version."""
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.3",
    )
    assert ref.fully_qualified_experiment_identifier == "solve_mip@1.2.3"


def test_reference_parameterized_experiment_identifier_major_version() -> None:
    """major_version_parameterized_experiment_identifier uses major version form when version is set."""
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.3",
        parameterization=[
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier="timeout"), value=120
            )
        ],
    )
    assert (
        ref.major_version_parameterized_experiment_identifier
        == "solve_mip@v1-timeout.120"
    )


def test_reference_parameterized_experiment_identifier_no_version() -> None:
    """major_version_parameterized_experiment_identifier uses base form when unversioned (backward compat)."""
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        parameterization=[
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier="timeout"), value=120
            )
        ],
    )
    assert (
        ref.major_version_parameterized_experiment_identifier == "solve_mip-timeout.120"
    )


def test_reference_eq_same_major_different_minor() -> None:
    """References with same base, same major, and same params compare equal."""
    ref_v100 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.0.0",
    )
    ref_v120 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.0",
    )
    assert ref_v100 == ref_v120


def test_reference_eq_different_major() -> None:
    """References with different major versions are NOT equal."""
    ref_v1 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.0.0",
    )
    ref_v2 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="2.0.0",
    )
    assert ref_v1 != ref_v2


# ─── experimentIdentifier validation and parsing ─────────────────────────────


def test_experiment_identifier_rejects_at_sign() -> None:
    """experimentIdentifier must not contain '@'; use experimentVersion instead."""
    with pytest.raises(ValueError, match="must not contain '@'"):
        ExperimentReference(
            experimentIdentifier="solve_mip@v1",
            actuatorIdentifier="act",
        )


def test_reference_from_string_rejects_legacy_v1_suffix() -> None:
    """Legacy @v1 string forms are rejected at parse time."""
    with pytest.raises(ValueError, match="Cannot parse version suffix"):
        ExperimentReference.referenceFromString("act.solve_mip@v1")


# ─── Catalog lookup via experimentForReference ───────────────────────────────


def test_catalog_lookup_both_versioned_same_major() -> None:
    """Catalog matches references with the same major version."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.5.0",
    )
    assert catalog.experimentForReference(ref) is not None


def test_catalog_lookup_both_versioned_different_major() -> None:
    """Catalog does not match references with different major versions."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="2.0.0",
    )
    assert catalog.experimentForReference(ref) is None


def test_catalog_lookup_unversioned_vs_versioned() -> None:
    """Unversioned reference does not match a versioned catalog experiment."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    assert catalog.experimentForReference(ref) is None


def test_catalog_lookup_both_unversioned() -> None:
    """Unversioned reference matches an unversioned catalog experiment."""
    exp = _make_experiment("solve_mip")
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(exp)
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    assert catalog.experimentForReference(ref) is not None


def test_catalog_lookup_base_single_versioned_match() -> None:
    """Base matching resolves an unversioned reference to a sole versioned entry."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    result = catalog.experimentForReference(ref, match_on="base")
    assert result is not None
    assert result.version == "1.0.0"


def test_catalog_lookup_base_ambiguous_raises() -> None:
    """Base matching raises when multiple catalog versions share a base identifier."""
    catalog = _catalog_with_multiple_major_versions()
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    with pytest.raises(AmbiguousExperimentIdentifierError, match="ambiguous"):
        catalog.experimentForReference(ref, match_on="base", resolve=False)


def test_catalog_lookup_any_exact_version() -> None:
    """Any matching prefers fully qualified matches."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.0.0",
    )
    result = catalog.experimentForReference(ref, match_on="any")
    assert result is not None
    assert result.version == "1.0.0"


def test_catalog_lookup_any_major_version() -> None:
    """Any matching falls back to major version matching."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.0",
    )
    result = catalog.experimentForReference(ref, match_on="any")
    assert result is not None
    assert result.version == "1.0.0"


def test_catalog_lookup_any_base_identifier() -> None:
    """Any matching falls back to base identifier matching."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    result = catalog.experimentForReference(ref, match_on="any")
    assert result is not None
    assert result.version == "1.0.0"


def test_catalog_lookup_any_ambiguous_raises() -> None:
    """Any matching raises when base identifier matching is ambiguous."""
    catalog = _catalog_with_multiple_major_versions()
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    with pytest.raises(AmbiguousExperimentIdentifierError, match="ambiguous"):
        catalog.experimentForReference(ref, match_on="any", resolve=False)


def test_registry_experiment_for_reference_any(
    global_registry: ActuatorRegistry,
) -> None:
    """Registry delegates any matching to the actuator catalog."""
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(
            _make_experiment("registry_any_lookup_exp", version="1.0.0").model_copy(
                update={"actuatorIdentifier": "mock"}
            )
        )

    ref = ExperimentReference(
        experimentIdentifier="registry_any_lookup_exp",
        actuatorIdentifier="mock",
    )
    result = global_registry.experimentForReference(ref, match_on="any")
    assert result.identifier == "registry_any_lookup_exp"
    assert result.version == "1.0.0"


# ─── ExperimentCatalog ────────────────────────────────────────────────────────


def test_catalog_keys_on_major_version_identifier() -> None:
    """Catalog keys on major_version_identifier — v1.0.0 and v1.2.0 map to the same key."""
    exp = _make_experiment("solve_mip", version="1.2.0")
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(exp)
    assert "solve_mip@v1" in catalog.experiment_major_version_identifiers


def test_catalog_add_same_experiment_idempotent() -> None:
    """Re-adding the exact same experiment to a catalog is idempotent."""
    exp = _make_experiment("solve_mip", version="1.0.0")
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(exp)
        catalog.addExperiment(exp)  # should not raise


def test_catalog_add_different_experiment_same_major_version_id_raises() -> None:
    """Adding two different experiments with the same major version identifier raises ValueError."""
    exp_v100 = _make_experiment("solve_mip", version="1.0.0")
    exp_v120 = Experiment(
        actuatorIdentifier="test_actuator",
        identifier="solve_mip",
        targetProperties=[
            AbstractPropertyDescriptor(identifier="output"),
            AbstractPropertyDescriptor(identifier="extra_output"),
        ],
        version="1.2.0",
    )
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(exp_v100)
        with pytest.raises(ValueError, match="major version identifier"):
            catalog.addExperiment(exp_v120)


def test_catalog_different_major_versions_coexist() -> None:
    """Two experiments with the same base name but different major versions can coexist."""
    exp_v1 = _make_experiment("solve_mip", version="1.0.0")
    exp_v2 = _make_experiment("solve_mip", version="2.0.0")
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(exp_v1)
        catalog.addExperiment(exp_v2)
    assert "solve_mip@v1" in catalog.experiment_major_version_identifiers
    assert "solve_mip@v2" in catalog.experiment_major_version_identifiers


# ─── experimentForReference with resolve=True ─────────────────────────────────


def test_experiment_for_reference_resolve_major_version_mode_same_major() -> None:
    """experimentForReference with resolve=True succeeds when major matches."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.0",
    )
    result = catalog.experimentForReference(ref, resolve=True)
    assert isinstance(result, Experiment)
    assert result.identifier == "solve_mip"


def test_experiment_for_reference_resolve_major_version_mode_different_major_raises() -> (
    None
):
    """experimentForReference with resolve=True raises when major mismatches."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="2.0.0",
    )
    with pytest.raises(UnknownExperimentError):
        catalog.experimentForReference(ref, resolve=True)


def test_experiment_for_reference_fully_qualified_mode_exact_match() -> None:
    """experimentForReference with fully_qualified_version succeeds on exact match."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.0.0",
    )
    result = catalog.experimentForReference(
        ref, match_on="fully_qualified_version", resolve=True
    )
    assert isinstance(result, Experiment)


def test_experiment_for_reference_fully_qualified_mode_minor_mismatch_raises() -> None:
    """experimentForReference with fully_qualified_version raises on minor mismatch."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.0",
    )
    with pytest.raises(ExperimentVersionMismatchError):
        catalog.experimentForReference(
            ref, match_on="fully_qualified_version", resolve=True
        )


def test_experiment_for_reference_fully_qualified_mode_minor_mismatch_returns_none() -> (
    None
):
    """experimentForReference with fully_qualified_version returns None when resolve=False."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.2.0",
    )
    assert (
        catalog.experimentForReference(
            ref, match_on="fully_qualified_version", resolve=False
        )
        is None
    )


def test_experiment_for_reference_resolve_unversioned_raises_for_versioned_catalog() -> (
    None
):
    """Unversioned reference does not resolve against a versioned catalog experiment."""
    catalog = _catalog_with_versioned_experiment(version="1.0.0")
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
    )
    with pytest.raises(UnknownExperimentError):
        catalog.experimentForReference(ref, resolve=True)


def test_experiment_for_reference_resolve_with_parameterization_returns_parameterized() -> (
    None
):
    """experimentForReference with resolve=True returns ParameterizedExperiment."""
    base = _make_parameterizable_experiment("solve_mip", version="1.0.0")
    catalog = ExperimentCatalog(catalogIdentifier="test")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(base)
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="test_actuator",
        experimentVersion="1.0.0",
        parameterization=[
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier="timeout"), value=120
            )
        ],
    )
    result = catalog.experimentForReference(ref, resolve=True)
    assert isinstance(result, ParameterizedExperiment)
    assert result.major_version_parameterized_identifier == "solve_mip@v1-timeout.120"


# ─── Memoisation cache miss / hit ─────────────────────────────────────────────


def test_memoisation_minor_bump_is_cache_hit() -> None:
    """v1.0.0 and v1.2.0 references with same params produce the same parameterized identifier."""
    ref_v100 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="act",
        experimentVersion="1.0.0",
    )
    ref_v120 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="act",
        experimentVersion="1.2.0",
    )
    assert (
        ref_v100.major_version_parameterized_experiment_identifier
        == ref_v120.major_version_parameterized_experiment_identifier
    )


def test_memoisation_major_bump_is_cache_miss() -> None:
    """v1.x and v2.x references produce DIFFERENT parameterized identifiers."""
    ref_v1 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="act",
        experimentVersion="1.0.0",
    )
    ref_v2 = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="act",
        experimentVersion="2.0.0",
    )
    assert (
        ref_v1.major_version_parameterized_experiment_identifier
        != ref_v2.major_version_parameterized_experiment_identifier
    )


# ─── referenceFromString and __str__ round-trip ────────────────────────────────


def test_reference_str_uses_fully_qualified_form() -> None:
    """__str__ uses FQ version, not major version @vMAJOR form."""
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="act",
        experimentVersion="1.2.3",
    )
    assert str(ref) == "act.solve_mip@1.2.3"


def test_reference_str_fq_with_parameterization() -> None:
    """__str__ includes FQ version and parameterization."""
    ref = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="act",
        experimentVersion="1.0.0",
        parameterization=[
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier="timeout"), value=120
            )
        ],
    )
    assert str(ref) == "act.solve_mip@1.0.0-timeout.120"


def test_reference_from_string_parses_fq_version() -> None:
    """referenceFromString parses @MAJOR.MINOR.PATCH into experimentVersion."""
    parsed = ExperimentReference.referenceFromString("act.solve_mip@1.2.3")
    assert parsed.actuatorIdentifier == "act"
    assert parsed.experimentIdentifier == "solve_mip"
    assert parsed.experimentVersion == "1.2.3"
    assert parsed.parameterization is None


def test_reference_from_string_parses_fq_version_and_parameterization() -> None:
    """referenceFromString parses FQ version and parameterization suffix."""
    parsed = ExperimentReference.referenceFromString("act.solve_mip@1.0.0-timeout.120")
    assert parsed.experimentIdentifier == "solve_mip"
    assert parsed.experimentVersion == "1.0.0"
    assert parsed.parameterization is not None
    assert len(parsed.parameterization) == 1
    assert parsed.parameterization[0].property.identifier == "timeout"
    assert parsed.parameterization[0].value == 120


def test_reference_from_string_round_trip_with_fq_form() -> None:
    """referenceFromString(str(ref)) round-trips a versioned reference."""
    original = ExperimentReference(
        experimentIdentifier="solve_mip",
        actuatorIdentifier="act",
        experimentVersion="1.0.0",
        parameterization=[
            ConstitutivePropertyValue(
                property=ConstitutivePropertyDescriptor(identifier="timeout"), value=120
            )
        ],
    )
    round_tripped = ExperimentReference.referenceFromString(str(original))
    assert round_tripped == original


def test_reference_from_string_without_version_sets_none() -> None:
    """Strings without a version suffix produce experimentVersion=None."""
    parsed = ExperimentReference.referenceFromString("act.solve_mip")
    assert parsed.experimentIdentifier == "solve_mip"
    assert parsed.experimentVersion is None


def test_registry_unknown_experiment_error_lists_available_versions(
    global_registry: ActuatorRegistry,
) -> None:
    """Registry UnknownExperimentError hints at available versions when omitted."""
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    versioned_exp = Experiment(
        actuatorIdentifier="mock",
        identifier="version_hint_exp",
        targetProperties=[AbstractPropertyDescriptor(identifier="output")],
        version="1.0.0",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(versioned_exp)
    ref = ExperimentReference(
        experimentIdentifier="version_hint_exp",
        actuatorIdentifier="mock",
    )
    with pytest.raises(
        UnknownExperimentError, match=r"Available versions in catalog: 1\.0\.0"
    ):
        global_registry.experimentForReference(ref)


def test_registry_experiment_for_reference_unknown_actuator_raises(
    global_registry: ActuatorRegistry,
) -> None:
    """Unknown actuator must raise UnknownActuatorError, not UnknownExperimentError."""
    ref = ExperimentReference(
        experimentIdentifier="some_experiment",
        actuatorIdentifier="nonexistent_actuator",
    )
    with pytest.raises(UnknownActuatorError, match="nonexistent_actuator"):
        global_registry.experimentForReference(ref, resolve=False)


def test_registry_experiment_for_reference_unknown_experiment_raises(
    global_registry: ActuatorRegistry,
) -> None:
    """Known actuator with missing experiment must raise UnknownExperimentError."""
    ref = ExperimentReference(
        experimentIdentifier="nonexistent_experiment",
        actuatorIdentifier="mock",
    )
    with pytest.raises(UnknownExperimentError, match="actuator was found"):
        global_registry.experimentForReference(ref, resolve=False)


def test_registry_experiment_for_reference_miss_without_actuator_catalog(
    global_registry: ActuatorRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When only supplementary catalogs are searched, miss text must not claim actuator found."""
    from unittest.mock import Mock

    ref = ExperimentReference(
        experimentIdentifier="missing_experiment",
        actuatorIdentifier="mock",
    )
    supplementary = ExperimentCatalog(catalogIdentifier="supplementary", experiments={})

    def raise_missing_configuration(*_args: object, **_kwargs: object) -> None:
        raise MissingActuatorConfigurationForCatalogError(
            "Actuator mock requires configuration information to create catalog."
        )

    monkeypatch.setattr(
        global_registry,
        "catalogForActuatorIdentifier",
        Mock(side_effect=raise_missing_configuration),
    )

    with pytest.raises(UnknownExperimentError, match="No match for"):
        global_registry.experimentForReference(
            ref, additionalCatalogs=[supplementary], resolve=False
        )


def test_registry_experiment_for_reference_no_catalogs_raises(
    global_registry: ActuatorRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no catalogs can be searched, raise UnexpectedCatalogRetrievalError."""
    from unittest.mock import Mock

    ref = ExperimentReference(
        experimentIdentifier="some_experiment",
        actuatorIdentifier="mock",
    )

    def raise_missing_configuration(*_args: object, **_kwargs: object) -> None:
        raise MissingActuatorConfigurationForCatalogError(
            "Actuator mock requires configuration information to create catalog."
        )

    monkeypatch.setattr(
        global_registry,
        "catalogForActuatorIdentifier",
        Mock(side_effect=raise_missing_configuration),
    )

    with pytest.raises(UnexpectedCatalogRetrievalError, match="No catalogs available"):
        global_registry.experimentForReference(ref, resolve=False)


# ─── FQ version pin at measurement space creation ─────────────────────────────


def _add_versioned_experiment_to_mock_catalog(
    global_registry: ActuatorRegistry,
    identifier: str = "fq_pin_exp",
    version: str = "1.0.0",
) -> None:
    catalog = global_registry.catalogForActuatorIdentifier("mock")
    experiment = Experiment(
        actuatorIdentifier="mock",
        identifier=identifier,
        targetProperties=[AbstractPropertyDescriptor(identifier="output")],
        version=version,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        catalog.addExperiment(experiment)


def test_resolve_experiment_for_measurement_space_fq_exact_match(
    global_registry: ActuatorRegistry,
) -> None:
    """Versioned YAML ref matching catalog version resolves successfully."""
    _add_versioned_experiment_to_mock_catalog(global_registry)
    ref = ExperimentReference(
        experimentIdentifier="fq_pin_exp",
        actuatorIdentifier="mock",
        experimentVersion="1.0.0",
    )
    result = global_registry.experimentForReference(
        ref, match_on="fully_qualified_version", resolve=True
    )
    assert result.identifier == "fq_pin_exp"
    assert result.version == "1.0.0"


def test_resolve_experiment_for_measurement_space_fq_mismatch(
    global_registry: ActuatorRegistry,
) -> None:
    """Versioned YAML ref with wrong patch/minor raises AlgorithmVersionMismatchError."""
    _add_versioned_experiment_to_mock_catalog(global_registry)
    ref = ExperimentReference(
        experimentIdentifier="fq_pin_exp",
        actuatorIdentifier="mock",
        experimentVersion="1.1.0",
    )
    with pytest.raises(ExperimentVersionMismatchError):
        global_registry.experimentForReference(
            ref, match_on="fully_qualified_version", resolve=True
        )


def test_measurement_space_from_selection_fq_mismatch(
    global_registry: ActuatorRegistry,
) -> None:
    """measurementSpaceFromSelection enforces exact version when experimentVersion is set."""
    from ado.schema.measurementspace import MeasurementSpace

    _add_versioned_experiment_to_mock_catalog(global_registry)
    ref = ExperimentReference(
        experimentIdentifier="fq_pin_exp",
        actuatorIdentifier="mock",
        experimentVersion="1.1.0",
    )
    with pytest.raises(ExperimentVersionMismatchError):
        MeasurementSpace.measurementSpaceFromSelection(selectedExperiments=[ref])


def test_measurement_space_from_selection_fq_exact_match(
    global_registry: ActuatorRegistry,
) -> None:
    """measurementSpaceFromSelection stores catalog version when YAML pin matches."""
    from ado.schema.measurementspace import MeasurementSpace

    _add_versioned_experiment_to_mock_catalog(global_registry)
    ref = ExperimentReference(
        experimentIdentifier="fq_pin_exp",
        actuatorIdentifier="mock",
        experimentVersion="1.0.0",
    )
    space = MeasurementSpace.measurementSpaceFromSelection(selectedExperiments=[ref])
    assert space.experiments[0].version == "1.0.0"
