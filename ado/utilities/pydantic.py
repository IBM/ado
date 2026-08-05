# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import json
import re
import typing
from typing import Annotated, Any, TypeVar

import pydantic
from pydantic import AfterValidator, BeforeValidator
from pydantic_core import PydanticUseDefault


def pydantic_aware_json_serializer(value: Any) -> str:  # noqa: ANN401
    """Serialise *value* to a JSON string, using Pydantic when available.

    For Pydantic models (including ``RootModel``), delegates to
    ``model_dump(mode="json")``, which correctly handles types the standard
    library ``json`` module cannot — ``datetime``, ``UUID``, ``Enum``, nested
    models, etc. — then encodes the result to a JSON string.  For all other
    values, falls back to ``json.dumps``.

    Intended for use as the ``json_serializer`` argument to
    ``sqlalchemy.create_engine``, which requires a ``Callable[[Any], str]``.
    This ensures every ``JSON`` column in the database uses Pydantic-aware
    serialisation without requiring call sites to pre-convert values.

    Args:
        value: Any Python value to serialise.

    Returns:
        A JSON string.
    """
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump_json()
    return json.dumps(value)


def default_if_none(value: typing.Any) -> typing.Any:  # noqa: ANN401
    if value is None:
        raise PydanticUseDefault
    return value


T = TypeVar("T")
Defaultable = Annotated[T, BeforeValidator(default_if_none)]


def model_dict_representation_with_field_exclusions_for_custom_model_serializer(
    model: pydantic.BaseModel, info: pydantic.SerializationInfo
) -> dict[str, typing.Any]:

    dict_representation = dict(model)

    # We need to enforce the behaviour for field exclusions
    if info.exclude:
        field_names_to_exclude = (
            set(info.exclude.keys()) if isinstance(info.exclude, dict) else info
        )
        for field_name in field_names_to_exclude:
            dict_representation.pop(field_name, None)

    for field_name, field_info in model.__class__.model_fields.items():
        if field_name not in dict_representation:
            continue

        # Enforce exclude_unset
        if (  # noqa: SIM114
            info.exclude_unset and field_name not in model.model_fields_set
        ):
            del dict_representation[field_name]

        # Enforce exclude_none
        elif (  # noqa: SIM114
            info.exclude_none and dict_representation[field_name] is None
        ):
            del dict_representation[field_name]

        # Enforce exclude_defaults
        elif (
            info.exclude_defaults
            and dict_representation[field_name] == field_info.default
        ):
            del dict_representation[field_name]

    return dict_representation


rfc_1123_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"
rfc_1123_regex = re.compile(rfc_1123_pattern)


def validate_rfc_1123(value: str | None) -> str | None:

    if value is None:
        return value

    if len(value) == 0 or len(value) >= 64:
        raise ValueError("The string must be between 1 and 63 characters")

    if not rfc_1123_regex.match(value):
        raise ValueError(
            f"The string does not match RFC1123. Regex: {rfc_1123_pattern}"
        )

    return value


ignore_plugin_validation_context: dict[str, bool] = {"ignore_plugin_validation": True}

do_not_populate_ado_provenance_context: dict[str, bool] = {
    "populate_ado_provenance": False
}


def merge_validation_context(
    *contexts: dict[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
    """Merge optional pydantic validation context dictionaries.

    Args:
        *contexts: Context dicts to merge; ``None`` entries are skipped.

    Returns:
        Merged context, or ``None`` when no contexts were supplied.
    """
    merged: dict[str, typing.Any] = {}
    for context in contexts:
        if context:
            merged.update(context)
    return merged or None


def ignore_plugin_validation(info: pydantic.ValidationInfo) -> bool:
    """Return True when plugin registry validation should be skipped.

    Args:
        info: Pydantic validation info for the current validation step.

    Returns:
        True if the validation context requests skipping plugin validation.
    """
    return bool(info.context and info.context.get("ignore_plugin_validation"))


def validate_pep440_version(value: str) -> str:
    """Validate that *value* is a valid PEP 440 version string.

    Args:
        value: The version string to validate.

    Returns:
        The original version string unchanged.

    Raises:
        ValueError: If *value* is not a valid PEP 440 version string.
    """
    from packaging.version import InvalidVersion, Version

    try:
        Version(value)
    except InvalidVersion as exc:
        raise ValueError(
            f"Version {value!r} is not a valid PEP 440 version string: {exc}"
        ) from exc
    return value


Pep440VersionStr = Annotated[str, AfterValidator(validate_pep440_version)]


_STRICT_SEMVER_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def validate_strict_semver(value: str) -> str:
    """Validate that *value* is a strict MAJOR.MINOR.PATCH SemVer string.

    Pre-release identifiers and build metadata are not accepted.

    Args:
        value: The version string to validate.

    Returns:
        The original version string unchanged.

    Raises:
        ValueError: If *value* is not a valid strict SemVer string.
    """
    if not _STRICT_SEMVER_PATTERN.match(value):
        raise ValueError(
            f"Version {value!r} is not a valid strict SemVer string. "
            "Expected MAJOR.MINOR.PATCH where each component is a non-negative integer "
            "(e.g. '1.0.0', '2.3.1'). Pre-release identifiers and build metadata are not accepted."
        )
    return value


StrictSemVerStr = Annotated[str, AfterValidator(validate_strict_semver)]


def semver_major(version: StrictSemVerStr) -> int:
    """Extract the MAJOR component from a strict SemVer string.

    Args:
        version: A strict SemVer string (e.g. ``"1.2.3"``).

    Returns:
        The integer major version component.

    Raises:
        ValueError: If *version* is not a valid strict SemVer string.
    """
    validated = validate_strict_semver(version)
    return int(validated.split(".", maxsplit=1)[0])
