# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import re
import typing
from typing import Annotated, TypeVar

import pydantic
from pydantic import AfterValidator, BeforeValidator
from pydantic_core import PydanticUseDefault


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
