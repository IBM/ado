# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
from typing import Any, Optional

import click
from click import Context, Parameter


class GenericChoiceType(click.Choice):
    def __init__(self, enum_type: enum.EnumMeta, case_sensitive: bool = True):
        choices = [value.value for value in enum_type]
        super().__init__(choices, case_sensitive)
        self.choices = choices
        self.name = enum_type


class HiddenPluralChoice(GenericChoiceType):

    def convert(
        self, value: Any, param: Optional["Parameter"], ctx: Optional["Context"]
    ) -> Any:
        value = value.removesuffix("s")

        if value not in self.choices:
            ctx.fail(
                f"Invalid value for {param.human_readable_name}: '{value}' is not one of {self.choices}"
            )

        return value


class HiddenSingularChoice(GenericChoiceType):

    def convert(
        self, value: Any, param: Optional["Parameter"], ctx: Optional["Context"]
    ) -> Any:
        if not value.endswith("s"):
            value += "s"

        if value not in self.choices:
            ctx.fail(
                f"Invalid value for {param.human_readable_name}: '{value}' is not one of {self.choices}"
            )

        return value
