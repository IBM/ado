# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import enum
from typing import Optional

import click
from click import Context, Parameter

from orchestrator.cli.utils.input.parsers import resource_shorthands_to_full_names


class GenericChoiceType(click.Choice):
    def __init__(self, enum_type: enum.EnumMeta, case_sensitive: bool = True) -> None:
        choices = [value.value for value in enum_type]
        super().__init__(choices, case_sensitive)
        self.choices = choices
        self.name = enum_type


class HiddenShorthandChoice(GenericChoiceType):
    def convert(
        self, value: str, param: Optional["Parameter"], ctx: Optional["Context"]
    ) -> str:
        value = resource_shorthands_to_full_names(value)

        if value not in self.choices:
            ctx.fail(
                f"Invalid value for {param.human_readable_name}: '{value}' is not one of {self.choices}"
            )

        return value


class HiddenPluralChoice(HiddenShorthandChoice):

    def convert(
        self, value: str, param: Optional["Parameter"], ctx: Optional["Context"]
    ) -> str:
        value = value.removesuffix("s")
        return super().convert(value=value, param=param, ctx=ctx)
