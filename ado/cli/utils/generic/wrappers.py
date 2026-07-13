# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
import typing

import typer
from rich.status import Status

from ado.cli.utils.output.prints import (
    ADO_SPINNER_CONNECTING_TO_DB,
    ERROR,
    console_print,
)
from ado.metastore.project import ProjectContext

if typing.TYPE_CHECKING:
    from ado.metastore.sqlstore import SQLResourceStore


def get_sql_store(project_context: ProjectContext) -> "SQLResourceStore":
    from sqlalchemy.exc import OperationalError

    from ado.metastore.sqlstore import SQLStore

    with Status(ADO_SPINNER_CONNECTING_TO_DB) as status:
        try:
            # SQLStore.__new__ returns SQLResourceStore
            return SQLStore(project_context=project_context)  # type: ignore[abstract]
        except OperationalError as e:
            status.stop()
            console_print(
                f"{ERROR}Unable to instantiate the SQLStore:\n\n{e}", stderr=True
            )
            raise typer.Exit(1) from e
