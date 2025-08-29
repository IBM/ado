# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import sqlalchemy.exc
import typer
from rich.status import Status

from orchestrator.cli.utils.output.prints import (
    ADO_SPINNER_CONNECTING_TO_DB,
    ERROR,
    console_print,
)
from orchestrator.metastore.project import ProjectContext
from orchestrator.metastore.sqlstore import SQLStore


def get_sql_store(project_context: ProjectContext) -> SQLStore:
    with Status(ADO_SPINNER_CONNECTING_TO_DB) as status:
        try:
            return SQLStore(project_context=project_context)
        except sqlalchemy.exc.OperationalError as e:
            status.stop()
            console_print(
                f"{ERROR}Unable to instantiate the SQLStore:\n\n{e}", stderr=True
            )
            raise typer.Exit(1) from e
