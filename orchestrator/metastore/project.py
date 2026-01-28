# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path
from typing import Annotated

import pydantic
import typer

import orchestrator.utilities.logging
from orchestrator.utilities.location import (
    SQLiteStoreConfiguration,
    SQLStoreConfiguration,
    db_scheme_discriminator,
)

FORMAT = orchestrator.utilities.logging.FORMAT
LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL, format=FORMAT)
moduleLog = logging.getLogger("config")


#
# Pydantic Based Schema
# Work-in-progres: Eventually all elements of above YAML will be expressed instead
# via pydantic classes, and we will load and validate via pydantic
# This means no-class consumes configuration as a YAML/dict.
#


class ProjectContext(pydantic.BaseModel):
    """
    Provides information for storing/retrieving discovery space information from a project data-store.

    Note: The project name determines the names of the timeseries and metadata-store databases and users following
    orchestrator rules.

    If None is passed for project name (its default), the name is deduced if possible from the timeseries and
    metadata-store databases and users following orchestrator rules.

    Notes:
    - The project, user and sslVerify fields are used to set the related fields in metadataStore
      This overwrites any value given for those fields in metadataStore either explicitly or by default
    - If active is not explicitly set for metadataStore it will default to True
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    project: Annotated[
        str,
        pydantic.Field(
            description="The name of the project for this context",
        ),
    ] = "local"

    metadataStore: Annotated[
        Annotated[SQLiteStoreConfiguration, pydantic.Tag("sqlite")]
        | Annotated[SQLStoreConfiguration, pydantic.Tag("mysql")],
        pydantic.Field(description="The SQL backend to use to store data"),
        pydantic.Discriminator(db_scheme_discriminator),
    ] = SQLiteStoreConfiguration(
        path=str(Path(typer.get_app_dir("ado")) / Path("databases/local.db")),
        database="local",
    )

    @pydantic.model_validator(mode="after")
    def ensure_project_and_db_name_match(self) -> "ProjectContext":
        if self.metadataStore.scheme == "sqlite":
            if not self.metadataStore.path.endswith(f"{self.project}.db"):
                raise ValueError(f"The SQLite DB must be called {self.project}.db")
        else:
            if self.project != self.metadataStore.database:
                raise ValueError("Project and database names must match")

        return self
