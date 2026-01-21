# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import pydantic
import typer
import yaml

import orchestrator.metastore.project
from orchestrator.cli.utils.output.prints import (
    ERROR,
    HINT,
    INFO,
    console_print,
    cyan,
)
from orchestrator.core import CoreResourceKinds
from orchestrator.metastore.project import ProjectContext
from orchestrator.utilities.location import SQLiteStoreConfiguration

ADO_APP_NAME = "ado"
ADO_CONTEXTS_DIR_NAME = "contexts"
ADO_LOCAL_DBS_DIR_NAME = "databases"

ADO_DEFAULT_LOCAL_CONTEXT_NAME = "local"
ADO_CONFIG_FILE_NAME = "ado_cli_config.json"


class AdoConfiguration(pydantic.BaseModel):
    _app_dir: Path = Path(typer.get_app_dir(ADO_APP_NAME))
    _project_context: ProjectContext | None = None
    active_context: str | None = None
    latest_resource_ids: dict[CoreResourceKinds, str] = {}

    @classmethod
    def load(
        cls,
        from_project_context: Path | None = None,
        do_not_fail_on_available_contexts: bool = False,
        _override_config_dir: Path | None = None,
    ) -> "AdoConfiguration":
        """
        Loads the AdoConfiguration from the default file or returns
        an instance with default values.
        Optionally allows creating an AdoConfiguration instance from
        a ProjectContext file.
        """
        # We need to instantiate an AdoConfiguration, as we could already
        # have a valid one on disk which contains the active context.
        ado_config = AdoConfiguration()

        # For testing, we might want to override the config directory
        if _override_config_dir:
            if not (_override_config_dir.exists() and _override_config_dir.is_dir()):
                console_print(
                    f"{ERROR}The provided config directory override does not exist or is not a directory",
                    stderr=True,
                )
                raise typer.Abort
            ado_config._app_dir = _override_config_dir

        if ado_config._app_dir.is_dir() and ado_config.config_file.is_file():
            ado_config = cls.model_validate_json(ado_config.config_file.read_text())

            # AP 04/06/2025:
            # As we overwrite the ado_config instance on which we had set the
            # config dir, we need to do so again.
            if _override_config_dir:
                ado_config._app_dir = _override_config_dir

        # Now that we have the AdoConfiguration instance, we need to set
        # the project_context (and possibly the active_context).
        if from_project_context and from_project_context.is_file():
            context_path = from_project_context
        else:
            context_path = ado_config.project_context_path_from_active_context()

        # If we have a context path, we will attempt to load it and set it
        # as active context.
        # This can happen in two cases:
        # - The configuration existed and had an active context.
        # - We are being asked to load from a context file.
        if context_path and context_path.is_file():
            try:
                project_context = ProjectContext.model_validate(
                    yaml.safe_load(context_path.read_text())
                )
            except pydantic.ValidationError as e:
                console_print(
                    f"{ERROR}The provided project context was not valid:\n\t{e}",
                    stderr=True,
                )
                raise typer.Exit(1) from e

            ado_config._project_context = project_context
            ado_config.active_context = project_context.project
            return ado_config

        # At this point we don't have a context_path to load from.
        # If there are available contexts, we will tell the user to
        # choose one and move on. Otherwise, we will create, activate,
        # and store a default local context.
        if available_contexts := ado_config.available_contexts:

            # If the user is running ado context do not fail
            if do_not_fail_on_available_contexts:
                return ado_config

            console_print(
                f"{ERROR}No default context is set or the context file for it is missing.\n"
                f"{HINT}You can use {cyan('ado context')} to activate one of your available contexts: "
                f"{available_contexts}",
                stderr=True,
            )
            raise typer.Exit(1)

        console_print(
            f"{INFO}Initializing contexts - {cyan(ADO_DEFAULT_LOCAL_CONTEXT_NAME)} is now your default context.",
            stderr=True,
        )

        # We need the path to the SQLiteDB
        local_db_location = ado_config.local_dbs_dir / Path(
            f"{ADO_DEFAULT_LOCAL_CONTEXT_NAME}.db"
        )
        Path.mkdir(local_db_location.parent, parents=True, exist_ok=True)
        if local_db_location.exists():
            console_print(
                f"{INFO}Found a pre-existing database at {local_db_location}",
                stderr=True,
            )

        local_context = ProjectContext(
            project=ADO_DEFAULT_LOCAL_CONTEXT_NAME,
            metadataStore=SQLiteStoreConfiguration(
                path=str(local_db_location),
                database=ADO_DEFAULT_LOCAL_CONTEXT_NAME,
            ),
        )

        # Write the context to file
        ado_config.project_context_path_for_context(
            ADO_DEFAULT_LOCAL_CONTEXT_NAME
        ).write_text(local_context.model_dump_json())

        # Set the fields in the configuration
        ado_config._project_context = local_context
        ado_config.active_context = ADO_DEFAULT_LOCAL_CONTEXT_NAME

        # Store the configuration
        ado_config.store()
        return ado_config

    def store(self) -> None:
        Path.mkdir(self._app_dir, parents=True, exist_ok=True)
        self.config_file.write_text(self.model_dump_json())

    @property
    def project_context(self) -> ProjectContext | None:
        return self._project_context

    @property
    def available_contexts(self) -> list[str]:
        """Get a list of available contexts.

        Returns:
            A list of strings representing the names of the available contexts.
        """
        Path.mkdir(self.contexts_dir, parents=True, exist_ok=True)
        return [
            file.removesuffix(".json")
            for file in os.listdir(self.contexts_dir)
            if not file.startswith(".")  # Required for macOS's .DS_Store files
        ]

    @property
    def config_file(self) -> Path:
        return self._app_dir / Path(ADO_CONFIG_FILE_NAME)

    @property
    def contexts_dir(self) -> Path:
        return self._app_dir / Path(ADO_CONTEXTS_DIR_NAME)

    @property
    def local_dbs_dir(self) -> Path:
        return self._app_dir / Path(ADO_LOCAL_DBS_DIR_NAME)

    def project_context_path_for_context(self, context_name: str) -> Path:
        """
        This function takes in a context name and returns the path to the project context file for that context.
        Note that this function does not check whether the context exists or not.

        Parameters:
            context_name (str): The name of the context.

        Returns:
            Path: The path to the project context file for the given context.

        Raises:
            ValueError: If the context name is empty.
        """

        if len(context_name) == 0:
            raise ValueError

        return self.contexts_dir / Path(f"{context_name}.json")

    def local_db_path_for_context(self, context_name: str) -> Path:
        """
        This function takes in a context name and returns the path to the local db for that context.
        Note that this function does not check whether the context exists or not.

        Parameters:
            context_name (str): The name of the context.

        Returns:
            Path: The path to the local db for the given context.

        Raises:
            ValueError: If the context name is empty.
        """

        if len(context_name) == 0:
            raise ValueError

        return self.local_dbs_dir / Path(f"{context_name}.db")

    def project_context_model_for_context(
        self, context_name: str
    ) -> orchestrator.metastore.project.ProjectContext:
        """
        This function takes in a context name and returns the corresponding project context model.

        Parameters:
        context_name (str): The name of the context.

        Returns:
        orchestrator.metastore.project.ProjectContext: The project context model for the given context.

        Raises:
        ValueError: If no context file exists for the given context name.
        """
        path = self.project_context_path_for_context(context_name)
        if not path or not path.is_file():
            raise ValueError(f"No context file exists for context {context_name}")

        return orchestrator.metastore.project.ProjectContext.model_validate(
            yaml.safe_load(
                self.project_context_path_for_context(context_name).read_text()
            )
        )

    def project_context_path_from_active_context(self) -> Path | None:
        """
        Get the path to the project context file for the active context.

        Returns:
            Path: The path to the project context file. None if there is no active context.
        """
        return (
            self.project_context_path_for_context(self.active_context)
            if self.active_context
            else None
        )
