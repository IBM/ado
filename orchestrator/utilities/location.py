# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import logging
import os
import typing
from typing import Annotated

import pydantic
from pydantic import ConfigDict

if typing.TYPE_CHECKING:
    from rich.console import RenderableType

moduleLog = logging.getLogger("location")


class ResourceLocation(pydantic.BaseModel):
    """A model for URLs/URIs"""

    @classmethod
    def locationFromURL(cls, string: str) -> "ResourceLocation":

        # Note2: Users can pass the port as part of host but pydantic will

        url = pydantic.AnyUrl(string)
        return cls(
            scheme=url.scheme,
            host=url.host,
            port=url.port,
            path=url.path,
            user=url.username,
            password=url.password,
        )

    scheme: Annotated[str, pydantic.Field(description="The resource access scheme")]
    host: Annotated[
        str | None,
        pydantic.Field(
            description="The host name for the resource. Should not contain port"
        ),
    ] = None
    # validating default of None to allow detecting if the port was placed in the host field
    port: Annotated[
        int | None,
        pydantic.Field(description="Port number", validate_default=True),
    ] = None
    path: Annotated[
        str | None, pydantic.Field(description="The path of the resource")
    ] = None
    user: Annotated[str | None, pydantic.Field(description="The user")] = None
    password: Annotated[str | None, pydantic.Field(description="The password")] = None

    model_config = ConfigDict(extra="forbid")

    @pydantic.model_validator(mode="after")
    def check_if_host_specifies_port(self) -> "ResourceLocation":
        """port should not be included in the host name, but use the port field

        This validator checks if the port is in the host field.
        If it is it takes it from the host field and moves it to the port field

        The reason not to validate the host field and emit a ValidationError is we have existing stored data
        where host includes port and if we raised such an error it could not be read"""

        if self.host is not None:
            url = pydantic.AnyUrl.build(scheme=self.scheme, host=self.host)
            if str(url.port) in self.host:
                moduleLog.warning(
                    f"You should not include port number in host field - {url.port} detected. Use the port field. "
                    f"The port will be migrated to the port field"
                )
                self.host = url.host
                self.port = url.port

        return self

    def url(self, hide_pw: bool = False) -> pydantic.AnyUrl:
        """Note: pydantic 2 up to at least 2.6.4 adds trailing path separators to host names

        e.g. https://localhost -> https://localhost/

        Or it may be adding pre path separators to path

        e.g. path = dir/ becomes /dir/

        https://github.com/pydantic/pydantic/issues/7186"""

        urlClass = pydantic.FileUrl if self.scheme == "file" else pydantic.AnyUrl

        if hide_pw:
            return urlClass.build(
                scheme=self.scheme,
                password="*" * len(self.password),
                host=self.host if self.host is not None else "",
                port=self.port,
                username=self.user,
                path=self.path.lstrip("/"),
            )
        return urlClass.build(
            scheme=self.scheme,
            password=self.password,
            host=self.host if self.host is not None else "",
            port=self.port,
            username=self.user,
            path=None if self.path is None else self.path.lstrip("/"),
        )

    def baseUrl(self) -> pydantic.AnyUrl | pydantic.FileUrl:
        """Returns URL without password or user components"""

        urlClass = pydantic.FileUrl if self.scheme == "file" else pydantic.AnyUrl

        return urlClass.build(
            scheme=self.scheme,
            host=self.host if self.host is not None else "",
            port=self.port,
            path=self.path.lstrip("/"),
        )

    def __rich__(self) -> "RenderableType":
        """Render this location using rich."""
        from rich.text import Text

        return Text(self.url().unicode_string())


class FilePathLocation(ResourceLocation):

    scheme: Annotated[str, pydantic.Field(description="The resource access scheme")] = (
        "file"
    )

    @pydantic.field_validator("path", mode="before")
    def check_if_path_exists(cls, value: str) -> str:
        """Check if the path exists and emit a debug log if it does not"""
        import os

        value = os.path.expandvars(value)
        if not os.path.exists(value):
            moduleLog.debug(f"Specified file-path path, {value}, does not exist")

        return value

    @property
    def hash_identifier(self) -> str:
        """Returns an identifier for the file-path of the form {filename}-{file hash}"""

        import hashlib

        import pandas as pd

        file_hash = hashlib.md5(
            pd.util.hash_pandas_object(pd.read_csv(self.path), index=True).values,
            usedforsecurity=False,
        ).hexdigest()
        filename = os.path.split(os.path.expandvars(self.path))[1]
        return f"{filename}-{file_hash}"

    def url(self, hide_pw: bool = False) -> pydantic.AnyUrl:
        """Note: pydantic 2 up to at least 2.6.4 adds trailing path separators to host names

        e.g. https://localhost -> https://localhost/

        https://github.com/pydantic/pydantic/issues/7186"""

        return pydantic.FileUrl.build(
            scheme=self.scheme,
            path=self.path,
            host="",
        )


class StorageDatabaseConfiguration(ResourceLocation):
    """Configuration for accessing database.

    Note: The property 'active' is for specifying to consumers of this conf whether it should
    in fact be used. By: default it is True. Ability to set to False allows for debugging
    """

    sslVerify: Annotated[
        bool, pydantic.Field(description="If False SSL verification is turned of")
    ] = True
    database: Annotated[str, pydantic.Field(description="The database to access")]
    model_config = ConfigDict(extra="forbid")


class SQLStoreConfiguration(StorageDatabaseConfiguration):
    """
    - If created directly "active" will the True if not specifically set
    """

    model_config = ConfigDict(extra="forbid")

    scheme: Annotated[
        str, pydantic.Field(description="The access scheme to the SQL database.")
    ] = "mysql+pymysql"

    path: Annotated[
        str | None,
        pydantic.Field(
            validate_default=True,
            description="The SQL database name. Will be automatically set based on database field.",
        ),
    ] = None

    @pydantic.model_validator(mode="after")
    def set_url_path_to_database_name(self) -> "SQLStoreConfiguration":

        moduleLog.debug(
            f"Replacing path value {self.path} with database value {self.database}"
        )
        self.path = self.database

        return self

    @pydantic.model_validator(mode="after")
    def check_valid_dsn(self) -> "SQLStoreConfiguration":
        """
        Validates that the url produced by this class is a valid
        MySQLDsn.

        This makes sure:
        - scheme is one of the supported MySQL schemes
        - host is required (as specified by the URLConstraints)

        The following fields are also ensured to be valid:
        - database (is required in StorageDatabaseConfiguration)
        - path (it's filled in by set_url_path_to_database_name)

        For MySQL, we also require the following fields:
        - user

        """
        # AP 02/09/2025:
        # We use a RootModel in case down the line we want to support more DSNs
        _AdoSupportedDsn = pydantic.RootModel[pydantic.MySQLDsn]
        m = _AdoSupportedDsn.model_validate(self.url(), strict=True).root

        if isinstance(m, pydantic.MySQLDsn) and not self.user:
            raise ValueError("You must specify the user when using MySQL")

        return self


class SQLiteStoreConfiguration(StorageDatabaseConfiguration):

    scheme: Annotated[
        typing.Literal["sqlite"], pydantic.Field(description="The SQLite access scheme")
    ] = "sqlite"
    path: Annotated[str, pydantic.Field(description="The path to the SQLite database")]
    database: Annotated[
        str | None,
        pydantic.Field(
            description="In SQLite the database is decided by the path. "
            "This field will be automatically set to None"
        ),
    ] = None

    @pydantic.model_validator(mode="after")
    def purge_unsupported_fields(self) -> "SQLiteStoreConfiguration":
        self.host = None
        self.port = None
        self.user = None
        self.database = None
        self.password = None
        self.sslVerify = False
        return self

    def url(self, hide_pw: bool = False) -> pydantic.AnyUrl:
        if " " in self.path:
            import warnings

            warnings.warn(
                "The path to the SQLite database contains whitespace. "
                "The URL being generated should not be used to connect to the database.",
                stacklevel=2,
            )

        return pydantic.AnyUrl.build(
            scheme=self.scheme,
            path=self.path,
            host="",
        )


def db_scheme_discriminator(
    configuration: dict | SQLiteStoreConfiguration | SQLStoreConfiguration,
) -> str:
    if isinstance(configuration, dict):
        scheme = configuration.get("scheme")
    else:
        scheme = configuration.scheme

    return "sqlite" if scheme == "sqlite" else "mysql"
