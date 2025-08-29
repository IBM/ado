# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import pydantic
import pytest

import orchestrator.utilities.location


def test_resource_location_from_url():

    url = "https://user:mypass123@localhost:8080/path"

    location = orchestrator.utilities.location.ResourceLocation.locationFromURL(url)

    assert location.scheme == "https"
    assert location.host == "localhost"
    assert location.port == 8080
    assert location.user == "user"
    assert location.password == "mypass123"
    assert location.path == "/path"

    # url with no scheme fails
    # This actual works because it thinks the scheme is user
    # url = "user:mypass123@localhost:8080/path"

    url = "mypass123@localhost:8080/path"

    with pytest.raises(pydantic.ValidationError):
        orchestrator.utilities.location.ResourceLocation.locationFromURL(url)

    with pytest.raises(pydantic.ValidationError):
        orchestrator.utilities.location.ResourceLocation(
            host="localhost",
            password="mypass123",
            user="user",
            port=8080,
            path="path",
        )


def test_resource_location_extra_forbid():

    # extra should not be allowed
    with pytest.raises(pydantic.ValidationError):
        orchestrator.utilities.location.ResourceLocation(
            scheme="https", host="localhost", port=8080, extra=10
        )


def test_resource_location_url_formation():

    url = "https://michaelj:mypass123@localhost:8080/path"

    location = orchestrator.utilities.location.ResourceLocation.locationFromURL(url)
    assert (
        location.url().unicode_string()
        == "https://michaelj:mypass123@localhost:8080/path"
    )
    assert (
        location.url(hide_pw=True).unicode_string()
        == "https://michaelj:*********@localhost:8080/path"
    )
    assert location.baseUrl().unicode_string() == "https://localhost:8080/path"


def test_resource_location_port_in_host_migration():

    location = orchestrator.utilities.location.ResourceLocation(
        scheme="https", host="localhost:8080"
    )

    assert location.host == "localhost"
    assert location.port == 8080


def test_pretty_resource_location():

    from IPython.lib.pretty import pretty

    url = "https://user:mypass123@localhost:8080/path"

    location = orchestrator.utilities.location.ResourceLocation.locationFromURL(url)

    pretty(location)


def test_file_path_location_with_existing():

    location = orchestrator.utilities.location.FilePathLocation(
        path="examples/pfas-generative-models/operation_transformer_benchmark.yaml"
    )
    assert location


def test_file_path_location_with_non_existing():

    # We don't want FilePathLocation to raise an error if the path doesn't exist
    # Instead it prints a warning
    # This is because the file-path may exist on one users computer but not on another
    # If the ExistingFilePathLocation model is stored somewhere the second user will not be able to read it
    # if it throws an error.

    location = orchestrator.utilities.location.FilePathLocation(
        path="examples/pfas-generative-models/config-benchmar.yaml"
    )
    assert location


def test_path_location():

    rl = orchestrator.utilities.location.ResourceLocation(
        scheme="file", path="/tmp/file.db"
    )

    assert rl.path == "/tmp/file.db"

    conf = rl.model_dump()

    orchestrator.utilities.location.ResourceLocation.model_validate(conf)

    assert rl.baseUrl().unicode_string() == "file:///tmp/file.db"
    assert rl.url().unicode_string() == "file:///tmp/file.db"


### SQLStoreConfiguration tests


### MySQLDsn
def test_valid_mysql_store_configuration():

    configuration = orchestrator.utilities.location.SQLStoreConfiguration(
        scheme="mysql+pymysql",
        host="localhost",
        port=3306,
        sslVerify=True,
        database="mydb",
        user="admin",
        password="somepass",
        path="mydb",
    )

    assert (
        configuration.url().unicode_string()
        == "mysql+pymysql://admin:somepass@localhost:3306/mydb"
    )

    assert (
        configuration.baseUrl().unicode_string()
        == "mysql+pymysql://localhost:3306/mydb"
    )

    orchestrator.utilities.location.SQLStoreConfiguration.model_validate(
        configuration.model_dump()
    )


def test_mysql_store_configuration_user_required():
    with pytest.raises(
        ValueError,
        match=r"You must specify the user when using MySQL",
    ):
        orchestrator.utilities.location.SQLStoreConfiguration(
            scheme="mysql+pymysql",
            host="localhost",
            port=3306,
            sslVerify=True,
            database="mydb",
            password="somepass",
            path="mydb",
        )


# def test_mysql_store_configuration_password_required():
#     with pytest.raises(
#         ValueError,
#         match=r"You must specify the password when using MySQL",
#     ):
#         orchestrator.utilities.location.SQLStoreConfiguration(
#             scheme="mysql+pymysql",
#             host="localhost",
#             port=3306,
#             sslVerify=True,
#             database="mydb",
#             user="admin",
#             path="mydb",
#         )


### SQLiteDsn
def test_valid_sqlite_store_configuration():

    configuration = orchestrator.utilities.location.SQLiteStoreConfiguration(
        scheme="sqlite",
        database="mydb",
        path="mydb.db",
    )

    assert configuration.url().unicode_string() == "sqlite:///mydb.db"

    orchestrator.utilities.location.SQLiteStoreConfiguration.model_validate(
        configuration.model_dump()
    )


def test_sqlite_store_configuration_purges_unused_fields():

    configuration = orchestrator.utilities.location.SQLiteStoreConfiguration(
        scheme="sqlite",
        host="localhost",
        port=3306,
        sslVerify=True,
        database="mydb",
        path="mydb.db",
        user="admin",
        password="password",
    )

    assert configuration.host is None
    assert configuration.port is None
    assert configuration.user is None
    assert configuration.password is None
    assert not configuration.sslVerify


### Generic SQLStoreConfiguration tests
def test_sql_store_configuration_database_required():
    with pytest.raises(
        ValueError,
        match=r"1 validation error for SQLStoreConfiguration\ndatabase\n  Field required",
    ):
        orchestrator.utilities.location.SQLStoreConfiguration(
            scheme="mysql+pymysql",
            host="localhost",
            port=3306,
            sslVerify=True,
            user="admin",
            password="somepass",
            path="mydb",
        )


def test_store_configuration_path_filled_in():
    configuration = orchestrator.utilities.location.SQLStoreConfiguration(
        scheme="mysql+pymysql",
        host="localhost",
        sslVerify=True,
        database="mydb",
        user="admin",
        password="somepass",
    )
    assert configuration.path == configuration.database
