#!/bin/bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

set -e
source "ensure_variable_is_defined.sh"

print_usage() {
    printf "Usage: %s --admin-user MYSQL_ADMIN_USER
                      --admin-pass MYSQL_ADMIN_PASSWORD
                      <--project-name | -p PROJECT_NAME>
                      [--mysql-endpoint | -e <ENDPOINT>]\n\n" "$0"
    echo "NOTE: Endpoint is defaulted to localhost if not provided."
}

# Creates a DB called $PROJECT_NAME
create_database() {
  MYSQL_CMD="CREATE DATABASE \`${PROJECT_NAME}\`;"
  mysql -h"${MYSQL_ADDRESS}" -u"${MYSQL_ADMIN_USER}" -p"${MYSQL_ADMIN_PASSWORD}" -e "$MYSQL_CMD"
}

# Creates a user called $PROJECT_NAME and outputs its password
create_user() {
  if [[ $(uname) == 'Darwin' ]]; then
    PASSWORD_FOR_USER=$(openssl rand -base64 32 | tr -dc _A-Z-a-z-0-9)
  else
    PASSWORD_FOR_USER=$(< /dev/urandom tr -dc _A-Z-a-z-0-9 | head -c 32)
  fi

  MYSQL_CMD="CREATE USER '${PROJECT_NAME}'@'%' IDENTIFIED BY '${PASSWORD_FOR_USER}';"
  mysql -h"${MYSQL_ADDRESS}" -u"${MYSQL_ADMIN_USER}" -p"${MYSQL_ADMIN_PASSWORD}" -e "$MYSQL_CMD"
  echo "$PASSWORD_FOR_USER"
}

# Grants user $PROJECT_NAME all privileges on DB $PROJECT_NAME
grant_privileges_to_user() {
  # Backticks needed to allow for dashes to be in the db name
  MYSQL_CMD="GRANT ALL PRIVILEGES ON \`${PROJECT_NAME}\`.* TO '${PROJECT_NAME}'@'%'; FLUSH PRIVILEGES;"
  mysql -h"${MYSQL_ADDRESS}" -u"${MYSQL_ADMIN_USER}" -p"${MYSQL_ADMIN_PASSWORD}" -e "$MYSQL_CMD"
}

MYSQL_ADDRESS="127.0.0.1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--mysql-endpoin)
            MYSQL_ADDRESS="$2"
            shift 2
            ;;
        -p|--project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --admin-user)
            MYSQL_ADMIN_USER="$2"
            shift 2
            ;;
        --admin-pass)
            MYSQL_ADMIN_PASSWORD="$2"
            shift 2
            ;;
        *)
            print_usage >&2
            exit 1
            ;;
    esac
done

ensure_variable_is_defined MYSQL_ADMIN_USER
ensure_variable_is_defined MYSQL_ADMIN_PASSWORD
ensure_variable_is_defined PROJECT_NAME

echo
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Creating MySQL Database $PROJECT_NAME"
create_database
echo
echo "Creating MySQL User $PROJECT_NAME"
USER_PWD=$(create_user)
echo
echo "Granting user $PROJECT_NAME permissions on the $PROJECT_NAME database"
grant_privileges_to_user
echo
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo

# Save to file
echo "
metadataStore:
  database: $PROJECT_NAME
  host: $MYSQL_ADDRESS
  password: $USER_PWD
  scheme: mysql+pymysql
  sslVerify: true
  user: $PROJECT_NAME
project: $PROJECT_NAME
" > ${PROJECT_NAME}_context.yaml

echo "-------------------------------------------------------------"
echo "The project has now been onboarded!"
cat "${PROJECT_NAME}_context.yaml"
echo "-------------------------------------------------------------"
echo "Use ado create context -f ${PROJECT_NAME}_context.yaml to use it in ado"
