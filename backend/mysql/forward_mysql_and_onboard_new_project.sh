#!/bin/bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

set -e
source "ensure_variable_is_defined.sh"

print_usage() {
    printf "Usage: %s --admin-user MYSQL_ADMIN_USER
                      --admin-pass MYSQL_ADMIN_PASSWORD
                      --pxc-name PERCONA_CLUSTER_NAME
                      <--project-name | -p PROJECT_NAME>
                      [--namespace | -n <namespace>]\n\n" "$0"
    echo "NOTE: Namespace is defaulted to current namespace ($CURRENT_NAMESPACE) if not provided."
}

CURRENT_NAMESPACE=$(kubectl config view --minify --output 'jsonpath={..namespace}')
MYSQL_NAMESPACE=$CURRENT_NAMESPACE

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--namespace)
            MYSQL_NAMESPACE="$2"
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
        --pxc-name)
            PXC_NAME="$2"
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
ensure_variable_is_defined PXC_NAME

# Inspired by https://stackoverflow.com/a/68824178
MYSQL_SVC_NAME=$(kubectl get svc -n "$MYSQL_NAMESPACE" -l app.kubernetes.io/instance="$PXC_NAME" -o jsonpath='{.items[0].metadata.name}')
echo "Starting to forward the PXC service $MYSQL_SVC_NAME locally"
kubectl port-forward -n "$MYSQL_NAMESPACE" "svc/${MYSQL_SVC_NAME}" mysql >/dev/null 2>&1 & 
PORT_FORWARD_PID=$!
trap '{
    echo "Cleaning up port-forward"
    # Kill the port-forwarding
    kill $PORT_FORWARD_PID
    # Avoid printing out the "Killed" message
    wait $PORT_FORWARD_PID 2>/dev/null
    exit 0
}' EXIT SIGINT

echo "Waiting for the forwarding to start"
while ! nc -vz localhost 3306 > /dev/null 2>&1 ; do
    sleep 1
done

./onboard_new_project.sh -e 127.0.0.1 --admin-user "$MYSQL_ADMIN_USER" --admin-pass "$MYSQL_ADMIN_PASSWORD" -p "$PROJECT_NAME"