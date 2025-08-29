#!/bin/bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


ensure_variable_is_defined() {
    if [[ -z "${!1}" ]]; then
        printf "Error: Variable %s is not defined and is required.\n\n" "$1"
        print_usage
        exit 1
    fi
}