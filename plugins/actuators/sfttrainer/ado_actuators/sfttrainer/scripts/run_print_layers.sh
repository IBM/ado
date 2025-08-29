#!/usr/bin/env bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


ray job submit --address http://localhost:8265  --working-dir $PWD -v python print_linear_layers.py 2>&1 | tee linear_layers.txt

