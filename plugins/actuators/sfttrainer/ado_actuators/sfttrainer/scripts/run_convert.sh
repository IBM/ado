#!/usr/bin/env bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


ray job submit --address http://localhost:8265  --working-dir $PWD -v python convert_weights.py

