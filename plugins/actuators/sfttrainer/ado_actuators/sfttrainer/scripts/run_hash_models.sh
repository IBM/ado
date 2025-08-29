#!/usr/bin/env bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


ray job submit --address http://localhost:8265  --working-dir $PWD -v python hash_models.py 2>&1 | tee model_hashes.txt

