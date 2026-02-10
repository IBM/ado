#!/usr/bin/env bash
# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


ray job submit --address http://localhost:8265  --working-dir $PWD -v python hash_models.py 2>&1 | tee model_hashes.txt

