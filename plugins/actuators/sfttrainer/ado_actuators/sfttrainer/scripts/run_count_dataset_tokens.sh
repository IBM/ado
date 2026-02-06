#!/usr/bin/env bash
# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


ray job submit --address http://localhost:8265  --working-dir $PWD -v python count_dataset_tokens.py 2>&1 | tee count_dataset_tokens.txt

