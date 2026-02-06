#!/usr/bin/env bash
# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT


ray job submit --address http://localhost:8265  --working-dir $PWD -v -- \
  python generate_dataset.py -o /data/fms-hf-tuning/artificial-dataset/news-tokens-16384plus-entries-4096.jsonl

