#!/usr/bin/env bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT


ray job submit --address http://localhost:8265 --runtime-env environment_download_weights_s3.yaml \
  --working-dir $PWD -v python download_weights_from_s3.py

