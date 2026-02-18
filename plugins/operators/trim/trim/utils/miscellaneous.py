# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT

import logging
import os
import shutil


def delete_dir(model_dir: str | None) -> None:
    if model_dir and os.path.isdir(model_dir):
        try:
            shutil.rmtree(model_dir, ignore_errors=True)
            logging.info(f"Deleted model directory: {model_dir}")
        except Exception as e:
            logging.warning(f"Could not delete model directory '{model_dir}': {e}")
