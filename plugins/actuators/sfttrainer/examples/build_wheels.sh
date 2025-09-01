#!/usr/bin/env bash
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

# Populates the current directory with:
# 1. the wheel file for ado
# 2. the wheel file for the SFTTrainer actuator
# 3. a YAML file containing a Ray runtime environment definition that references the 2 above wheels

set -e

my_working_dir=$(pwd)
dir_sfttrainer=$(realpath $(dirname $0)/../)
dir_ado=$(realpath ${dir_sfttrainer}/../../../)

cd ${dir_ado}
echo Building the wheel for ado
rm -rf dist build
python -m build -w
mv dist/*.whl ${my_working_dir}
rm -rf build dist


echo Building the wheel for SFTTrainer
cd $dir_sfttrainer
rm -rf dist build
python -m build -w
mv dist/*.whl ${my_working_dir}
rm -rf build dist

cd ${my_working_dir}

sftrainer_wheel_prefix="ado_sfttrainer"
ado_wheel_prefix="ado_core"

cat <<EOF >ray_runtime_env.yaml
pip:
  - \${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/${ado_wheel_prefix}-$(ls ${ado_wheel_prefix}-* | cut -d "-" -f2-)
  - \${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/${sftrainer_wheel_prefix}-$(ls ${sftrainer_wheel_prefix}-* | cut -d "-" -f2-)
env_vars:
  env_vars:
  AIM_UI_TELEMETRY_ENABLED: "0"
  # We set HOME to /tmp because "import aim.utils.tracking" tries to write under \$HOME/.aim_profile.
  # However, the process lacks permissions to do so and that leads to an ImportError exception.
  HOME: "/tmp/"
  OMP_NUM_THREADS: "1"
  OPENBLAS_NUM_THREADS: "1"
  RAY_AIR_NEW_PERSISTENCE_MODE: "0"
  PYTHONUNBUFFERED: "x"
EOF

echo "Finished building the wheels and generating the ray_runtime_env.yaml file"
ls -lth .