#!/usr/bin/env bash
# Copyright IBM Corporation 2025, 2026
# SPDX-License-Identifier: MIT
#
# Builds the isolated environment used to record readme_try_it_out.tape.
#
# The recording must show only the density example, so this script creates a
# throwaway venv containing nothing but ado-core and the density example
# package, and a throwaway HOME so that ado's config directory (contexts and
# the SQLite metastore) is separate from the developer's real one.
#
# Usage:  ./docs/videos/tapes/setup_demo_env.sh  [scratch_root]
# Default scratch root is /tmp/ado-readme-demo

set -euo pipefail

SCRATCH_ROOT="${1:-/tmp/ado-readme-demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DEMO_HOME="${SCRATCH_ROOT}/home"
DEMO_DIR="${SCRATCH_ROOT}/demo"
VENV_DIR="${SCRATCH_ROOT}/venv"

echo "Repo root:    ${REPO_ROOT}"
echo "Scratch root: ${SCRATCH_ROOT}"

rm -rf "${SCRATCH_ROOT}"
mkdir -p "${DEMO_HOME}" "${DEMO_DIR}"

# ---------------------------------------------------------------------------
# Clean venv holding exactly what the README tells a reader to install, and
# nothing else. ado-core comes from PyPI rather than this working tree so the
# provenance block ado prints during the recording shows a released version
# rather than a "dirty" local dev build. Set ADO_SOURCE=local to record against
# the working tree instead.
# ---------------------------------------------------------------------------
ADO_SOURCE="${ADO_SOURCE:-pypi}"
if [[ "${ADO_SOURCE}" == "local" ]]; then
  ADO_SPEC="${REPO_ROOT}"
else
  ADO_SPEC="ado-core"
fi

uv venv --python 3.11 "${VENV_DIR}"
VIRTUAL_ENV="${VENV_DIR}" uv pip install \
  --python "${VENV_DIR}/bin/python" \
  "${ADO_SPEC}" "${REPO_ROOT}/examples/density_example"

# ---------------------------------------------------------------------------
# Demo working directory: only the files the recording refers to.
# ---------------------------------------------------------------------------
cp "${REPO_ROOT}/examples/density_example/space.yaml" "${DEMO_DIR}/"
cp "${REPO_ROOT}/examples/density_example/operation.yaml" "${DEMO_DIR}/"
cp -R "${REPO_ROOT}/examples/density_example/density" "${DEMO_DIR}/"
rm -rf "${DEMO_DIR}/density/__pycache__"

# ---------------------------------------------------------------------------
# Isolated ado context. HOME drives typer.get_app_dir("ado"), so everything
# ado writes (contexts, SQLite db) lands under the scratch home.
# ---------------------------------------------------------------------------
export HOME="${DEMO_HOME}"
export PATH="${VENV_DIR}/bin:${PATH}"

# The recording opens files in vim so they are syntax highlighted. An empty
# HOME means vim starts with syntax off, so install the shared vimrc.
cp "${REPO_ROOT}/docs/videos/tapes/vimrc" "${DEMO_HOME}/.vimrc"

CONTEXT_FILE="${SCRATCH_ROOT}/density-demo-context.yaml"
cat >"${CONTEXT_FILE}" <<EOF
project: density-demo
metadataStore:
  scheme: sqlite
  path: ${DEMO_HOME}/Library/Application Support/ado/databases/density-demo.db
  database: null
  host: null
  password: null
  port: null
  sslVerify: false
  user: null
EOF

mkdir -p "${DEMO_HOME}/Library/Application Support/ado/databases"
ado create context -f "${CONTEXT_FILE}"
ado context density-demo

# Pre-warm the actuator registry and metastore so the first recorded command
# is not artificially slow.
(cd "${DEMO_DIR}" && ado get experiments >/dev/null 2>&1 || true)
(cd "${DEMO_DIR}" && ado get spaces >/dev/null 2>&1 || true)

# ---------------------------------------------------------------------------
# Pre-start a Ray head node. `ado create operation` calls ray.init(), which
# otherwise spends ~15 extra seconds booting a cluster mid-recording.
# ---------------------------------------------------------------------------
ray stop --force >/dev/null 2>&1 || true
ray start --head --num-cpus=4 --disable-usage-stats >/dev/null 2>&1

echo
echo "Ready. Active context:"
ado context
echo
echo "Record with (from the repository root, so the tape's Output path resolves):"
echo "  cd ${REPO_ROOT}"
echo "  vhs docs/videos/tapes/readme_try_it_out.tape"
echo "  ./docs/videos/tapes/make_gif.sh"
echo
echo "Tear down the Ray head node afterwards with:"
echo "  HOME=${DEMO_HOME} PATH=${VENV_DIR}/bin:\$PATH ray stop --force"
