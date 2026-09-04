---
name: remote-execution
description:
  Run ado operations on remote Ray clusters using --remote execution context
  files. Use when the user wants to create an operation, asks about remote
  clusters, wants to ship local plugins or data files to a cluster, or asks
  about execution context YAML files. Also applies proactively when creating an
  operation if execution context files are present in the workspace.
---

# Running ado on remote Ray clusters

## Execution context files

An execution context YAML configures a specific cluster and environment.
Multiple files can exist in the same repo for different clusters or
environments:

```text
morrigan_execution.yaml          # Morrigan cluster, standard env
vela_execution.yaml              # Vela cluster
morrigan_vllm_dev_execution.yaml # Morrigan + vllm_performance from source
```

The file names are user-defined conventions. Check the repo root for any
`*_execution.yaml` files to discover what contexts are available.

*If the source repo is available, read
`docs/user-guide/advanced/remote-execution.md` directly. Otherwise see
<https://ibm.github.io/ado/latest/user-guide/advanced/remote-execution/>*

See [Execution Context YAML Reference](#execution-context-yaml-reference) below
for the essential fields inlined.

---

## Proactive prompt when creating operations

When the user asks to create an operation, **check the repo root for
`*_execution.yaml` files**. If any exist, ask which (if any) they want to use
before proceeding. Do not assume remote by default — local execution is still
common.

Do **not** dispatch other ado commands (`get`, `show`, `create space`, etc.)
remotely unless the user explicitly requests it.

---

## Prerequisites

Before dispatching to a cluster with port-forward, verify cluster login:

```bash
oc whoami   # OpenShift
# or
kubectl get nodes   # Kubernetes
```

If this fails, request user to log in first — the port-forward will
fail with a credentials error otherwise.

---

## Project context

The active local project context is automatically forwarded to the remote job.
To work on the same project locally and remotely, use the same active context
for both — do not add a separate `-c` flag unless explicitly switching context:

```bash
# Local: uses active context
uv run ado create space -f space.yaml

# Remote: forwards the same active context automatically
uv run ado --remote morrigan_execution.yaml create operation \
    -f operation.yaml --use-latest space
```

Only supply `-c context.yaml` when you need to target a different project than
the one currently active.

---

## Operation creation command patterns

**One step** — create space and operation together remotely:

```bash
uv run ado --remote execution_context.yaml create operation \
    -f operation.yaml \
    --with space=space.yaml
```

**Two steps** — create space locally, run operation remotely:

```bash
uv run ado create space -f space.yaml
uv run ado --remote execution_context.yaml create operation \
    -f operation.yaml --use-latest space
```

Prefer the two-step pattern when you want the space registered in the local
metastore (e.g. for local querying or validation) before submitting.

---

## Tips

Check if all entries under `additionalFiles` in the remote
execution context YAML are required for the current submission.
Comment or remove those that are not to avoid uploading
unnecessary data.

Check if the value of the `wait` field is suitable for
the command being executed. In general do not wait
for `create operation` as it can be hours long.
If you are executing `get` or `show` commands waiting is valid
as these may only take seconds to minutes.

## Common Issues

### file paths in YAML not valid on the remote cluster

Any file path appearing in a space, operation, or actuator configuration YAML
(e.g. `mps_file`, a model checkpoint, a dataset path) must satisfy **both**
conditions for the remote job to succeed:

- **File not present on cluster**: add the local path to `additionalFiles` in
  the execution context YAML.
- **Path invalid on cluster**: use a bare filename in the YAML; ado symlinks
  `additionalFiles` entries into the Ray working dir, so `my-file.gz` resolves
  but `/Users/me/data/my-file.gz` does not.

Failing either condition produces a file-not-found error at experiment runtime,
not at submission time, so the job starts successfully but measurements fail.

#### Pattern to follow

To avoid, if the experiment references a file use a **bare filename** (no path)
in the space/operation YAML. Add the absolute local path to `additionalFiles`;
ado symlinks it into the Ray working directory so the bare filename resolves on
the cluster.

```yaml
# space.yaml
entitySpace:
  - identifier: mps_file
    propertyDomain:
      variableType: OPEN_CATEGORICAL_VARIABLE_TYPE
      values:
        - pigeon-10.mps.gz # bare filename — resolves from working dir
```

```yaml
# execution_context.yaml
additionalFiles:
  - /absolute/local/path/to/pigeon-10.mps.gz
```

The same applies to actuator configuration files that reference local paths
(e.g. model weights, config files). Audit all `-f` files for local path
references before dispatching remotely.

### Ray version mismatch

If you see `Changing the ray version is not allowed`, pin the Ray version in
`fromPyPI` to match the cluster:

```yaml
fromPyPI:
  - ado-core
  - ray==2.52.1 # match the cluster's installed version
  - ado-ray-tune
```

### fromSource plugin changes not reflected in remote run

Local edits to a plugin included via `fromSource` may not be reflected in
the remote run. Symptoms include: a fixed import error still occurring, an
added log line not appearing, or a new parameter not being present.

The most likely cause is that the wheel built for the plugin has the same
version as a wheel already cached by Ray. Ray sees the version as already
installed and skips reinstallation.

Plugins use `uv-dynamic-versioning` with a `format-jinja` template that appends
the git node and a timestamp to every dirty or dev build (e.g.
`X.Y.Z.devN+g<commit>.d<timestamp>`), so each build produces a unique version.
If you are still seeing stale wheels, confirm the plugin's `pyproject.toml`
has the `format-jinja` block from the
[plugin-development](../plugin-development/SKILL.md) skill.

---

## Execution Context YAML Reference

This section is self-contained — use it without reading any external file.
Only go to the source for topics not covered here: read
`docs/user-guide/advanced/remote-execution.md` if the source repo is available,
otherwise see <https://ibm.github.io/ado/latest/user-guide/advanced/remote-execution/>.

### Minimal — direct cluster URL

```yaml
executionType:
  type: cluster
  clusterUrl: "http://ray-cluster.my-namespace.svc.cluster.local:8265"
packages:
  fromPyPI:
    - ado-core
    - ado-ray-tune  # add any other plugins required
envVars:
  PYTHONUNBUFFERED: "x"
  OMP_NUM_THREADS: "1"
wait: false  # true = stay attached until the job finishes
```

### Port-forward cluster (OpenShift / Kubernetes)

```yaml
executionType:
  type: cluster
  clusterUrl: "http://localhost:8265"  # must match localPort below
  portForward:
    namespace: my-namespace
    serviceName: my-ray-cluster-head-svc
    localPort: 8265
packages:
  fromPyPI:
    - ado-core
    - ado-ray-tune
envVars:
  PYTHONUNBUFFERED: "x"
wait: false
```

### `packages` block

```yaml
packages:
  fromPyPI:
    - ado-core
    - ado-ray-tune
    - ray==2.52.1                         # pin to match cluster version if needed
    - /remote/path/to/package.whl         # path to a wheel already on the cluster
  fromSource:
    - plugins/actuators/my_plugin  # relative to where ado --remote is run
```

### `additionalFiles` — ship local files to the cluster

```yaml
additionalFiles:
  - /absolute/local/path/to/data_file.csv
  - path/to/my_data_dir/   # directories also supported
```

Use bare filenames (no path) in space/operation YAML; ray copies
`additionalFiles` entries into the Ray working directory.

### `runtimeEnv` block

```yaml
runtimeEnv:
  setupTimeoutSeconds: 1200  # default 600; -1 to disable
  eagerInstall: false        # default true
```

### `envVars` block

```yaml
envVars:
  PYTHONUNBUFFERED: "x"
  OMP_NUM_THREADS: "1"
  OPENBLAS_NUM_THREADS: "1"
  RAY_AIR_NEW_PERSISTENCE_MODE: "0"
  MY_SECRET: "value"
```
