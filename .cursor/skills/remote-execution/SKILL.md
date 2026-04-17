---
name: remote-execution
description:
  Run ado operations on remote Ray clusters using --remote execution context
  files. Use when the user wants to create an operation, asks about remote
  clusters, wants to ship local plugins or data files to a cluster, or asks
  about execution context YAML files. Also applies proactively when creating an
  operation if execution context files are present in the workspace.
---

# Remote Execution with ado

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

See `website/docs/getting-started/remote_run.md` for full schema reference.

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

### Behaviour of packages specified in fromSource does not reflect local edits

Often local edits are made to a plugin - to add new features or fix an issue -
that is included in a remote execution context via `fromSource`. In some cases
you may observe that the changes made are not reflected in the behaviour of the
remote run. Some examples:

- A problem in experiment decorator preventing its use is fixed,
but it still can't be imported
- A print log is added to record some information but the log does not appear
- A new parameter or output is added to an experiment but its not present in
remote run

The most likely cause of this behaviour is that the versioning of the wheel
built for
the plugin is not specific enough to distinguish it from a previously
installed wheel of the same plugin. For example, the default versioning
scheme may include commit and date, where date denotes there are
"dirty" local changes in the wheel. However, after the first install
of a plugin with a local edit on a day, all subsequent installs, even
with different edits have the same version, and ray will not install them,
thinking they are already present.

The solution is to use a more fine-grained versioning scheme for dirty edits
in the plugin pyproject.toml. See [plugin development](../../rules/plugin-development.mdc)
for details.
