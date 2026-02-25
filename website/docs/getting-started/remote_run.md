# Running `ado` on remote Ray clusters

<!-- markdownlint-disable-next-line first-line-h1 -->

> [!NOTE] Overview
>
> Running `ado` on a remote Ray cluster enables long-running operations that can
> utilize multiple nodes and large amounts of compute-resource like GPUs. Such
> resources may also be a requirement for certain experiments or actuators.

The `--remote` option automates the steps required to dispatch any `ado` command
to a remote Ray cluster. It handles packaging files, building plugin wheels,
generating the Ray runtime environment, and running `ray job submit` for you.

## Prerequisites

> [!IMPORTANT] Only remote project contexts are supported
>
> The project context used must be
> [remote](https://ibm.github.io/ado/resources/metastore/#contexts-for-remote-projects),
> as it must be accessible when ado executes on the remote ray cluster. `ado`
> will fail with a clear error if a SQLite context is detected.

<!-- markdownlint-disable-next-line MD028 -->
> [!IMPORTANT] Cluster login
>
> If your cluster requires a port-forward, `oc` (OpenShift CLI) or `kubectl`
> must be installed, and you must be logged in to the cluster.

## Defining a remote execution context

The details about a remote execution environment, where it is, what packages to
install, and what environment variables to set, are defined in a YAML
configuration file. Here we will call this file `remote_context.yaml` but it can
have any name. There can be multiple such files for different remote clusters,
or for specifying different environments on those clusters.

The minimal example uses a Ray cluster that is directly reachable at a known
URL:

<!-- markdownlint-disable line-length -->

```yaml
executionType:
  type: cluster
  clusterUrl: "http://ray-cluster.my-namespace.svc.cluster.local:8265"
packages:
  fromPyPI:
    - ado-core
    - ado-ray-tune # Add any other plugins required by your operation
envVars:
  PYTHONUNBUFFERED: "x"
  OMP_NUM_THREADS: "1"
  OPENBLAS_NUM_THREADS: "1"
  RAY_AIR_NEW_PERSISTENCE_MODE: "0"
wait: false # Set to true to remain attached until the job finishes
```

<!-- markdownlint-enable line-length -->

If your cluster is only reachable via a port-forward (common on OpenShift), add
the `portForward` sub-field. `ado` will start the port-forward automatically
before submitting and tear it down after:

<!-- markdownlint-disable line-length -->

```yaml
executionType:
  type: cluster
  clusterUrl: "http://localhost:8265" # Must match localPort below
  portForward:
    namespace: my-namespace
    serviceName: my-ray-cluster-head-svc
    localPort: 8265 # Default; the port oc/kubectl will bind locally
packages:
  fromPyPI:
    - ado-core
    - ado-ray-tune
envVars:
  PYTHONUNBUFFERED: "x"
  OMP_NUM_THREADS: "1"
wait: false
```

<!-- markdownlint-enable line-length -->

## Submitting commands

<!-- markdownlint-disable MD007 -->

> [!WARNING] Ray version mismatch errors
>
> If you encounter an error like:
>
> ```text
> RuntimeError: Changing the ray version is not allowed:
>   current version: 2.54.0,   expect version: 2.52.1
> ```
>
> This means the Ray version installed in your cluster differs from the version
> that will be installed by your dependencies. To resolve this, explicitly pin
> the Ray version in your `fromPyPI` section to match the cluster's version:
>
> ```yaml
> packages:
>   fromPyPI:
>     - ado-core
>     - ray==2.52.1 # Match the cluster's Ray version
>     - ado-ray-tune
> ```

<!-- markdownlint-enable MD007 -->

Pass `--remote` as a global option before any `ado` command.

By default, `ado` will use the current active context as the context for the
remote command.

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->

```commandline
ado --remote remote_context.yaml create operation -f operation.yaml
```

<!-- markdownlint-enable line-length -->

All `ado` commands are supported. For example, to query the metastore remotely:

```commandline
ado --remote remote_context.yaml get space
```

You can also supply a project context directly using `-c`

```commandline
ado -c mysql_project.yaml --remote remote_context.yaml create operation -f operation.yaml
```

> [!NOTE] What `--remote` does
>
> For each invocation `ado` will:
>
> 1. Copy the project context file and any `-f` resource files to a temporary
>    working directory.
> 2. Build wheels for any `fromSource` plugin paths.
> 3. Generate a `runtime_env.yaml` from the `packages` and `envVars` fields.
> 4. Start a port-forward if `portForward` is configured.
> 5. Run `ray job submit` with the assembled working directory and runtime
>    environment.
> 6. Tear down the port-forward (if started) and exit with the job's exit code.

## Installing python packages on a remote Ray cluster

When executing on a remote Ray cluster you often need to install additional
packages, either from PyPI or local development. There are three methods
available:

- [Pre-installing](#pre-installing-ado-packages): Best when you are using the
  same actuators and operators constantly
- [Dynamic installation from pypi](#dynamic-installation-from-pypi): Best in
  general case
- [Dynamic installation from source](#dynamic-installation-from-source): Best
  for developers

> [!NOTE] Ray python package caching
>
> Ray caches packages it is asked to install so they are only downloaded, and
> potentially built, the first time they are requested.

### Pre-installing ado packages

In this method `ado` and the required plugins are already installed in the Ray
cluster's base python environment i.e. in the image used for head and worker
nodes.

In this case you do not need to specify any packages in your
`remote_context.yaml`. This method has the benefit of not having any overhead in
job start from python package download or build steps.

<!-- markdownlint-disable MD007 -->

> [!WARNING] Using additional plugins with pre-installed ado
>
> If you need additional plugins or different versions of pre-installed plugins
> **you must do a dynamic installation of `ado-core` and all actuators you
> need**. This is because:
>
> - The pre-installed `ado` command is tied to the base-environment
>   - It will not see new packages. You need to install it into the job's
>     virtualenv
> - The ado_actuators namespace package will be superseded by one created in the
>   job's virtualenv
>   - Actuators in the same namespace package in the base environment will not
>     be seen

<!-- markdownlint-enable MD007 -->

### Dynamic installation from pypi

The recommended method is to specify `ado-core` and the pypi package names of
any plugins required in the `packages.fromPyPI` section of your
`remote_context.yaml`.

### Dynamic installation from source

If you need to install plugins or packages from source, specify the path to them
in the `packages.fromSource` section of your `remote_context.yaml`. Note: If the
path is relative it will be resolved from where you execute `ado --remote ...`

<!-- markdownlint-disable line-length -->

```yaml
executionType:
  type: cluster
  clusterUrl: "http://localhost:8265"
packages:
  fromPyPI:
    - ado-core
  fromSource:
    - plugins/actuators/vllm_performance # Assumes execute ado --remote from route of ado repo
wait: false
envVars:
  PYTHONUNBUFFERED: "x"
```

<!-- markdownlint-enable line-length -->

`ado` then will:

1. Build python wheels for those packages
2. Instruct Ray to install the wheels as part of the Ray job submission

## Sending additional files

If you have additional files that need to be sent use the `additionalFiles`
field of the remote execution context YAML. This can be required for example if
an operator or actuator requires these files as input.

The paths can be absolute or relative. If relative they are resolved with
respect to the directory `ado --remote [COMMAND]` is executed from.

```yaml
executionType:
  type: cluster
  clusterUrl: "http://localhost:8265"
packages:
  fromPyPI:
    - ado-core
  fromSource:
    - plugins/actuators/vllm_performance
wait: false
envVars:
  PYTHONUNBUFFERED: "x"
additionalFiles:
  - /absolute/path/to/data_file.csv
  - path/to/my_data_dir/ # directories are also supported
```
