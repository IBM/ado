# Running `ado` on remote Ray clusters

<!-- markdownlint-disable-next-line first-line-h1 -->

> [!NOTE] Overview
>
> Running `ado` on a remote ray cluster enables
> long-running operations that can utilize multiple nodes
> and large amounts of compute-resource like GPUs.
> Such resources may also be a requirement for certain
> experiments or actuators.

## Quickstart

### Getting ready

First, create an empty directory to store the files that need to be
uploaded to the remote Ray cluster for the `ado` command to run. In the
following we will refer to this directory as `$RAY_JOB_DATA_DIR`.

This will usually contain the following three YAML files:

<!-- markdownlint-disable MD007 -->
- A YAML file describing [the context](../resources/metastore.md) to use for the
  operation.
    - You can use `ado get context -o yaml` to get this file for the context you
      want to use
- A YAML file describing [the operation](../resources/operation.md) to create.
- A YAML file describing [the environment of the Ray job](#ray-runtime-environment-runtime-env)
<!-- markdownlint-enable MD007 -->

We will refer to the first file as `context.yaml`, the second as
`operation.yaml`, and the last as `runtime_env.yaml`, although they
can have any names.

### Setting the Ray job environment

An example `runtime_env.yaml` which dynamically installs the latest release
of `ado`, and some `ado` plugins, is:
<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```yaml
uv: # One line for each plugin (actuator, operator) to install
  - ado-core
  - ado-ray-tune # If you aim to run a ray-tune operation
  - ado-vllm-performance # Substitute with whatever plugins you need
env_vars: # These env_vars are recommended.
  PYTHONUNBUFFERED: "x" # Turns off buffering of the jobs logs. Useful if there is some error
  OMP_NUM_THREADS: "1" # Restricts the number of threads started by the python process in the job. If this is not set it can cause the ray job to exceed OpenShift node thread limits.
  OPENBLAS_NUM_THREADS: "1" # Same as above
  RAY_AIR_NEW_PERSISTENCE_MODE: "0" # Required for using the ray_tune operator
  #The following envars may be required or useful depending on your specific needs
  HOME: "/tmp" # Optional: Use if python code used by operation assumes $HOME is writable which it may not be
  LOGLEVEL: "WARNING" # Optional: Set this to get more/less debug logs from ado
```
<!-- markdownlint-enable line-length -->

>[!NOTE] Ray python package caching
>
> Ray caches packages it is asked to install so they are
> only downloaded, and potentially built, the first time
> they are requested.

### Submitting the operation

The command for submitting the job looks like:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --no-wait --address http://localhost:8265 \
  --working-dir $PWD --runtime-env ray_runtime.yaml -v -- \
  ado -c context.yaml create operation -f operation.yaml
```
<!-- markdownlint-enable line-length -->

This creates a detached Ray job that runs on the cluster.
You can go to the ray dashboard
(in this example at <http://localhost:8265>)
to see the job status, logs, and more.
You can read more about [ray submit command line options here](#ray-job-submit-options).

>[!NOTE] Specifying the context
>
> When you run remotely there is no active context.
> Hence, you have to specify the context to use with the `-c` option to ado,
> as well as sending the YAML file with the context to the Ray cluster

## Installing ado and ado plugins

There are three ways of installing ado and plugins on the remote cluster.

- [Pre-installing](#pre-installing-ado-packages): Best when you are using the
same actuators and operators constantly
- [Dynamic installation from pypi](#dynamic-installation-from-pypi):
Best in general case
- [Dynamic installation from source](#dynamic-installation-from-source):
Best for developers

### Pre-installing ado packages

In this method `ado` and the required plugins are installed in the
Ray cluster's base python environment i.e. in the image used for head and
worker nodes.

In this case you do not need to specify any packages in the `runtime_env.yaml`,
if the pre-installed ones are sufficient.
This method has the benefit of not having any overhead in job start from
python package download or build steps.

<!-- markdownlint-disable MD007 -->
>[!WARNING] Using additional plugins with pre-installed ado
>
> If you need additional plugins or different versions of pre-installed
> plugins **you must do a dynamic installation of `ado-core` and all
> actuators you need**.
> This is because:
>
> - The pre-installed `ado` command is tied to the base-environment
>     - It will not see new packages. You need to install it
> into the job's virtualenv
> - The ado_actuators namespace package will be superseded by one created in
> the job's virtualenv
>     - Actuators in the same namespace package in the base environment
> will not be seen
<!-- markdownlint-enable MD007 -->

### Dynamic installation from pypi

The recommended method is to specify `ado-core` and the pypi package names
of any plugins required in the `uv` section of the `runtime_env.yaml` as
shown in [the quickstart example](#setting-the-ray-job-environment).

### Dynamic installation from source

If the `ado` plugins or `ado-core` version you need are not on pypi you can install
them from source. There are two steps:

1. Build python wheels for `ado` and/or the required plugins
you need to install from source
2. Tell Ray to install the wheels as part of the Ray job submission

#### Building python wheels

> [!TIP]
>
> Repeat this step when the source code changes between Ray jobs
> and you want to include the changes.

##### Build the `ado` wheel

In the top-level of the `ado` repository:

  ```bash
  # Creates  `dist/` directory with the wheel. It will have a name like `ado_core-$VERSION-py3-none.whl`
  uv build -o dist --clear
  # Copy the wheel to the Ray job directory
  mv dist/*.whl $RAY_JOB_DATA_DIR
  ```

##### Build the plugin wheels

In the top-level of the plugins package, for example, one of the
subdirectories of "plugins/actuators/" in the `ado` repository, execute:

  ```bash
  : # Creates `dist/` directory in plugin directory with the wheel file. 
  : # It will have a name like `$PLUGIN-NAME-$VERSION-py3-none.whl`
  uv build -o dist --clear
  # Copy the wheel to the ray job directory
  mv dist/*.whl $RAY_JOB_DATA_DIR
  ```

#### Configuring installation of the wheels

Once you have built the required wheels, change directory to
`$RAY_JOB_DATA_DIR`.

Now, create a `ray_runtime_env.yaml` with a `uv` section, as in [quickstart example](#setting-the-ray-job-environment).
Here you can reference the wheels with the following format.

<!-- markdownlint-enable line-length -->

> [!IMPORTANT] RAY_RUNTIME_ENV_CREATE_WORKING_DIR
>
> Do not remove or modify the string ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}
> when specifying the wheel names. It is required before every wheel you want to
> upload and if it is changed the wheel installation will fail.
<!-- markdownlint-disable-next-line MD028 -->

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```yaml
uv: # One line for each wheel to install, in this example there is two. Be sure to check spelling.
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/$ADO_CORE.whl
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/$ADO_PLUGIN1.whl
env_vars: # See below
  ...
```

> [!NOTE] pypi packages
>
> You can specify pypi packages along with wheels in the `uv` section

## ray job submit options

The following sections explain the various flags and values
in the submission command line.

### Specifying the remote ray cluster to submit to: `--address`

To submit a job to a remote Ray cluster you need the address (URL) of its
dashboard. If the ray cluster is running on kubernetes or OpenShift you will
likely need to connect your laptop to this URL via a "port-forward".

For example with OpenShift you can do this with the following `oc` command in a
terminal other that the one you will submit the job from:

<!-- markdownlint-disable-next-line code-block-style -->
```commandline
oc port-forward --namespace $NAMESPACE svc/$RAY_SERVICE_NAME 8265
```

You will need to find out the name of the $NAMESPACE and the $RAY_SERVICE_NAME
from the administrator of the OpenShift cluster/the namespace. Once started the
ray cluster address will be `http://localhost:8265`

The port-forward command remains active until terminated, or until deemed
inactive. Once it stops you will not be able to get to the ray cluster until it
is restarted.

> [!IMPORTANT]
>
> `ray job submit` communicates to the ray cluster at the given URL using
> multiple protocols. This means that if only http traffic is allowed,
> `ray job submit` will not work. This is usually why you need a port-forward
> compared to, e.g. an OpenShift route.
<!-- markdownlint-disable-next-line MD028 -->

> [!TIP] Accessing the ray cluster dashboard
>
> You can navigate to the dashboard of the remote ray cluster by pasting the URL
> into your browser. From the dashboard, you can view running jobs, browse the
> logs of your job, see its workers, etc. You may also be able to reach this
> dashboard by a different URL that doesn't require port-forward to access.

### ray runtime environment: `runtime-env`

The environment of the ray job is given in a YAML file. An example is:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```yaml
uv: # One line for each wheel/package to install, in this example there is two. Be sure to check spelling.
  - ado-core
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/$ADO_PLUGIN1.whl
env_vars: # These envars are recommend. Some plugins may require others. Check plugin docs.
  PYTHONUNBUFFERED: "x" # Turns off buffering of the jobs logs. Useful if there is some error
  OMP_NUM_THREADS: "1" # Restricts the number of threads started by the python process in the job. If this is not set it can cause the ray job to exceed OpenShift node thread limits.
  OPENBLAS_NUM_THREADS: "1" # Same as above
  RAY_AIR_NEW_PERSISTENCE_MODE: "0" # Required for using the ray_tune operator
  #The following envars may be required or useful depending on your specific needs
  HOME: "/tmp" # Optional: Use if python code used by operation assumes $HOME is writable which it may not be
  LOGLEVEL: "WARNING" # Optional: Set this to get more/less debug logs from ado
```
<!-- markdownlint-enable line-length -->

For further details on what you can configure via this file see the
[ray documentation](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments).

### Other options

#### `--no-wait`

If specified `ray job submit` immediately disconnects from the remote job.
Otherwise, it stays connected until the job finishes.

If you want the job to keep running when you close your laptop, or be immune to
the port-forward deactivating, use this option.

#### `--working-dir`

Use this to specify the data to copy over with the ray job. Everything in this
directory and subdirectories is copied over and the ray job started in it. Here
this is the `$RAY_JOB_DATA_DIR`.
