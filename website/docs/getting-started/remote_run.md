<!-- markdownlint-disable-next-line first-line-h1 -->
!!! info end

    It's recommended to read the documentation on
    [working with operators](../operators/working-with-operators.md) and
    [operation resources](../resources/operation.md) before trying to run `ado`
    remotely.

The following sections explain how to run an `ado` operation on a remote ray
cluster.

## Why run `ado` on a remote ray cluster?

ado operations, normally started with
[ado create operation](../resources/operation.md), can benefit from running on,
or even require running on, a remote ray cluster.

This is because certain operations require (or can use) more compute resource
than is available on your laptop. This is usually because one or both of the
following is true:

- the operation can leverage the fact that `ado` is built on `Ray` to run large
  numbers of distributed tasks
- the operation requires access to large amounts of compute resources, like
  GPUs.

Often it is [explore operations](../operators/explore_operators.md) that require
remote execution, due to the requirements of the actuators performing the
measurements in the explore process. However, any `ado` operation can be run on
a remote ray cluster.

### Running `ado` remotely versus using an actuator that can run remotely

Some actuators are capable of spawning jobs on remote clusters, which may be
remote ray clusters. This is different to running `ado` remotely as in this case
the `ado` main process is still running on your laptop, and if you close your
laptop the process will stop.

When running ado remotely the `ado` main process runs on the remote cluster.

## Getting ready

First, create an empty directory to store all necessary files that need to be
uploaded to the remote ray cluster for the `ado` command to run. In the
following we will refer to this directory as `$RAY_JOB_DATA_DIR`.

## Installing `ado` and required plugins on a remote ray cluster (from source)

If `ado` or the required plugins are not present on the remote ray cluster, then
they can be installed as part of the ray job submission. The simplest approach
is installing from source. For this you need to have
[cloned the ado repository](install.md#installing-ado) and
[cloned the plugin repositories](install.md#installing-plugins)(if they are not
in the `ado` repository).

Then there are two steps:

1. Build python wheels for `ado` and any required plugins
2. Tell ray to install the wheels as part of ray-job submission

### Building the python wheels

!!! info

    Repeat this step only if the source code changes between ray jobs
    and you want to include the changes.

=== "Build `ado` wheel"

    In the top-level of the `ado` repository:

    ```commandline
    rm -rf dist/ build/
    python -m build -w
    mv dist/*.whl `$RAY_JOB_DATA_DIR
    ```

    - First command removes any previous build artifacts and wheels. This prevents
      issues with old files being included in the new wheel
    - Second command creates a `dist/` directory with the wheel. It will have a name
      like `ado_core-$VERSION-py3-none.whl`
    - Last command copies the wheel to the directory you made

=== "Build the plugin wheels"

    In the top-level of the plugins repository e.g. in one of the orchestrator
    repositories' `plugins/actuators/$ACTUATOR` directories, execute:

    ```commandline
    rm -rf dist/ build/
    python -m build -w
    mv dist/*.whl `$RAY_JOB_DATA_DIR
    ```

    - First command removes any previous build artifacts and wheels. This prevents
      issues with old files being included in the new wheel
    - Second command creates a `dist/` directory with the wheel. It will have a name
      like `$pluginname-$VERSION-py3-none.whl`
    - Last command copies the wheel to the directory you made

#### Configuring installation of the wheels

Once you have built the required wheels, change directory to
`$RAY_JOB_DATA_DIR`.

Now, create a `ray_runtime_env.yaml` file that will direct ray to install the
wheels when the job is submitted. The section of this file relevant to wheel
installation is under the `pip` field which is a list of packages for pip to
install. In our case it is the wheels we just created e.g.

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```yaml
pip: # One line for each wheel to install, in this example there is two. Be sure to check spelling.
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/$ADO_CORE.whl
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/$ADO_PLUGIN1.whl
env_vars: # See below
  ...
```
<!-- markdownlint-enable line-length -->

See [ray runtime environment](#ray-runtime-environment-runtime-env) for more
information on this file.

!!! warning

    Do not remove or modify the string ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}
    when specifying the wheel names. It is required before every wheel you want to
    upload and if it is changed the wheel installation will fail.

## Submitting the `ado` operation

After completing the previous section, create the `ado` input files in
`$RAY_JOB_DATA_DIR` (or copy them there).

This will be usually the following two YAML files:

- A YAML file describing [the context](../resources/metastore.md) to use for the
  operation.
- A YAML file describing [the operation](../resources/operation.md) to create.

We will refer to the first file as `context.yaml` and the second as
`operation.yaml`, although they can have any names.

Then the execution command will look like:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```commandline
ray job submit --no-wait --address http://localhost:8265  --working-dir . --runtime-env ray_runtime.yaml -v -- \
  ado -c context.yaml create operation -f operation.yaml
```
<!-- markdownlint-enable line-length -->

The following sections explain the various flags and values in this command line

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

!!! important

    `ray job submit` communicates to the ray cluster using different protocols via
    the given URL. This means if only http is allowed to be sent to the URL
    `ray job submit` will not work. This is usually why you need a port-forward
    compared to, say, an OpenShift route.

!!! note

    You can navigate to the dashboard of the remote ray cluster by pasting the URL
    into your browser. From the dashboard, you can view running jobs, browse the
    logs of your job, see its workers etc. You may also be able to reach this
    dashboard by a different URL that doesn't require port-forward to access.

### ray runtime environment: `runtime-env`

The environment of the ray job is given in a YAML file. An example is:

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line code-block-style -->
```yaml
pip: # One line for each wheel to install, in this example there is two. Be sure to check spelling.
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/$ADO_CORE.whl
  - ${RAY_RUNTIME_ENV_CREATE_WORKING_DIR}/$ADO_PLUGIN1.whl
env_vars: # These envars are recommend. Some plugins may require others. Check plugin docs.
  PYTHONUNBUFFERED: "x" # Turns of buffering of the jobs logs. Useful if there is some error
  OMP_NUM_THREADS: "1" # Restricts the number of threads started by the python process in the job. If this is not set it can cause the ray job to exceed OpenShift node thread limits.
  OPENBLAS_NUM_THREADS: "1" # Same as above
  RAY_AIR_NEW_PERSISTENCE_MODE: "0" # Required for using the ray_tune operator
  #The following envars may be required or useful depending
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

#### `--working-dr`

Use this to specify the data to copy over with the ray job. Everything in this
directory and subdirectories is copied over and the ray job started in it. Here
this is the `$RAY_JOB_DATA_DIR`.
