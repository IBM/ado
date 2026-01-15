# Running experiments on single entities

> [!Note]
>
> The `run_experiment` script provides a quick and convenient way to execute
> any experiment from an actuator on a single point (entity) without the
> need to create a `discoveryspace`.
>
> This is especially useful for **rapid testing and debugging of actuator logic
> or experiment definitions**.

## Purpose

- **Quick Testing:** Instantly run an experiment on a single point/entity
  to verify the behavior of the actuator or experiment.
- **No `discoveryspace` required:** You do not need to create a `discoveryspace`
  or `operation` to use this tool.
- **Supports Local and Remote Execution:** You can run experiments either locally
  or through an `ado` REST API endpoint.

## Usage

<!-- markdownlint-disable line-length -->

```bash
run_experiment <point_file.yaml> [--remote <ENDPOINT>] [--timeout <SECONDS>] [--no-validate] \
               [--actuator-configuration-id <identifier>] [--verify-certs | --no-verify-certs] \
               [--request-timeout <timeout-in-seconds>]
```

<!-- markdownlint-enable line-length -->

- `<point_file.yaml>`: (REQUIRED) Path to a YAML file describing the point/entity
  and the experiments to run. This file should conform to the `ado` point
  definition schema.
- `--remote <ENDPOINT>`: If provided, the experiment will be executed
  through an `ado` REST API endpoint. If omitted, execution is local.
- `--timeout <SECONDS>`: Timeout for remote execution (default: 300 seconds).
- `--no-validate`: Skip entity validation before execution.
  This is useful if the experiment is not installed locally but is available remotely.
- `--actuator-configuration-id`: Actuator configuration identifiers to use
  in the experiment. Can be specified multiple times.
- `--verify-certs`, `--no-verify-certs`: Enable or disable SSL certificate verification
  of remote hosts. (default: `--no-verify-certs`).
- `--request-timeout`: Timeout for web requests. (default: 60s)

## Example

An example point.yaml is given below.
It defines the execution of the `nevergrad_opt_3d_test_func` from the `custom_experiments`
actuator on the point `x0=1, x1=3, x2=-1` (see
[optimizations with ado](../examples/best-configuration-search.md) for more
about this custom experiment).

The `entity` section contains a set of property/value pairs defining
the point to run the experiment on.
The `experiments` section is a list of experiment references detailing the
experiment to run.

```yaml
entity:
  x0: 1
  x1: 2
  x2: -1
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: nevergrad_opt_3d_test_func
```

To execute the experiment, run

```commandline
run_experiment example_point.yaml
```

## Limitations & Notes

- **Single Point Only:** This tool is designed for running experiments
  on a single point/entity.
  - For evaluating multiple points or running large-scale experiments, use a `discoveryspace`
    and the standard `ado` workflow.
- **No Metastore Tracking:** The results from `run_experiment`
  are **not** tracked or stored in the `ado` [metastore](../resources/metastore.md).
  - This means results are ephemeral and only available in the console output.
- **For Development & Debugging:** This script is best suited for
  actuator/experiment development, debugging, or quick checks — not for
  production or persistent experiment tracking.

## When to Use

- Testing new actuator or experiment implementations.
- Debugging experiment logic or entity validation.
- Quickly verifying actuator installation and configuration.
