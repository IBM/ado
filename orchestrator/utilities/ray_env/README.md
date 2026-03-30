# OrderedPip Ray Runtime Environment Plugin

The `OrderedPipPlugin` is a Ray RuntimeEnvPlugin that enables you to control the
build order of Python packages. This is useful when installing packages with
build-time dependencies. For example, `mamba-ssm` requires `torch` to be
installed before it can be built. We suggest using
`pip_install_options: ["--no-build-isolation"]`
which ensures that `pip` will use the same virtual environment to build and
install the wheels it builds.

## Overview

The plugin allows you to define multiple installation phases, where each phase
is executed sequentially. All phases install the wheels in the same virtual
environment, ensuring that packages installed in earlier phases are available
during the build of packages in later phases provided you also use
`--no-build-isolation`.

## Availability

The `OrderedPipPlugin` is pre-installed in ado Docker images and bundled with
`ado-core`.

## Enabling the Plugin

To enable the `OrderedPipPlugin`, set the `RAY_RUNTIME_ENV_PLUGINS` environment
variable before starting the Ray head and workers.

```bash
export RAY_RUNTIME_ENV_PLUGINS='[{"class":"orchestrator.utilities.ray_env.ordered_pip.OrderedPipPlugin"}]'
```

### Enabling in KubeRay

When deploying a RayCluster via KubeRay, add the environment variable to both
head and worker node configurations:

```yaml
head:
  containerEnv:
    - name: RAY_RUNTIME_ENV_PLUGINS
      value: '[{"class":"orchestrator.utilities.ray_env.ordered_pip.OrderedPipPlugin"}]'

worker:
  containerEnv:
    - name: RAY_RUNTIME_ENV_PLUGINS
      value: '[{"class":"orchestrator.utilities.ray_env.ordered_pip.OrderedPipPlugin"}]'
```

## Configuration Details

> [!IMPORTANT]
>
> Each entry in `phases` uses the **identical schema** as Ray's standard `pip`
> runtime environment field. If you know how to configure `pip`, you already
> know how to configure each phase in `ordered_pip`.

The `ordered_pip` runtime environment accepts a dictionary with a `phases` key:

- **`phases`**: Each phase can be one of:
  - A list of package names (e.g., `["torch==2.6.0"]`)
  - A dictionary with `packages` and optional `pip_install_options` fields
  - Any other valid `pip` specification format

## Usage Examples

### Using ordered_pip in Python Code

Here's a complete example showing how to use `ordered_pip` in a Ray task:

```python
import ray

@ray.remote(
    runtime_env={
        "ordered_pip": {
            "phases": [
                # Phase 1: Install PyTorch first
                ["torch==2.6.0"],
                # Phase 2: Install packages that depend on PyTorch during build
                {
                    "packages": ["mamba-ssm==2.2.5"],
                    # IMPORTANT.
                    # --no-build-isolation tells pip to build the wheel
                    # in the same venv where torch is already installed
                    "pip_install_options": ["--no-build-isolation"],
                }
            ]
        }
    }
)
def my_task():
    import torch
    import mamba_ssm
    return torch.__version__

result = ray.get(my_task.remote())
print(f"PyTorch version: {result}")
```

### Using ordered_pip with ray job submit

You can also use `ordered_pip` with `ray job submit` by providing a runtime
environment YAML file:

```yaml
# ray_runtime_env.yaml
ordered_pip:
  phases:
    # Phase 1: Install PyTorch first
    - packages:
        - torch==2.6.0
    # Phase 2: Install packages that depend on PyTorch during build
    - packages:
        - mamba-ssm==2.2.5
      pip_install_options:
        # IMPORTANT.
        # --no-build-isolation tells pip to build the wheel
        # in the same venv where torch is already installed
        - --no-build-isolation
```

Then submit your job with:

```bash
ray job submit --runtime-env-json ray_runtime_env.yaml -- python my_script.py
```

## Key Points

- **Sequential Execution**: Phases execute sequentially in the order specified
- **Shared Environment**: All phases reuse the same virtual environment
- **Build Isolation**: The `--no-build-isolation` flag is critical for packages
  that need build-time dependencies. It instructs pip to build wheels in the
  existing virtual environment rather than in an isolated one
- **Phase Order Matters**: Package order within a phase doesn't matter, but the
  order of phases does

## Integration with ado Actuators

Actuators like `SFTTrainer` automatically use `OrderedPipPlugin` when available
to ensure correct installation of their dependencies.

## Technical Details

For implementation details, see the source code in
[`ordered_pip.py`](./ordered_pip.py).
