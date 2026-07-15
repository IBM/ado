<!-- markdownlint-disable code-block-style -->

# Search a space with an optimizer

!!! abstract "Intermediate tutorial"

    This walkthrough assumes you are already familiar with the core `ado`
    workflow — custom experiments, discovery spaces, and operations. If you are
    new to `ado`, work through
    [Your first ado experiment](tutorials/density-example.md) first.

Finding the best entity according to some metric is a common task — for
example, finding the configuration of an LLM fine-tuning workload that
maximises throughput. `ado`'s `ray_tune` operator wraps the
[RayTune](https://docs.ray.io/en/latest/tune/index.html) framework and gives
you access to a wide range of optimization algorithms with no changes to how
you define your space or experiments.

This walkthrough uses a well-known benchmark — the
[Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function) in
three dimensions — so the focus stays on `ado` mechanics rather than the
domain.

We will:

1. Install the `ray_tune` operator and a custom experiment that wraps standard
   optimization test functions
2. Define a continuous three-dimensional discovery space
3. Run a Bayesian optimization over it
4. Inspect the best configuration found and the full trace of visited points

!!! warning "Prerequisites"

    - `ado-core` installed (`pip install ado-core`)
    - The example package cloned from GitHub (the wheel is not published to
      PyPI)

    Clone the repository if you have not already done so:

    ```bash
    git clone https://github.com/ibm/ado.git
    cd ado
    ```

    Then install the operator and experiment packages:

    ```bash
    pip install ado-ray-tune
    pip install examples/optimization_test_functions/custom_experiments
    ```

    All commands in this walkthrough are run from the **repository root**
    (`ado/`).

!!! example "TL;DR"

    Once the packages are installed:

    <!-- markdownlint-disable MD013 -->
    ```bash
    ado create space -f examples/optimization_test_functions/space.yaml --use-default-sample-store
    ado create operation -f examples/optimization_test_functions/operation_bayesopt.yaml --use-latest space
    ado show related operation --use-latest
    ado describe datacontainer $DATACONTAINER_ID
    ado show measurements operation --use-latest
    ```
    <!-- markdownlint-enable MD013 -->

## Step 1 — Install the required packages

### The `ray_tune` operator

The `ray_tune` operator is distributed as a separate package:

```bash
pip install ado-ray-tune
```

Confirm it is registered:

```bash
ado get operators
```

You should see `ray_tune` listed:

```text
Available operators by type:
┌───────┬─────────────┬─────────┬─────────┐
│ INDEX │ OPERATOR    │ VERSION │ TYPE    │
├───────┼─────────────┼─────────┼─────────┤
│ 0     │ random_walk │ 2.0.0   │ explore │
│ 1     │ ray_tune    │ 2.0.0   │ explore │
│ 2     │ rifferla    │ 2.0.0   │ modify  │
└───────┴─────────────┴─────────┴─────────┘
```

### The `nevergrad_opt_3d_test_func` custom experiment

The example ships a custom experiment that wraps a set of standard
optimization test functions — `discus`, `sphere`, `cigar`, `griewank`,
`rosenbrock`, `st1` — over a three-dimensional continuous space. See the
[nevergrad source](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/functions/corefuncs.py)
for the function definitions.

Install it from the example directory:

```bash
pip install examples/optimization_test_functions/custom_experiments/
```

Confirm `ado` can see it:

```bash
ado get experiments
```

You should see `nevergrad_opt_3d_test_func` listed under `custom_experiments`:

<!-- markdownlint-disable line-length -->

```text
┌───────┬────────────────────┬────────────────────────────┬─────────┐
│ INDEX │ ACTUATOR ID        │ EXPERIMENT ID              │ VERSION │
├───────┼────────────────────┼────────────────────────────┼─────────┤
│ 0     │ custom_experiments │ nevergrad_opt_3d_test_func │ 1.0.0   │
└───────┴────────────────────┴────────────────────────────┴─────────┘
```

<!-- markdownlint-enable line-length -->

Inspect the experiment interface:

```bash
ado describe experiment nevergrad_opt_3d_test_func
```

```terminaloutput
Identifier: custom_experiments.nevergrad_opt_3d_test_func@1.0.0
Version: 1.0.0

Required Inputs:

   Constitutive Properties:
    ─────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: x0
     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE

    ─────────────────────────────────────────────────────────────────────────────────────────────────
    ─────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: x1
     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE

    ─────────────────────────────────────────────────────────────────────────────────────────────────
    ─────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: x2
     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE

    ─────────────────────────────────────────────────────────────────────────────────────────────────

Optional Inputs and Default Values:

    ─────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: num_blocks
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [1, 10]

     Default value: 1
    ─────────────────────────────────────────────────────────────────────────────────────────────────
    ─────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: name
     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['discus', 'sphere', 'cigar', 'griewank', 'rosenbrock', 'st1']

     Default value: 'rosenbrock'
    ─────────────────────────────────────────────────────────────────────────────────────────────────

Outputs:
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
   nevergrad_opt_3d_test_func@v1-function_value
 ───────────────────────────────────────────────────────────────────────────────────────────────────────
```

The three required inputs (`x0`, `x1`, `x2`) map to entity-space dimensions.
The optional `name` parameter selects which test function to evaluate; it
defaults to `rosenbrock`. The single output, `function_value`, is the metric
we will optimize against.

## Step 2 — Define a discovery space

The file `examples/optimization_test_functions/space.yaml` defines a
three-dimensional continuous space spanning `[-10, 10]` in each dimension,
with `nevergrad_opt_3d_test_func` (defaulting to `rosenbrock`) as the
measurement:

```bash
ado create space -f examples/optimization_test_functions/space.yaml --use-default-sample-store
```

```text
Success! Created space with identifier: $DISCOVERY_SPACE_IDENTIFIER
```

Inspect the space to confirm the entity and measurement spaces are correct:

```bash
ado describe space --use-latest
```

```terminaloutput
Identifier: 'space-3d6891-default'

Entity Space:

   Space with non-discrete dimensions. Cannot count entities

   Continuous properties:

      name ┃ range
     ━━━━━━╋━━━━━━━━━━━
      x2   ┃ [-10, 10]
      x1   ┃ [-10, 10]
      x0   ┃ [-10, 10]


Measurement Space:

   Experiments:

      base identifier                               ┃ required major version ┃ parameterization
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━
      custom_experiments.nevergrad_opt_3d_test_func ┃ v1                     ┃ None

    ───────────────────────────────── nevergrad_opt_3d_test_func@v1 ─────────────────────────────────
     Expected Interface (from v1.0.0)

     Inputs:

        parameter  ┃ type     ┃ value      ┃ parameterized
       ━━━━━━━━━━━━╋━━━━━━━━━━╋━━━━━━━━━━━━╋━━━━━━━━━━━━━━━
        x0         ┃ required ┃ None       ┃ na
        x1         ┃ required ┃ None       ┃ na
        x2         ┃ required ┃ None       ┃ na
        num_blocks ┃ optional ┃ 1          ┃ False
        name       ┃ optional ┃ rosenbrock ┃ False

     Outputs:

        target property
       ━━━━━━━━━━━━━━━━━
        function_value

    ─────────────────────────────────────────────────────────────────────────────────────────────────


Sample Store identifier: default
```

!!! tip

    The entity space has **continuous** dimensions, so `ado` cannot enumerate
    entities in advance — the optimizer proposes points at runtime. This also
    means memoization rarely saves work here, since the probability of revisiting
    an identical point is negligible.

## Step 3 — Run an optimization

The file `examples/optimization_test_functions/operation_bayesopt.yaml`
configures a Bayesian optimization run of 40 steps via RayTune:

<!-- markdownlint-disable MD013 -->
```bash
ado create operation -f examples/optimization_test_functions/operation_bayesopt.yaml --use-latest space
```
<!-- markdownlint-enable MD013 -->

RayTune logs progress to the terminal as it runs. When it finishes you will see
the operation summary:

```yaml
Space ID: space-3d6891-default
Sample Store ID:  default
Operation:
 config:
  actuatorConfigurationIdentifiers: []
  metadata: {}
  operation:
    module:
      operationType: explore
      operatorName: ray_tune
      operatorVersion: 2.0.0
    parameters:
      orchestratorConfig:
        failed_metric_value: NaN
        metric_format: target
        result_dump: none
        single_measurement_per_property: true
      runtimeConfig: {}
      tuneConfig:
        max_concurrent_trials: 2
        metric: function_value
        mode: min
        num_samples: 40
        search_alg:
          name: bayesopt
          params: {}
  spaces:
  - space-3d6891-default
created: '2026-07-09T12:19:14.569324Z'
identifier: ray_tune@2.0.0-bayesopt-bbfbc5
kind: operation
```

### Choosing the target metric

The `metric` field in the operation YAML tells `ray_tune` which experiment
output to optimize, and `mode` specifies the direction (`min` or `max`).
In this example there is a single output (`function_value`), but in spaces
with multiple experiments or multi-output experiments you will need to be
explicit:

<!-- markdownlint-disable line-length -->

```yaml
parameters:
  tuneConfig:
    metric: "function_value"  # the output property to optimize
    mode: "min"
    num_samples: 40
    max_concurrent_trials: 2
    search_alg:
      name: bayesopt
```

<!-- markdownlint-enable line-length -->

## Step 4 — Inspect the results

### Best configuration found

`ray_tune` persists the best configuration it found as a `datacontainer`
resource linked to the operation. Retrieve its identifier:

```bash
ado show related operation --use-latest
```

```terminaloutput
datacontainer
  - datacontainer-391e170c
discoveryspace
  - space-3d6891-default
samplestore
  - default
```

Inspect the data container:

```bash
ado describe datacontainer $DATACONTAINER_ID
```

```terminaloutput
Identifier: datacontainer-391e170c

 ───────────────────────────────────────────── Basic Data ──────────────────────────────────────────────

    Label: 'best_result'
    {
        'config': {'x2': -2.241552006735698, 'x1': -0.18479435692216645, 'x0': 0.957477270215893},
        'metrics': {
            'function_value': 640.6298323077821,
            ...
        },
        'error': None
    }

 ───────────────────────────────────────────────────────────────────────────────────────────────────────
```

The best point found by the optimizer was `(x0≈0.957, x1≈-0.185, x2≈-2.242)`
with a `function_value` of approximately 640.63.

!!! tip

    The global minimum of the 3-d Rosenbrock function is 0, located at
    `(1, 1, 1)`. Running more samples (`num_samples`) or switching to a
    stronger optimizer will get you closer.

### Full trace of visited configurations

To see every point the optimizer evaluated, along with its measured
`function_value`:

```bash
ado show measurements operation --use-latest
```

To see all measurements accumulated on the space across all operations (not
just this run):

```bash
ado show measurements space --use-latest
```

### Retrieving the full operation configuration

If you need to review or reproduce the exact options used in a run:

```bash
ado get operation $OPERATION_IDENTIFIER -o yaml
```

## Step 5 — Parameterizable experiments

`nevergrad_opt_3d_test_func` is a **parameterizable experiment**: it has
optional inputs (`name`, `num_blocks`) with default values that you can fix
at space-creation time.

Parameterization is set in the `discoveryspace` YAML, not in the operation.
For example, to optimize the `cigar` function instead of `rosenbrock`:

```yaml
- actuatorIdentifier: custom_experiments
  experimentIdentifier: nevergrad_opt_3d_test_func
  experimentVersion: 1.0.0
  parameterization:
    - value: "cigar"
      property:
        identifier: "name"
```

The resulting parameterized experiment gets a distinct identifier that encodes
the parameterization, keeping it separate from the base experiment in the
sample store.

!!! tip

    You cannot change the parameterization of an experiment in an existing
    discovery space — doing so would change what is being measured, invalidating
    the space's identity. Create a new space with the updated parameterization.

## Summary

| Step | What you did | `ado` concept |
| :--- | :----------- | :------------ |
| 1 | Installed `ray_tune` and the custom experiment | Operator / custom experiment |
| 2 | Defined a continuous 3-D discovery space | Discovery space |
| 3 | Ran a Bayesian optimization for 40 steps | Operation / `ray_tune` operator |
| 4 | Retrieved the best configuration and the full trace | `datacontainer` / `ado show measurements` |
| 5 | Learned how to fix optional experiment inputs | Parameterizable experiments |

## Going further

Try extending this example:

- **Change optimizer** — the file `operation_nevergrad.yaml` uses the CMA
  optimizer from nevergrad instead of Bayesian optimization
- **Different result views** — compare `ado show measurements space` with
  `ado show measurements operation` to understand how space-level and
  operation-level views differ
- **Modify the entity space** — change the bounds or number of dimensions to
  see how the optimizer adapts
- **Tune optimizer options** — adjust `num_samples`, `max_concurrent_trials`,
  or `search_alg`; see
  [the ray_tune operator documentation](/ado/user-guide/operators/ray-tune/)
  for the full configuration reference
  <!-- codespell:ignore discus -->
- **Parameterize the experiment** — optimize the `discus` function by
  parameterizing `nevergrad_opt_3d_test_func`; observe how the space
  description changes
- **Discretize the space** — switch to a discrete entity space and verify that
  memoization reuses measurements when the optimizer revisits a point
- **Search across test functions** — it is possible to include `name` as an
  entity-space dimension and find which test function has the smallest minimum
  in a single operation

### Extending `nevergrad_opt_3d_test_func`

The experiment can be extended to cover additional functions or more
dimensions. See the
[custom experiments documentation](/ado/developer-guide/creating-custom-experiments/)
for guidance.

!!! warning

    If you modify what the experiment computes, update its name accordingly.
    An experiment with the same name as an existing one but different behaviour
    will corrupt stored measurements.

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable no-inline-html -->
<!-- markdownlint-disable MD046 -->
<!-- prettier-ignore-start -->

<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Creating custom objective functions**

    ---

    Define a custom experiment that uses outputs from another experiment as
    its inputs — a common pattern for derived metrics.

    [Search a space based on a custom objective function :octicons-arrow-right-24:](search-custom-objective.md)

- :octicons-workflow-24:{ .lg .middle } **Discovering important entity space dimensions**

    ---

    Use `ado` to identify which entity space dimensions most influence a
    target metric.

    [Identify the important dimensions of a space :octicons-arrow-right-24:](lhu.md)

</div>

<!-- prettier-ignore-end -->
<!-- markdownlint-enable MD046 -->
<!-- markdownlint-enable no-inline-html -->
<!-- markdownlint-enable line-length -->
