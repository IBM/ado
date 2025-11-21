# Optimizations with ado

> [!NOTE]
>
> This example demonstrates:
>
> 1. Creating and installing custom experiments
>
> 2. Performing optimizations with `ray_tune`
>
> 3. Parameterizable and parameterized experiments
<!-- markdownlint-disable-next-line no-blanks-blockquote -->
> [!NOTE]
>
> We recommend trying the
> [talking a random walk example](https://ibm.github.io/ado/examples/random-walk/)
> first to get familiar with some basic concepts and commands.

## The scenario

**Finding the best entity, or point, according to some metric, is a common
task.** For example, finding the configuration of an LLM fine-tuning workload
that gives the highest throughput. Many optimization methods have been developed
to address this problem and you can access a variety of them via `ado`'s
`ray_tune` operator, which provides access to the RayTune framework.

**This example demonstrates running optimizations in `ado`** using the problem
of finding the minimum of standard optimization test functions.

> [!CAUTION]
>
> The commands below assume you are in the directory
> `examples/optimization_test_functions` in **the ado source repository**. See
> [the instructions for cloning the repository](/ado/getting-started/install/#__tabbed_1_3).

## Setup

### Install the ray_tune ado operator

If you haven't already installed the ray_tune operator, run:

```commandline
pip install ado-ray-tune
```

then, executing

```commandline
ado get operators
```

should show an entry for `ray_tune` like below

```commandline
Available operators by type:
      OPERATOR     TYPE
0  random_walk  explore
1     ray_tune  explore
```

### Install the custom `nevergrad_opt_3d_test_func` experiment

The `nevergrad_opt_3d_test_func` experiment enables measuring the following
optimization test functions on a 3d space: 'discus', 'sphere', 'cigar',
'griewank', 'rosenbrock', 'st1'. See the
[nevergrad docs](https://github.com/facebookresearch/nevergrad/blob/main/nevergrad/functions/corefuncs.py)
for definitions of these functions.

To install it:

```bash
pip install custom_experiments/
```

after this running `ado get actuators --details` should show the following line:

<!-- markdownlint-disable line-length -->
```commandline
1   custom_experiments  CustomExperiments                             nevergrad_opt_3d_test_func       True
```
<!-- markdownlint-enable line-length -->

and `ado describe experiment nevergrad_opt_3d_test_func` should output

```text
Identifier: custom_experiments.nevergrad_opt_3d_test_func

Required Inputs:
  Constitutive Properties:
      x0
      Domain:
        Type: CONTINUOUS_VARIABLE_TYPE

      x1
      Domain:
        Type: CONTINUOUS_VARIABLE_TYPE

      x2
      Domain:
        Type: CONTINUOUS_VARIABLE_TYPE


Optional Inputs and Default Values:
  name
  Domain:
    Type: CATEGORICAL_VARIABLE_TYPE
    Values: ['discus', 'sphere', 'cigar', 'griewank', 'rosenbrock', 'st1']


  Default value: rosenbrock

  num_blocks
  Domain:
    Type: DISCRETE_VARIABLE_TYPE Interval: 1.0 Range: [1, 10]

  Default value: 1


Outputs: nevergrad_opt_3d_test_func-function_value
```

## Running the example

### Set active context

You can use any context, for examples `ado`'s default local context:

```commandline
ado context local
```

### Create the discovery space

The file "space.yaml" contains an example space describing the rosenbrock
function in 3d, from [-10,10] in each dimension. To create the space execute:

```commandline
ado create space -f space.yaml --use-default-sample-store
```

> [!NOTE]
>
> `samplestores` can store samples and measurements from multiple different
> experiments and `discoveryspaces`.

This will output a `discoveryspace` id you can use to run an optimization
operation.

Assuming you did not modify `space.yaml`, running
`ado describe space --use-latest` will output (identifiers will
be different):

```text
Identifier: space-f529ab-85161d

Entity Space:

  Space with non-discrete dimensions. Cannot count entities
  Continuous properties:
      name      range
    0   x2  [-10, 10]
    1   x1  [-10, 10]
    2   x0  [-10, 10]


Measurement Space:

                                         experiment  supported
  0  custom_experiments.nevergrad_opt_3d_test_func       True


  'custom_experiments.nevergrad_opt_3d_test_func'


  Inputs:
      parameter      type       value parameterized
  0          x0  required        None            na
  1          x1  required        None            na
  2          x2  required        None            na
  3        name  optional  rosenbrock         False
  4  num_blocks  optional           1         False


  Outputs:

    target property
  0  function_value



Sample Store identifier: '85161d'
```

Here we see,

- the Entity Space is a 3-dimensional space, with continuous dimensions,
  spanning [-10,10] in each dimension.
- the Measurement Space, describing the measurements to apply to each point in
  the space, contains one experiment - in this case the
  `custom_experiments.nevergrad_opt_3d_test_func`.
- The `custom_experiments.nevergrad_opt_3d_test_func` experiment defines one
  metric, `function_value`.
- Since the default function used by
  `custom_experiments.nevergrad_opt_3d_test_func` is `rosenbrock`, for a given
  point `function_value` will be the value of the 3d `rosenbrock` function at
  that point.

Also try:

```commandline
ado get spaces
```

This will output a list of the spaces created. If this is the first time you
are following this example it will contain one entry, the identifier of the
space you just created above.

### Run an optimization

The file `operation_bayesopt.yaml` is an example of running
[Bayesian Optimization](https://bayesian-optimization.github.io/BayesianOptimization)
via RayTune. To run execute the following:

```commandline
ado create operation -f operation_bayesopt.yaml --use-latest space
```

This will run the optimization for 40 steps. You will see a lot of information
from RayTune on the progress of the optimization, finishing with a description
of the operation like below:

```yaml
Space ID: space-3fbaad-c3a5f6
Sample Store ID:  c3a5f6
Operation:
 config:
  actuatorConfigurationIdentifiers: []
  metadata: {}
  operation:
    module:
      moduleClass: RayTune
      moduleName: ado_ray_tune.operator
      modulePath: .
      moduleType: operation
    parameters:
      tuneConfig:
        max_concurrent_trials: 2
        metric: function_value
        mode: min
        num_samples: 40
        search_alg:
          name: bayesopt
  spaces:
  - space-3fbaad-c3a5f6
created: '2025-09-06T10:40:58.158982Z'
identifier: raytune-1.0.2.dev11+1c62218-bayesopt-b7f779
kind: operation
metadata:
  entities_submitted: 40
  experiments_requested: 40
operationType: search
operatorIdentifier: raytune-1.0.2.dev11+1c62218
status:
- event: created
  recorded_at: '2025-09-06T10:40:47.558298Z'
- event: added
  recorded_at: '2025-09-06T10:40:58.168519Z'
- event: started
  recorded_at: '2025-09-06T10:40:58.180530Z'
- event: updated
  recorded_at: '2025-09-06T10:40:58.180540Z'
- event: finished
  exit_state: success
  message: Ray Tune operation completed successfully
  recorded_at: '2025-09-06T10:42:42.804220Z'
- event: updated
  recorded_at: '2025-09-06T10:42:43.857310Z'
version: v1
```

### Specifying the property to optimize

In this case there is one experiment with one property in the measurement space,
so there is only one choice for the property to optimize against i.e.
`function_value`. However, usually an experiment will measure many properties
and there may be many measurements.

The target property to optimize against is set by the `metric` field, under the
operations `parameters` field.

<!-- markdownlint-disable line-length -->
```yaml
parameters:
  tuneConfig:
    metric: "function_value" # The experiment property/metric to optimize against
    mode: "min"
    num_samples: 40
    max_concurrent_trials: 4 # This is set for debugging. Increase if you want multiple measurements at once.
    search_alg:
      name: nevergrad
      params:
        optimizer: "CMA"
```
<!-- markdownlint-enable line-length -->

## See the optimization results

### Best configuration found

The `ray_tune` operation will create a `datacontainer` resource containing
information on the best configuration found.

To get the id of the `datacontainer` related to the `operation` use:

```commandline
ado show related operation --use-latest
```

This will output something like:

```commandline
datacontainer
  - datacontainer-d6a6501b
discoveryspace
  - space-047b6a-f60613
```

To see the best point found (and in general the contents of the datacontainer)
use the `describe` CLI command:

```commandline
ado describe datacontainer $DATACONTAINER_ID
```

In this case the output will be something like:

```commandline
Identifier: datacontainer-d6a6501b
Basic Data:

  Label: best_result

  {'config': {'x2': -1.1192905253425014,
    'x1': 2.081208150586974,
    'x0': 0.5621591414422049},
   'metrics': {'function_value': 20.788056393697595,
    'timestamp': 1756804287,
    'checkpoint_dir_name': None,
    'done': True,
    'training_iteration': 1,
    'trial_id': '7a7153ed',
    'date': '2025-09-02_10-11-27',
    'time_this_iter_s': 1.0576610565185547,
    'time_total_s': 1.0576610565185547,
    'pid': 52036,
    'hostname': 'Michaels-MacBook-Pro-2.local',
    'node_ip': '127.0.0.1',
    'config': {'x2': -1.1192905253425014,
     'x1': 2.081208150586974,
     'x0': 0.5621591414422049},
    'time_since_restore': 1.0576610565185547,
    'iterations_since_restore': 1,
    'experiment_tag': '40_x0=0.5622,x1=2.0812,x2=-1.1193'},
   'error': None}
```

We can see here that the point found is
`{'x2': -1.1192905253425014, 'x1': 2.081208150586974, 'x0': 0.5621591414422049}`
where `function_value` was ~20.8.

### Configurations visited

To see the configurations visited during the optimization you just ran, execute:

```commandline
ado show entities operation --use-latest
```

This will output a dataframe containing the results of that operation.

### Operation resource YAML

If at any point you want to see the details for an operation, for example the
options used, execute:

```commandline
ado get operation $OPERATION_IDENTIFIER -o yaml
```

Where `$OPERATION_IDENTIFIER` is the identifier of the operation you just ran.
This will output the details of this operation in YAML format - this will be the
same YAML as shown in the previous section.

## Parameterizable experiments

<!-- markdownlint-disable descriptive-link-text -->
The `nevergrad_opt_3d_test_func` is an example of a **parameterizable
experiment**. A parameterizable experiment has optional inputs that have default
values. In this case the optional inputs are `name` and `num_blocks` which you
can see are listed in the output of `ado describe experiment`
[here](#install-the-custom-nevergrad_opt_3d_test_func-experiment). In particular
the "name" parameter defines the optimization test function the experiment will
use and its default value is 'rosenbrock'.
<!-- markdownlint-enable descriptive-link-text -->

If you want to set a different value for an optional parameter of an experiment
you do this when creating the `discoveryspace`. For example to set the function
to `cigar` you would write (snippet from full `discoveryspace` yaml)

```yaml
- actuatorIdentifier: custom_experiments
  experimentIdentifier: nevergrad_opt_3d_test_func
  parameterization:
    - value: "cigar"
      property:
        identifier: "name"
```

When you set an optional property of a parameterizable experiment we call the
result a parameterized experiment.

> [!NOTE]
>
> You can't change the parameterization of an experiment in an existing
> `discoveryspace` as this changes the measurement and hence the entire space.
> Using an experiment with a new parameterization requires creating a new
> `discoveryspace`.

## Exploring Further

Try the following:

- _change optimizer_: The file `optimization_nevergrad.yaml` shows using the CMA
  optimizer from nevergrad. Modify and run in the same way as the Ax example
- _different results views_: Use `ado show entities space $SPACE_ID` where
  `SPACE_ID` is the identifier of the space the operations run on. Compare to
  the output of `ado show entities operation`
- _modify the entity space_: Extending or limiting the dimensions of the entity
  space considered
- _change optimizer options_: Change the optimization options and run another
  optimization. See
  [the ray tune operator documentation](/ado/operators/optimisation-with-ray-tune/)
  for details and further examples on what can be configured.
<!-- markdownlint-disable-next-line line-length -->
- _parameterize the experiment_: Perform an optimization on the `discus` <!-- codespell:ignore discus -->
  function - this involves parameterizing the
  `nevergrad_opt_3d_test_func`.
  - See how this changes the description of `discoveryspace`.
- _discretize the space_: Run the optimization on a discretized version of one
  of the functions and see if memoization works. **Hint**: change the entity
  space.
- _find the minimum across all test-functions_: It's possible to search for
  which test function has the minimum value across the entity space in a single
  run. Hint: you can use any experiment parameters as entity-space dimensions.

### Extending the `nevergrad_opt_3d_test_func` experiment

The `nevergrad_opt_3d_test_func` experiment can be expanded to include more
functions or options. It is also straightforward to add custom experiment for
more dimensions. See
[the documentation for custom experiments](/ado/actuators/creating-custom-experiments/)
to find out more.

> [!IMPORTANT] If you change what the function does consider the name of the
> experiment. If it is not changed in some way the experiment will have the same
> name as an existing used experiment but do something different which is
> problematic.

## Takeaways

- **create-explore-view pattern**: A common pattern in `ado` is to create a
  `discoveryspace` to describe a set of points to measure, create `operations`
  on it to explore or analyse it, and then view the results
- **optimization**: `ado` provides an interface to RayTune allowing all the
  optimizers supported by RayTune to be used to explore `discoveryspaces`
- **parameterized experiments**: Experiments can have optional parameters you
  can set to change what they do. When experiment is parameterized it will have
  a different id including the parameterization to differentiate it from the
  base experiment.
- **custom experiments**: You can add your own Python functions as experiments
  using `ado`'s custom experiments feature.
- **continuous dimensions**: `ado` supports `discoveryspaces` with continuous
  dimensions - however in this case memoization is unlikely to provide benefit
  as the chances of visiting the same space twice are remote.
