<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable first-line-h1 -->

A `discoveryspace` describes a set of `entities` along with the experiments to
apply to them. It has three parts:

- an [`entityspace`](../concepts/entity-spaces.md): the `entities` you want to
  measure
- a [`measurementspace`](../concepts/actuators.md#measurement-space): the
  experiments used to measure them
- a [`samplestore`](sample-stores.md): where the result of measurements on `entities`
  are stored

>[!NOTE]
>
> For more details see [concept of a Discovery Space](../concepts/discovery-spaces.md)

## Quickstart

The quickest route to a valid `discoveryspace` is to start from the experiments
you want to run. First, list the available experiments:

```commandline
ado get experiments
```

<!-- markdownlint-disable line-length -->

```terminaloutput
┌───────┬──────────────────┬─────────────────────────────────────┬─────────┐
│ INDEX │ ACTUATOR ID      │ EXPERIMENT ID                       │ VERSION │
├───────┼──────────────────┼─────────────────────────────────────┼─────────┤
│ 0     │ SFTTrainer       │ finetune_full_benchmark-v1.0.0      │ None    │
│ ...   │ ...              │ ...                                 │ ...     │
│ 11    │ custom_experi... │ nevergrad_opt_3d_test_func          │ 1.0.0   │
│ ...   │ ...              │ ...                                 │ ...     │
│ 30    │ vllm_performance │ vllm-bench-endpoint                 │ 1.0.0   │
└───────┴──────────────────┴─────────────────────────────────────┴─────────┘
```

<!-- markdownlint-enable line-length -->

Then generate a `discoveryspace` YAML from the experiment you picked:

```commandline
ado template space --from-experiment vllm-bench-endpoint --output-file space.yaml
```

See [`ado template`](../cli-reference/index.md#ado-template) for more options.

The generated YAML contains an `entityspace` holding every constitutive
property the experiment requires (required input parameters),
each with the full domain the experiment supports,
and a `measurementspace` referencing the experiment.

>[!NOTE] Optional Properties
>
> `ado template` does not add optional input parameters
> of the experiment to the entity space.
> You can do this manually as described in
> [parameterizing experiments](#parameterizing-experiments).

An example YAML file is:

```yaml
entitySpace:
  - identifier: request_rate
    metadata:
      description: The number of requests to send per second
    propertyDomain:
      domainRange:
        - -1
        - 1000
      interval: 1
      probabilityFunction:
        identifier: uniform
    propertyType: CONSTITUTIVE_PROPERTY_TYPE
  - identifier: model
    propertyDomain:
      probabilityFunction:
        identifier: uniform
      values:
        - meta-llama/Llama-3.1-8B-Instruct
        - ibm-granite/granite-3.3-8b-instruct
        - openai/gpt-oss-20b
    propertyType: CONSTITUTIVE_PROPERTY_TYPE
  - identifier: endpoint
    propertyDomain:
      probabilityFunction:
        identifier: uniform
      values:
        - http://localhost:8000
    propertyType: CONSTITUTIVE_PROPERTY_TYPE
experiments:
  - actuatorIdentifier: vllm_performance
    experimentIdentifier: vllm-bench-endpoint
    experimentVersion: 1.0.0
metadata: {}
sampleStoreIdentifier: default
```

You can edit the YAML file e.g. narrow each `propertyDomain` to
the values of interest. Validate the result at any point with:

```commandline
ado create space -f space.yaml --dry-run
```

Then create the space:

```commandline
ado create space -f space.yaml
```

>[!NOTE]
>
> `ado template` outputs all the fields that have values
> including defaults and per-property metadata.
> These can be safely removed to create a more streamlined YAML.
> See
> [defining property domains](#defining-property-domains)
> for more.

## Structure of the `discoveryspace` YAML configuration

A `discoveryspace` configuration has four fields. An example is given below.

<!-- markdownlint-disable line-length -->

```yaml
sampleStoreIdentifier: source_abc123 # OPTIONAL: The id of the sample store to use
entitySpace: # A list of constitutive properties
  - identifier: my_property1 # The id of the first dimension/constitutive property of the space
    propertyDomain: # Defines the values my_property1 can take
      ... # Property domain fields
  - identifier: my_property2
    propertyDomain:
       ...
experiments: # A list of experiments. The measurementspace of this discovery space
  - actuatorIdentifier: someactuator # The id of the actuator that contains the experiment
    experimentIdentifier: experiment_one # The id of the experiment to execute
    experimentVersion: 1.0.0 # The version of the experiment. If omitted it means the version is "None"
metadata:
  description: "This is an example discovery space"
  name: exampleSpace
```

<!-- markdownlint-enable line-length -->

- **`entitySpace`**: the dimensions of the space, one entry per constitutive
  property. See [defining the entityspace](#defining-the-entityspace).
- **`experiments`**: the experiments that make up the `measurementspace`. See
  [defining the measurementspace](#defining-the-measurementspace).
- **`sampleStoreIdentifier`**: the `samplestore` holding the data. Defaults to
  `default` if not given. See [choosing the samplestore](#choosing-the-samplestore).
- **`metadata`**: a `name`, a `description` and any `labels` you want to attach
  to the space.

If there are errors or inconsistencies in the space definition the `ado create space`
command will output an error.

## Defining the `entityspace`

The `entityspace` is a list of constitutive properties, each with the domain of
values it takes in this space. The set of entities in the space is the cartesian
product of the domains of the constitutive properties in the entity space.

### Defining property domains

The YAML for the constitutive properties in the `entityspace` has the following
structure

<!-- markdownlint-disable line-length -->

```yaml
identifier: model_name # The name of the property
propertyDomain: # The domain describes the values the property can take
  variableType:# The type of the variable: CATEGORICAL_VARIABLE_TYPE, DISCRETE_VARIABLE_TYPE, CONTINUOUS_VARIABLE_TYPE or UNKNOWN_VARIABLE_TYPE
    # The type defines what values the next fields can take.
  values: # If the variable is CATEGORICAL_VARIABLE_TYPE this is a list of the categories
    -  # If the variable is DISCRETE_VARIABLE_TYPE this can be a list of discrete float or integer values it can take
  domainRange: # If the variables is DISCRETE_VARIABLE_TYPE or CONTINUOUS_VARIABLE_TYPE this is the min inclusive, max exclusive range it can take
    # If the variable is DISCRETE_VARIABLE_TYPE and values are given this must be compatible with the values
  interval: # If the variable is DISCRETE_VARIABLE_TYPE this is the interval between the values.
    # If given domainRange is required and values cannot be given
  probabilityFunction: # Optional. The sampling distribution, for example uniform

```

<!-- markdownlint-enable line-length -->

As long as all constitutive properties are not "UNKNOWN_VARIABLE_TYPE" there is
sufficient information to sample new entities from the `entityspace`
description.

For more on property types, domains and probability functions see
[properties and domains](../concepts/properties-and-domains.md).

>[!TIP] Writing Short Constitutive Properties
>
> You can often write the constitutive properties in a shorter form
> then output by `ado template` which is verbose by default.
>
> - In many cases you do not need to specify the `variableType`
> as [it can be inferred](../concepts/properties-and-domains.md#auto-inference-of-property-domain-types).
> - The field `probabilityFunction` is not currently used so
> can be safely omitted.
> - The `metadata` fields which are output by `ado template` are not required.
> The main reason to have `metadata` is if you want to
> record why a certain domain was chosen.
>

### Ensuring the `entityspace` and `measurementspace` are compatible

Experiments take entities as inputs and those entities must have values for
various properties in order for the experiments to be able to process them. This
means the domains of the properties in the `entityspace` must be compatible with
the experiments - if not entities could be sampled that experiments in the
`measurementspace` cannot measure.

For example, to see the input requirements of the experiment
`finetune_full_benchmark-v1.0.0` you can run:

```commandline
ado describe experiment SFTTrainer.finetune_full_benchmark-v1.0.0
```

you will get output like

<!-- markdownlint-disable line-length -->

```terminaloutput
Identifier: SFTTrainer.finetune_full_benchmark-v1.0.0
Description: Measures the performance of full-finetuning a model for a given (GPU model, number GPUS, batch_size,
model_max_length, number nodes) combination.

Required Inputs:

   Constitutive Properties:
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: model_name
     Description: The huggingface name or path to the model
     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: [
            'allam-1-13b',
            'granite-3-8b',
            'llama3-8b',
            ... 37 more values ...
        ]

    ───────────────────────────────────────────────────────────────────────────────────────────────────
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: model_max_length
     Description: The maximum context size. Dataset entries with more tokens they are truncated. Entri
     are padded
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [1, 131073]

    ───────────────────────────────────────────────────────────────────────────────────────────────────

   ... 2 more required inputs ...

Optional Inputs and Default Values:

    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: max_steps
     Description: The number of optimization steps to perform. Set to -1 to respect num_train_epochs i
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [-1, 10001]

     Default value: -1
    ───────────────────────────────────────────────────────────────────────────────────────────────────
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: torch_dtype
     Description: The torch datatype to use
     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['bfloat16', 'float16', 'float32']

     Default value: 'bfloat16'
    ───────────────────────────────────────────────────────────────────────────────────────────────────
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: gpu_model
     Description: The GPU model to use
     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: [
            None,
            'NVIDIA-A100-SXM4-80GB',
            'NVIDIA-H100-80GB-HBM3',
            ... 5 more values ...
        ]

     Default value: None
    ───────────────────────────────────────────────────────────────────────────────────────────────────

     ... 22 more optional inputs ...

Outputs:
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   finetune_full_benchmark-v1.0.0-is_valid
   finetune_full_benchmark-v1.0.0-dataset_tokens_per_second_per_gpu
   finetune_full_benchmark-v1.0.0-train_runtime
   ... 20 more outputs ...
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
```

<!-- markdownlint-enable line-length -->

You can see the required inputs under the section `Required Inputs` and the
optional inputs under `Optional Inputs and Default Values`.
[Parameterizing experiments](#parameterizing-experiments) explains how to use
optional properties.

## Defining the `measurementspace`

The `experiments` field lists the experiments that will be applied to the
`entities` in the space. Each entry references an experiment provided by an
actuator.

### Referencing an experiment

An experiment reference has the following fields:

<!-- markdownlint-disable line-length -->

```yaml
experiments:
  - actuatorIdentifier: vllm_performance # The ACTUATOR ID column of "ado get experiments"
    experimentIdentifier: vllm-bench-endpoint # The EXPERIMENT ID column of "ado get experiments"
    experimentVersion: 1.0.0 # The VERSION column of "ado get experiments". Required if the referenced experiment has a version
    parameterization: # Optional. Values to fix for the experiment's optional inputs
      - value: 30
        property:
          identifier: temperature
```

<!-- markdownlint-enable line-length -->

You can list as many experiments as you want. Each one contributes its target
properties to the space, and each one is applied to every entity sampled by an
explore operation.

### Setting the experiment version

`experimentVersion` is the algorithm version of the experiment, a
`MAJOR.MINOR.PATCH` SemVer string. Set it to the value in the `VERSION` column
of `ado get experiments`, which is also shown by `ado get experiment $ID`.

A reference resolves to the experiment whose version matches exactly, so
omitting `experimentVersion` checks for an experiment whose version is `None`.
Hence, experiments that declare a version require the field.

If you omit `experimentVersion` for an experiment that has a version you will
get this error:

<!-- markdownlint-disable line-length -->

```terminaloutput
ERROR:  Unknown experiment in configuration. This can be due to an actuator not being installed or if the referenced experiment is external: The vllm_performance actuator was found but a match to vllm_performance.vllm-bench-endpoint was not found using mode fully_qualified_version. Available versions in catalog: 1.0.0.
```

While if you supply a version that does not match the catalog you get:

```terminaloutput
ERROR:  Experiment version mismatch in configuration: Algorithm version mismatch for experiment 'vllm-bench-endpoint' in catalog 'vllm_performance'. Reference requires version 'vllm-bench-endpoint@1.0.1' but catalog provides 'vllm-bench-endpoint@1.0.0'.
```

### Experiment versions and memoization

Explore operations can be configured to
[memoize](../concepts/data-sharing.md#memoization) measurements: an entity that
has already been measured by an experiment is not measured again.

The key used to identify if a requested experiment has already been applied
to an entity is the experiments **major version parameterized identifier**.
This is made up of:

- the actuator identifier
- the experiment identifier
- the **major** version
- the parameterization.

So `vllm-bench-endpoint` at version `1.0.0` stores its
results under `vllm-bench-endpoint@v1`, and the observed property for its
`request_throughput` target property is
`vllm-bench-endpoint@v1-request_throughput`.

This means:

- Bumping the minor or patch version of an experiment reuses existing results,
  as `1.0.0` and `1.2.0` share the same major version `@v1`.
- Bumping the major version starts a fresh set of results under `@v2`.
- Adding a version to a previously unversioned experiment starts a fresh set of
  results, as `solve_mip` and `solve_mip@v1` are different keys.
- Each parameterization of the experiments gets its own key, for example
  `peptide_mineralization@v1-temperature.30`.

For the rules experiment authors follow when choosing a version see
[declaring an algorithm version](../developer-guide/creating-custom-experiments.md#declaring-an-algorithm-version).

### Parameterizing experiments

If an experiment has
[optional input properties](../concepts/actuators.md#optional-inputs) you
can define equivalent properties in the entity space. If you don't, the default
value for the property will be used.

In addition, you can define your own custom parameterization of the experiment.
For example, take the following experiment:

<!-- markdownlint-disable line-length -->

```terminaloutput
Identifier: robotic_lab.peptide_mineralization@1.0.0
Version: 1.0.0
Description: Measures adsorption of peptide lanthanide combinations

Required Inputs:

   Constitutive Properties:
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: peptide_identifier
     Description: The identifier of the peptide to use
     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['test_peptide', 'test_peptide_new']

    ───────────────────────────────────────────────────────────────────────────────────────────────────
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: peptide_concentration
     Description: The concentration of the peptide
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]

    ───────────────────────────────────────────────────────────────────────────────────────────────────
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: lanthanide_concentration
     Description: The concentration of lanthanide
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]

    ───────────────────────────────────────────────────────────────────────────────────────────────────

Optional Inputs and Default Values:

    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: temperature
     Description: The temperature at which to execute the experiment
     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE
        Range: [0, 100]

     Default value: 23
    ───────────────────────────────────────────────────────────────────────────────────────────────────
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: replicas
     Description: How many replicas to average the adsorption_timeseries over
     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [1, 4]

     Default value: 1
    ───────────────────────────────────────────────────────────────────────────────────────────────────
    ───────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: robot_identifier
     Description: The identifier of the robot to use to perform the experiment
     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['harry', 'hermione']

     Default value: 'hermione'
    ───────────────────────────────────────────────────────────────────────────────────────────────────

Outputs:
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
   peptide_mineralization@v1-adsorption_timeseries
   peptide_mineralization@v1-adsorption_plateau_value
 ─────────────────────────────────────────────────────────────────────────────────────────────────────────
```

<!-- markdownlint-enable line-length -->

It has three optional properties: `temperature`, `robot_identifier` and
`replicas`.

>[!NOTE]
>
> Each parameterization defines a distinct experiment for the purposes of
> [data reuse](#experiment-versions-and-memoization).

#### Example: Customizing an experiment

The default temperature is `23` degrees C, however imagine you want to run this
experiment at `30` degrees C. You can define a `discoveryspace` like:

```yaml
sampleStoreIdentifier: c04713
entitySpace:
  - identifier: peptide_identifier
    propertyDomain:
      values: ["test_peptide"]
  - identifier: peptide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: lanthanide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
experiments:
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
    experimentVersion: 1.0.0
    parameterization:
      - value: 30
        property:
          identifier: "temperature"
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

#### Example: Multiple customizations of the same experiment

You can add the multiple custom parameterizations of the same experiment e.g.
one experiment that runs at 30 degrees C and another at 25 degrees.

```yaml
sampleStoreIdentifier: c04713 # PUT REAL ID HERE
entitySpace:
  - identifier: peptide_identifier
    propertyDomain:
      values: ["test_peptide"]
  - identifier: peptide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: lanthanide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
experiments:
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
    experimentVersion: 1.0.0
    parameterization:
      - value: 30
        property:
          identifier: "temperature"
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
    experimentVersion: 1.0.0
    parameterization:
      - value: 25
        property:
          identifier: "temperature"
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

#### Example: Using an optional property in the `entityspace`

Finally, if you want to scan a range of temperatures in your discovery space,
the best would be to move this parameter into the `entityspace`:

```yaml
sampleStoreIdentifier: c04713 # PUT REAL ID HERE
entitySpace:
  - identifier: peptide_identifier
    propertyDomain:
      values: ["test_peptide"]
  - identifier: peptide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: lanthanide_concentration
    propertyDomain:
      values: [0.1, 0.4, 0.6, 0.8]
  - identifier: temperature
    propertyDomain:
      domainRange: [20, 30]
      interval: 1
experiments:
  - actuatorIdentifier: robotic_lab
    experimentIdentifier: peptide_mineralization
    experimentVersion: 1.0.0
metadata:
  description: Space for exploring the absorption properties of test_peptide
```

Here entities will be generated with a temperatures property that ranges from 20
to 30 degrees. When the experiment is run on the entity it will retrieve the
value of the temperature from it rather than the Experiment.

Our toy
[example actuator](https://github.com/IBM/ado/tree/main/plugins/actuators/example_actuator)
contains the above examples. You can use it to experiment and explore custom
parameterization.

## Choosing the `samplestore`

Every `discoveryspace` stores its `entities` and measurement results in a
`samplestore`. The `sampleStoreIdentifier` field holds its identifier and
defaults to `default`, the per-project store which `ado` creates the first time
it is used.

To see the existing stores run:

```commandline
ado get samplestores
```

You can set the store from the command line instead of editing the YAML:

<!-- markdownlint-disable MD007 -->
- `--use-default-sample-store` uses the project's `default` store
- `--new-sample-store` creates and uses a new, empty store
- `--use-latest samplestore` uses the most recently created store
- `--with store=$ID` uses the store `$ID`
- `--with store=FILE.yaml` creates the store described by `FILE.yaml` and uses
  it. This is how you
  [copy data into a new store](sample-stores.md#copying-data-into-a-samplestore)
  at the same time as creating a space.
- `--set sampleStoreIdentifier=$ID` overrides the field directly
<!-- markdownlint-enable MD007 -->

`--with`, `--new-sample-store` and `--use-latest samplestore` take precedence
over `--use-default-sample-store`, `--set` and the value in the YAML. See the
[samplestores](sample-stores.md) documentation for more details on the store
types and on the [default store](sample-stores.md#the-default-samplestore).

### `discoveryspaces` and shared `samplestores`

Multiple `discoveryspace` resources can use the same `samplestore` resource. In
this case you can think of the `discoveryspace` as a "view" on the `samplestore`
contents, filtering just the `entities` that match its description.

To be more rigorous, given a `discoveryspace` you can apply this filter in two
ways:

1. Filter `entities` that were placed in the `samplestore` via an operation on
   the `discoveryspace`
2. Filter `entities` in the `samplestore` that match the `discoveryspace`

To understand the difference in these two methods imagine two overlapping
`discoveryspaces`, A and B, that use the same `samplestore`. If someone uses
method one on `discoveryspace` A, they will only see the `entities` placed there
by operations on `discoveryspace` A. However, if someone uses method two on
`discoveryspace` A, they will see `entities` placed there via operations on both
`discoveryspace` A and space B.

Shared samples stores also allow data to be reused across `discoveryspaces`,
potentially accelerating exploration operations. See the
[shared sample store](../concepts/data-sharing.md) documentation for
further details.

## Running operations on a `discoveryspace`

A `discoveryspace` is a description of what can be measured. Data is added to it
by running an `operation` on it. There are two kinds:

- **explore** operations, such as a random walk or a Bayesian optimization,
  sample `entities` from the `entityspace`, apply the experiments in the
  `measurementspace` to them, and store the results in the `samplestore`
- **analysis** and other operations process an existing space, for example
  ranking its `entities` or producing a report

See the [operation](operation.md) documentation for how to configure and start
one, and
[working with operators](../user-guide/operators/working-with-operators.md) for
the available operators.

## Accessing measurement data

To see the measurement data collected by explore operations on a space use

```commandline
ado show measurements space --use-latest
```

By default, this will output the entities and their measurements as a table.
There are various option flags that control this behaviour e.g. output to a CSV
file.

As described in
[shared samplestores](#discoveryspaces-and-shared-samplestores) there are two
lists of entities this could show. The command above uses filter (1) -
`entities` that were placed in the `samplestore` via an operation on the
`discoveryspace`.

If you want to use filter (2) - `entities` in the `samplestore` that match the
`discoveryspace` - use:

```commandline
ado show measurements space --use-latest --include matching
```

>[!NOTE]
>
> In both cases measurements on the entity will be filtered to be only those
> defined by the `measurementspace` of the `discoveryspace`

Two other options list the `entities` of a finite space that have no data yet

- `--include unmeasured` lists `entities` defined by the `discoveryspace` with
  no measurements from operations on it
- `--include missing` lists `entities` defined by the `discoveryspace` with no
  measurements in the `samplestore`

### Target vs observed property formats

>[!NOTE]
>
> For the conceptual distinction between target and observed properties see
> [Target and Observed Properties](../concepts/actuators.md#target-and-observed-properties).

There are two formats the measurements can be output controlled by the
`--property-format` option to `show measurements`

The observed format outputs one row per entity. The columns are constitutive
property names and the observed property names i.e. they include both the
experiment id and target property id. This ensures that with one row per entity
there are no clashing column names.

The target format outputs one row per entity+experiment combination: so if there
are two experiments in the Measurement Space then there will be two rows per
entity. In this format the columns are constitutive property names and target
property names.

>[!NOTE]
>
> With `property-format=target` if the measurement space contains multiple
> experiments measuring _different_ target properties, this will result in many
> empty fields in the table. This is because the column for a given target
> of one experiment will not have values in the rows corresponding
> to other experiments.

### Accessing measurement data programmatically

Assuming you have your [context](metastore.md#contexts-and-projects) in a file
"my_context.yaml"

```python
import yaml
from ado.metastore.project import ProjectContext
from ado.core.discoveryspace.space import DiscoverySpace

with open("my_context.yaml") as f:
    c = ProjectContext.model_validate(yaml.safe_load(f))

space = DiscoverySpace.from_stored_configuration(
    project_context=c, space_identifier="space_abc123"
)
# Get the sampled and measured entities. Returns a pandas DataFrame
table = space.measuredEntitiesTable()
# Get the matching. Returns a pandas DataFrame
table = space.matchingEntitiesTable()
```

## Inspecting a `discoveryspace`

To list the spaces in your project:

```commandline
ado get spaces
```

```terminaloutput
┌───────┬──────────────────────┬─────────────────────────────────────┬─────────┐
│ INDEX │ IDENTIFIER           │ NAME                                │ AGE     │
├───────┼──────────────────────┼─────────────────────────────────────┼─────────┤
│ 0     │ space-3b85e4-f60613  │ ml_multicloud_basic                 │ 371d7h  │
│ 1     │ space-047b6a-f60613  │ rosenbrock_3d                       │ 367d7h  │
│ 2     │ space-ab21b4-f60613  │ test_vllm_performance_space         │ 366d0h  │
└───────┴──────────────────────┴─────────────────────────────────────┴─────────┘
```

Add `--details` to include the `DESCRIPTION` and `LABELS` of each space.
`--filter` and `--label` narrow the list, and `--related-to` restricts it to
spaces related to another resource.

To retrieve the stored configuration of a space:

```commandline
ado get space space-047b6a-f60613 -o yaml
```

`ado show stats space` reports what the space contains, counting the results of
the operations run on it:

```commandline
ado show stats space space-047b6a-f60613 -o yaml
```

```terminaloutput
space-047b6a-f60613:
  AGE: 367d7h
  ENTITIES_WITH_ALL_MEASUREMENTS: 28
  ENTITIES_WITH_PARTIAL_MEASUREMENTS: 12
  EXPERIMENTS: 1
  EXPLORE_OPERATIONS: 1
  MATCHING_ENTITIES: 148
  MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS: 148
  MATCHING_WITH_MEASUREMENTS: 148
  MEASURED_ENTITIES: 40
  NAME: rosenbrock_3d
  OPERATIONS: 1
  SIZE_OF_ENTITY_SPACE: null
  UNMEASURED_ENTITIES: .inf
```

The `MEASURED_ENTITIES` counts come from operations on this space, while the
`MATCHING_*` counts cover every entity in the `samplestore` that matches the
space - the two filters described in
[shared samplestores](#discoveryspaces-and-shared-samplestores).
`SIZE_OF_ENTITY_SPACE` and `UNMEASURED_ENTITIES` have values for finite spaces.

For a human-readable view of the `entityspace`, the `measurementspace` and the
`samplestore` a space uses:

```commandline
ado describe space space-047b6a-f60613
```

An example of this output is in the
[Discovery Space concepts page](../concepts/discovery-spaces.md#example-fine-tuning-deployment-configuration).

See the [ado CLI reference](../cli-reference/index.md) for the full set of flags
these commands accept.

### Differences between input configuration YAML and stored configuration YAML

After creating a `discoveryspace`, if you `ado get` its YAML you will notice
that the information output is different from the one you provided in input.
This is because the list of experiment references set in the YAML is expanded
into the full experiment definitions and stored with the `discoveryspace`.
