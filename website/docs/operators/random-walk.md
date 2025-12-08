<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
## Overview

> [!TIP]
>
> The `random_walk` operator is installed when you install `ado`

### What does the `random_walk` operator do?

The `random_walk` operator provides different ways to randomly sample and
measure points from a `discoveryspace`. Despite its name, it can also perform
deterministic sampling.

`random_walk` is an `explore` type operator.

### When should you use the `random_walk` operator?

Use the `random_walk` operator when you want to:

- get an unbiased idea of the distribution of property values across entities in
  the `discoveryspace`
- sample and measure all the entities in the space in random or sequential order
  (finite size spaces)
- sample entities matching particular conditions (see below)

The `random_walk` operator supports `memoization`: if it samples the same entity
twice, and that entity has already had the measurement space applied, it will
replay the already measured values (by default).

### What happens if I apply multiple `random_walk` operations to a space?

If you apply multiple `random_walk` operations, you just get multiple random
walks of the different lengths and types you have requested.

All explore operations are independent. This means each proceeds as configured -
the only influence of previous operations is to enable `memoization` if a
subsequent operation visits the same point.

To concretize this, consider two `random_walk` operations that sample
deterministically (i.e. aren't actually random). The first is configured to
sample 50 entities, the second 200 entities. Suppose also that the first 50
entities are common to both `random_walk` operations. At the start no entities
have been sampled and measured from a given discovery space.

After the first operation:

- 50 entities will have been sampled and measured from the `discoveryspace`
  (assuming no errors) and placed in its `samplestore`
- The timeseries of this first operation is stored. It has 50 entities in it.

After the second operation:

- 200 entities will have been sampled and measured from the `discoveryspace`
  (assuming no errors) and placed in its `samplestore`
- 150 will have been measured by this operation, with the first 50 being
  replayed (as they were already measured during the first operation)
- The timeseries of this second operation is stored. It has 200 entities in it.

## Controlling sampling and measurements: Continuous batching

When a `random_walk` operation encounters an unmeasured entity in the
`discoveryspace`, it applies the experiments defined by its `measurementspace`.
Depending on the experiments, you may want to control how many concurrent
experiments are being executed.

`random_walk` uses continuous batching to set the number of concurrent
**requested** experiments and ensure that, as far as possible, there is always
this number of experiments in flight.

This approach maximizes throughput compared to standard batch-wise submission.
In the normal case the time to finish measuring batch of N entities is, at a
minimum, the time taken for the longest experiment to complete. This means if
one experiment is very long and the others short, there can be capacity in the
system for (N-1) additional entities to be measured but it will not be used.

The next section explains more about configuring continuous batching

## Configuring a `random_walk` operation

The parameters for a `random_walk` operation are (default values shown):

<!-- markdownlint-disable line-length -->

```yaml
numberEntities: 1 # The maximum number of entities to sample. Can also be the string "all" - see "Sampling all Entities".
batchSize: 1 # The Number of entities in the initial batch. For more on this see "Batch Size and Concurrent Experiments" below
samplerConfig:
  mode: random # How to sample - can be random, sequential, sequentialgrouped or randomgrouped. sequential requires the sampling supports sequencing entities
  samplerType: selector # How to sample entities. Can be selector or generator. For more see Sampling Types and Modes below
  grouping: [] # If the mode sequentialgrouped or randomgrouped this is a list of constitutive properties identifiers to group the entities by
singleMeasurement: true # If true memoization is used. If false already measured entities will be re-measured. For more see Multiple Measurement below
maxRetries: 0 # The number of times to retry a failed measurement on an entity. See Retrying Failed Measurements below.
filter:
  filterMode:
    nofilter # Sets filters on the entities in the space that should be sampled. Entities not matching the filter will not be sampled or measured.
    # See "Filtering Entities" below for more details
```

<!-- markdownlint-enable line-length -->

An example `operation` YAML with a sequential selector:

```yaml
metadata:
  name: "Example random walk operation"
operation:
  module:
    operatorName: random_walk
    operatorType: explore
  parameters:
    batchSize: 1
    samplerConfig:
      mode: random
      samplerType: selector
    numberEntities: 1
    singleMeasurement: true
    filter:
      filterMode: unmeasured
spaces:
  - your-spaces
```

!!! info end

    You can get a default `random_walk` operation template and the schema of its
    parameters by running
    `ado template operation --operator-name random_walk --include-schema`.
    The information output by this command should always be preferred
    over the information presented here if there is an inconsistency.

### Batch Size and Concurrent Experiments

When it comes to managing resources during an exploration, the key variable one
wants to control is the number of concurrent experiments.

For the `random_walk` operator, this number is its `batchSize` parameter (the
number of initial entities submitted) multiplied by the number of experiments in
the `measurementspace` of the `discoveryspace` it is operating on. For example,
if the `batchSize` is 2 and there are 2 experiments defined in the
`measurementspace` there will be 4 (2\*2) experiments requested. Each experiment
will be measuring one entity. The continuous batching will endeavour to keep
this many concurrent experiment requests during the operation.

!!! info end

    The `random_walk` operator only knows how many experiments it has requested,
    not how many are actually executing.
    Hence, continuous batching can only maintain that there are
    N experiments requested at any time.

### Base Sampling Types and Modes

The `samplerConfig` field controls how Entities are sampled during the
operation. The base `samplerConfig` is shown in the examples above and has the
following fields and defaults:

```yaml
mode: random
samplerType: selector
grouping: []
```

#### Sampling Types

There are two sampling types: `generator` and `selector`.

The `generator` sampling type _generates_ valid entities based on the
`entityspace` definition. It currently only works with entities space that have
a finite size i.e. all constitutive properties are DISCRETE or CATEGORICAL and
are bounded.

The `selector` sampling type draws _existing matching entities_ from the
`samplestore` of the `discoveryspace` i.e. it doesn't use the entity space.

#### Sampler Modes

Both sampling types support four modes, which can be categorised as flat or
grouped:

<!-- markdownlint-disable ul-indent -->

- flat
    - sequential
    - random
- grouped
    - sequentialgrouped
    - randomgrouped
<!-- markdownlint-enable ul-indent -->
!!! info end

    The flat modes sample entities directly.
    For the grouped modes, the sampling is done on 2 levels - 
    groups and then entities in the groups. The group level
    sampling can be either sequential or random, while group member level is always
    sequential

The details of how each mode works can differ slightly depending on if a sampler
is `generator` or `selector`.

- `sequential`
  - `generator`: The entities are generated by iterating over the constitutive
    properties values with the first property being innermost and last outermost
    (see below for more details)
  - `selector`: The entities are iterated in the order they are returned by the
    `samplestore`
- `random`
  - For both samplers the entities are sampled in a random order.
- `sequentialgrouped`
  - The entities are grouped by a user defined condition. Either all the
    entities in space (`generator`) or the entities in the `samplestore`
    (`selector`)
    - The groups are iterated in order
    - The group members are iterated in order
- `randomgrouped`
  - The entities are grouped in the same way as `sequentialgrouped`
    - The groups are iterated randomly
    - The group members are iterated in order

In pseudocode, sequential mode iterates as follows if there are N constitutive
properties

```python
for x in propertyN.values:
  for y in propertyN_1.values:
     ...
     for z in property1.values:
         entity({'propertyN':x, 'propertyN_1':y, ..., 'property1':z})
```

#### Why Grouped Modes?

The advantage of the group modes is that they can allow
[actuators](../actuators/working-with-actuators.md) to reuse their test
environments, providing faster measurements. For example, consider an actuator
that needs to create different test environments for different groups of
entities. This creation may incur a significant overhead or there may be a
limited on the number of simultaneous environments that can be created. In this
case the measurements would be most efficient if all entities in a group are
submitted to the actuator in sequence, as the actuator can create a test
environment once and then reuse it for all group members. This is what grouping
allows.

!!! info end

    See the docs of the specific actuator you are using to see if and how it can
    benefit from grouping.

#### Enabling Grouping

To use the grouped modes (`randomgrouped`, `sequentialgrouped`) you need to
supply a list of constitutive properties to group by using the `grouping`
parameter. Here is an example configuration for using a `generator` sampler with
`sequentialgrouped` mode:

```yaml
metadata:
  name: "Example grouped random walk operation"
operation:
  module:
    operatorName: random_walk
    operatorType: explore
  parameters:
    batchSize: 1
    samplerConfig:
      samplerType: generator
      grouping:
        - $CONSTITUTIVE_PROPERTY_ONE
        - $CONSTITUTIVE_PROPERTY_TWO
      mode: sequentialgrouped
    numberEntities: 1

    singleMeasurement: true
    filter:
      filterMode: unmeasured
spaces:
  - your-spaces
```

### Custom Samplers

It is also possible to specify that `random_walk` uses a custom sampler. This is
a class that inherits from
`orchestrator.core.discoveryspace.samplers.BaseSampler`. This is useful for
implementing more complex sampling schemes. For example, for developers who want
to use random_walk to drive an exploration but have custom logic to execute
before choosing each sample/entity.

For custom samplers the `samplerConfig` field has the following structure:

<!-- markdownlint-disable line-length -->

```yaml
module:
  moduleClass: #The name of the custom sampler class
  moduleName: #The name of the python module containing the sampler
parameters: # A dictionary of key value pairs with the values for the custom samplers input parameters
  ...
```

<!-- markdownlint-enable line-length -->

#### Implementing a Custom Sampler

To implement a custom sampler create a sub-class of
`orchestrator.core.discovery.samplers.BaseSampler` and implement all required
methods

The `BaseSampler` class does not specify any `__init__` parameters. If your
custom class requires initialization parameters then

- define a pydantic model for them
- override the `parameters_model` class method to return this model
- add a non keyword parameter to your custom classes `__init__` that is this
  type.

For example:

```python
# Class for the custom samplers parameters
class MySamplerParams(BaseModel):
   ...

# Subclass of BaseSampler implementing the custom sampling logic
class MySampler(BaseSampler):

    @classmethod
    def parameters_model(cls) -> Optional[Type[BaseModel]]:

        # Return the custom samplers parameters model
        return MySamplerParams

    # Add an init arg to take the parameters model
    def __init__(self, parameters: MySamplerParams):
         ...
```

### Sampling all Entities

If either of the following conditions are true you can specify a value of "all"
for the `numberOfEntities` field in the random walk configuration:

- All dimensions in the `entityspace`s are discrete and bounded or categorical
- The sampling type is `selector` i.e. you are iterating over an existing set
  number of entities in a `samplestore`

In the first case `all` will be converted to the size of the space. In the
second case `all` will be converted to the number of matching entities in the
`samplestore`.

If both of these conditions is False the `random_walk` operator will raise a
ValueError when the execution starts.

!!! info end

    Depending on the Filter settings a randomwalk operation may not sample "all"
    entities even if "all" is specified. This is because the filter may filter out
    some entities.

!!! warning end

    For `discoveryspaces` where one/both of the above conditions are True setting
    `numberOfEntities` greater than the corresponding size (size of space, or number
    of matching entities in `samplestore`) will raise a ValueError. This means you
    cannot set `numberOfEntities` to an arbitrarily large number to ensure sampling
    all of them - use `all` instead.

### Filtering Entities

In some circumstance you may want to only sample a subset of Entities. Some
examples include

- You want to skip replaying entities already measured as there are many and
  just sample the unmeasured ones
- You might want to complete measurement of partially measured entities but not
  want to start measuring unmeasured ones - for example for testing a
  cost-function
- You might want to add additional measurements to all entities already fully
  measured but not want to add measurements to any other entities.

The `filter` field provides this capability. It has one sub-field `filterMode`
which can take the following values:

- `noFilter`: The default value. No filtering is applied
- `unmeasured`: Only Entities with no measurements by any of the experiments in
  the `measurementspace` will be sampled
- `partial`: Only Entities with measured by at least one and less than all the
  experiments in the `measurementspace` will be sampled
- `measured`: Only Entities fully measured by the experiments in the
  `measurementspace` will be sampled

### Multiple Measurement

By setting `singleMeasurement:` to False the random walk operation will measure
ALL entities it samples, even if they already have measurements.

If entities have multiple measurements e.g. you turned this off and then turned
it on again, then if an entity has multiple measurements each one will be
replayed.

Check [replayed measurements](explore_operators.md#replayed-measurements) for
more details.

### Retrying Failed Measurements

If the measurement of an entity by an experiment fails `random_walk` can retry
it. The parameter controlling this is `maxRetries` which by default is 0 - no
retries. If `maxRetries` is N then failing measurements will be retried up to
`N` times.

#### Experiment request index v number of experiments requested

To understand a `random_walk` operations logs when maxRetries is greater than 0
it's necessary to understand how it tracks the entity+experiment combinations it
wants to measure versus the number of experiments it has requested to do these
measurements.

`random_walk` assigns an integer to each entity+experiment combination it wants
to measure. This is called the request index - the Nth entity+experiment
combination will have request index N-1.

`random_walk` tracks retries based on request index. For example, it tracks that
request index 5 has been retried 2 times.

!!! info end

    At the end of an `random_walk` operation, a summary of each request
    index that was retried is output.
    This includes how many times it was retried and what the
    final status was - either it performed maxRetries and still was FAILED
    or one of the retries indicated SUCCESS

    Example summary output. Here `Request 8` means request index 8.

    <!-- markdownlint-disable line-length -->
    ```terminaloutput
    Summary of 2 retried experiments
    Request 0: Request d39947. Entity: fuchsia-raft (mock-sample-store-31799c). Experiment: mock.test-experiment. Retried: 2 times. Final status: Success
    Request 8: Request f11853. Entity: flush-oasis (mock-sample-store-31799c). Experiment: mock.test-experiment. Retried: 1 times. Final status: Success
    ```
    <!-- markdownlint-enable line-length -->

!!! important end

    When retrying is enabled, the highest request index is not equal to the number
    of experiments submitted. For example there may be 40 entity+experiment
    combinations tested, meaning greatest requestIndex is 39, but 50 experiments
    requested due to failures and retries.

    This also means that when retrying is enabled the experimentRequested metadata
    recorded with each `random_walk` operation is not equal to (number entities
    sampled) x (number of `random_walk` in measurement space)

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-workflow-24:{ .lg .middle } **Try our Random Walk Example**

      ---

      Try using the `random_walk` operator with our [example](../examples/random-walk.md).

      [Random Walk example :octicons-arrow-right-24:](../examples/random-walk.md)

- :octicons-rocket-24:{ .lg .middle } **Create new Operators**

    ---

    Learn about extending ado with new [Operators](../operators/working-with-operators.md).

    [Creating new Operators :octicons-arrow-right-24:](../actuators/working-with-actuators.md)

</div>
<!-- markdownlint-enable line-length -->
