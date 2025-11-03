<!-- markdownlint-disable-next-line first-line-h1 -->
## Using externally obtained data: the replay actuator

The replay actuator allows you to leverage results that were obtained via
experiments outside of `ado` that are contained in external sources like
[CSV files](../resources/sample-stores.md#csvsamplestore). We can't repeat these
experiments, or add new data using them, in `ado` as no actuator exists to do
so. However, you still might want to define measurement spaces with them so
entities that have the relevant data can be sampled and the data used, perhaps
in a custom objective function.

The [taking a random walk](../examples/random-walk.md) tutorial uses external
data and the replay actuator.

### Importing data from a CSV

Often external data is stored in a CSV or table where each row contains
measurement results for some entity. One set of columns defines the entity (the
thing being measured) and another set of columns the results of one or more
experiments on the entity.

To use this data with `ado` the first step is to copy it into a `samplestore` at
creation time. When copying this data into the `samplestore` the columns
containing measured values (observed properties) and which columns containing
constitutive properties are defined. With this information `ado` can create
entities for each row. The following example is from
[taking a random walk](../examples/random-walk.md):

```yaml
{% include "../../../examples/ml-multi-cloud/ml_multicloud_sample_store.yaml" %}
```

The `copyFrom` section is where the external sources data should be copied into
the `samplestore` are defined. There can be multiple but here there is just one.

The relevant fields are:

- `module`: These are the values you set to indicate the data is in a CSV file
- `storageLocation`: This is the path the CSV file
- `parameters.identifierColumn`: This is the column in the CSV, if any, to use
  as the identifier of the created entities.
- `parameters.constitutivePropertyColumns`: This is a list of the columns in the
  CSV file that define the constitutive properties of the entities
- `experiments`: This section defines the experiments that were used to generate
  the data in the CSV file
  - `experiments.experimentIdentifier`: This is the name for the experiment in
    ado
  - `experiments.propertyMap`: This is a dictionary mapping the names of the
    properties experiment as they will appear in `ado` to column names in the
    CSV

The above YAML says to associate the data in the columns `wallClockRuntime` and
`status` with an experiment 'benchmark_performance' that measures properties
with the same name.

The `propertyMap` field allows you to handle column headers had names that are
not suitable for names of properties. For example if there was a column with
measurements on a molecule called `Real_pKa (-0.83, 10.58)`, you might want to
associate this with a property called `pka` instead:

```yaml
propertyMap:
  pka: "Real_pKa (-0.83, 10.58)"
```

### Using the external data in a `discoveryspace`

If you copied entities from an external source to $SAMPLE_STORE_IDENTIFIER and
in the process defined an external experiment called `my_experiment` then you
can use it in a `discoveryspace` with:

```yaml
sampleStoreIdentifier: $SAMPLE_STORE_IDENTIFIER
experiments:
  - actuatorIdentifier: replay
    experimentIdentifier: my_experiment
```

The [ml multi cloud](../examples/search-custom-objective.md) example uses this
approach.

### How the `replay` actuator works

Looking at the example in
[Importing data from a CSV](#importing-data-from-a-csv), you may wonder how
`ado` can use it, if it does not have an actuator that provides the experiment
`benchmark_performance`!

<!-- markdownlint-disable descriptive-link-text -->
What happens is that when a measurement of an experiment associated with the
`replay` actuator is requested to be performed on an entity, if the data is
present (because it was copied in) it is reused as normal by `ado`'s memoization
mechanism. If there is no data, it cannot be measured as no real experiment
exists, and the `replay` actuator handles this case correctly - it creates the
`No value to replay` messages seen
[here](../examples/random-walk.md#looking-at-the-operation-output).
<!-- markdownlint-enable descriptive-link-text -->

> [!IMPORTANT]
>
> To use external data via the replay actuator the relevant
> operator must be configured to use memoization. With the randomwalk and
> ray_tune operators this means singleMeasurement parameter is set to True (the
> default).
