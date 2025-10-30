# Calculating the morgan fingerprint of a set of molecules

## About this example

This example illustrates calculating the morgan fingerprint of a set of
molecules.

### What features are shown

- Creating a `samplestore` from existing data
  - Including importing pre-existing experiment measurement data with the
    entities
- Creating a `discoveryspace` which includes a new experiment - the morgan
  fingerprint calculation
- Performing a random walk `operation` that samples only existing entities from
  a `samplestore`

## Pre-requisites

1. `ado` installed
2. Access to a project for storing resources i.e. a valid active context
3. The example files

To get the example files, checkout the ado repo:

```commandline
git clone https://github.com/IBM/ado.git
```

and navigate to the following example directory

```commandline
cd ado/examples/pfas-generative-models
```

## Create a samplestore, copying in existing molecular data

First we create a `samplestore` resource, preloading it with molecules generated
by a transformer model and stored in a CSV file under `data/`

The `samplestore` resource configuration looks like

```yaml
{% include "./gen_models_transformer_test_sample_store.yaml" %}
```

The command to create the `samplestore` is:

```commandline
ado create samplestore -f gen_models_transformer_test_sample_store.yaml
```

This outputs a `samplestore` id which you should record for the next step.

## Create a Space

The `discoveryspace` configuration looks like:

```yaml
{% include "./space_transformer_simple.yaml" %}
```

You can see the experiment `calculate_morgan_fingerprint` from the
`molecule_embeddings` actuator is selected in the `measurementspace`.

Create the space:

```commandline
ado create space -f space_transformer_simple.yaml --use-latest samplestore
```

## Create an Operation

This operation samples 10 molecules from the space and applies the measurement
space, including `calculate_morgan_fingerprint` to them.

The `operation` resource configuration looks like:

```yaml
{% include "./operation_random_walk_test.yaml" %}
```

Start the operation:

```commandline
ado create operation -f operation_random_walk_test.yaml --use-latest space
```

## View the Results

To see the fingerprints of the ten sampled entities (molecules) run:

```commandline
ado show entities space --use-latest
```

Try as well:

```commandline
ado show details space --use-latest
```

## Next Steps

1. Increase the number of molecules in the operation and run again.
      - You will see the first 10 molecules are not measured again - they are
        "memoized"
      - You can also try changing the `mode` to `random` to get a true random walk
2. Have someone with distributed access the same project and retrieve the
   results.
3. Have them run another operation and observe how you can retrieve the results
4. If you run multiple operations, try using `ado show entities operation` to
   get the output from a particular operation
