<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
A `samplestore` resource is a database containing
[`entities`](../core-concepts/entity-spaces.md#entities) along with results of
experiments that have been applied to them.

## `samplestore`s and `discoveryspace`s

When you create a [discovery space](discovery-spaces.md) you associate a
`samplestore` with it. This is where the `discoveryspace` will read and write
data i.e., entities and the results of experiments on them. You primarily access
the entities in a `samplestore` via a `discoveryspace` that is attached to it.

You can think of a `discoveryspace` as a view or filter on a sample store - when
you access data in a `samplestore` through a discovery space you only see data
that matches the `discoveryspace`.

> [!TIP]
>
> - Multiple `discoveryspace`s can use the same `samplestore`
> - There is no restriction or condition on the `discoveryspace`s sharing a
>   `samplestore` i.e. they can be very similar or completely different.

## `samplestore`s and data-sharing

When multiple `discoveryspace`s use the same `samplestore` this enables
transparent data-sharing between the different `discoveryspace`s. When and how
data is shared is covered in detail in
[shared sample stores](../core-concepts/data-sharing.md).

To see the `discoveryspaces` using a given `samplestore` run

```commandline
ado show related samplestore $SAMPLE_STORE_IDENTIFIER
```

> [!TIP]
>
> The greater the similarity between two `discoveryspace`s, the greater
> the chance they can share data. So it is usually beneficial to ensure that
> such `discoveryspace`s use the same `samplestore`.
<!-- markdownlint-disable-next-line no-blanks-blockquote -->
> [!WARNING]
>
> If you use two different `samplestore`s for similar `discoveryspace`s there is
> no way to share results between them.

## active and passive Sample Stores

`ado` distinguishes two types of Sample Stores: **active** Sample Stores which
allow read and write; and **passive** Sample Stores that only have read
capabilities (for example a CSV file containing measurement data).

All `samplestore` resources created with `ado` will be **active**. However,
they can copy data in from **passive** Sample Stores.

## The primary Sample Store type: SQLSampleStore

The primary Sample Store used in `ado`, and represented by `samplestore`
resources, is SQLSampleStore. SQLSampleStore represents storage in SQL tables.
When you create a `samplestore` resource that uses SQLSampleStore the storage is
allocated automatically in the SQL db associated with the
[active context](metastore.md#contexts-and-projects)

## Creating a samplestore

Running `ado create samplestore --new-sample-store` will create an empty
SQLSampleStore in the current context.

## The default samplestore

`ado` provides a **default** `samplestore` (whose identifier is `default`) per
project, removing the need to create one explicitly unless necessary. This
`samplestore` is created **automatically** when it is first required.

There are three ways to use the default `samplestore` - each will create it, if
it doesn't already exist.

1. **Referencing it in the space configuration** by setting the
   `sampleStoreIdentifier` to `default` in the space YAML:

   ```yaml
   sampleStoreIdentifier: default
   ```

2. **Using the `--use-default-sample-store` flag** with the `ado create space`
   command:

   ```terminal
   ado create space --use-default-sample-store
   ```

3. **Using the `--set` flag** to explicitly override the sample store
   identifier:

   ```terminal
   ado create space --set sampleStoreIdentifier=default
   ```

These options are interchangeable and can be used depending on your workflow or
preference.

### Copying data into a samplestore

You can specify data to be copied into a new `samplestore` resource on creation.
The data comes from other Sample Stores. The general structure of the YAML when
copying from other sample stores is:

```yaml
specification:
  module:
    moduleClass: SQLSampleStore
    moduleName: orchestrator.core.samplestore.sql
copyFrom: # An array of Sample Stores data will be copied from
  - identifier: # Optional, the id of the Sample Store if not given in the storageLocation
    module: # The type of this Sample Store
      moduleClass: ... # The module class for this sample store
      moduleName: ... # The name of the module containing the class
    parameters: # Sample Store parameters
    storageLocation: # The location of this Sample Store
```

The [Sample Store types](#sample-store-types) section details how to fill the
above fields for the different available Sample Store. Here is an example of
copying data from a CSV file using `CSVSampleStore`:

```yaml
specification:
  module:
    moduleName: orchestrator.core.samplestore.sql
    moduleClass: SQLSampleStore
copyFrom:
  - module:
      moduleClass: CSVSampleStore
    storageLocation:
      path: "examples/ml-multi-cloud/ml_export.csv"
    parameters:
      generatorIdentifier: "multi-cloud-ml"
      identifierColumn: "config"
      experiments:
        - experimentIdentifier: "benchmark_performance"
          propertyMap:
            wallClockRuntime: "wallClockRuntime"
```

## Accessing the entities in a sample store

You access the entities in a `samplestore` via a discovery space attached to it.

For an existing `discoveryspace` to retrieve all entities that match it run

```commandline
ado show entities space $SPACEID --include matching
```

You can also define a `discoveryspace` in a YAML and run:

```commandline
ado show entities space --file $FILE
```

This allows you to see what entities match a space without having to create it.

## Sample Store types

### SQLSampleStore

This is an active Sample Store that stores entity data in SQL tables. In `ado` a
SQLSampleStore is always associated with a particular project.

When you want to copy from another SQLSampleStore you need the identifier and
the metastore URL to the project it is in

<!-- markdownlint-disable line-length -->
```yaml
copyFrom:
  - identifier: source_abc123
    module:
      moduleClass: SQLSampleStore
      moduleName: orchestrator.core.samplestore.sql
    storageLocation:
      host: localhost
      port: 30002
      database: my_project. # The database field is the name of the project containing the samplestore
      user: my_project # The user field is the name of the project containing the samplestore
      password: XXXXXXX
```
<!-- markdownlint-enable line-length -->

### CSVSampleStore

This is a passive Sample Store that can be used to extract entities from a CSV
file. It is assumed each row is an entity and the columns are constitutive
properties or observed properties

<!-- markdownlint-disable line-length -->
```yaml
copyFrom:
  - module:
      moduleClass: CSVSampleStore
      moduleName: orchestrator.core.samplestore.csv
    storageLocation:
      path: 'examples/ml-multi-cloud/ml_export.csv'. # The path to the CSV file
    parameters:
      generatorIdentifier: 'multi-cloud-ml' # A string that will be stored with the extracted entities as their generatorIdentifier
      identifierColumn: 'config'. # The column in the CSV file that contains the entity id
      constitutivePropertyColumns:
        -  # A list of columns which contain constitutive properties
      experiments: # A list of dictionaries that map CSV columns to experiments and target properties. Each dictionary is an experiment
        - experimentIdentifier: 'benchmark_performance' # The experiment name you want the following properties to be associated with
          propertyMap: # List of target property name:column id pairs
            wallClockRuntime: 'wallClockRuntime' # The key is the target property name, the value is the column containing the values for that property
```
<!-- markdownlint-enable line-length -->

Note, since CSV files contain arbitrary data in general there is no way `ado`
can know how a particular value was generated or how to generate new such
values. However, the measurements in a CSV can be mapped to the
"experiment+property" model that `ado` uses, if you want to copy them.

You do not have to copy all the columns in a CSV or have any experiments.

## Deleting sample stores

!!!info

      Please note that
      [standard deletion constraints](resources.md#deleting-resources) apply alongside
      the considerations discussed in this section.

Deleting a sample store is a high-impact operation and should be performed with
caution. When a sample store is deleted:

- **All stored entities will be deleted**.
- **All leftover measurement results stored in it will be permanently deleted**.
  These will be measurements that were copied into the `samplestore` i.e., not
  generated through `ado` operations. All results from `ado` operations would
  have already been subject to
  [standard deletion constraints](resources.md#deleting-resources).
- **The corresponding database tables will be dropped**.

This is especially critical when the sample store was populated externally, such
as via a `CSVSampleStore`. In such cases, **deletion may result in the loss of
externally sourced data** that cannot be recovered unless it has been backed up
beforehand.

To prevent this from unintentionally happening, `ado` will check if the sample
store contains stored results, and exit if this is the case. A warning such as
the following will be output:

> ERROR: Cannot delete sample store 995ff6 because there are 68 measurement
> results present in the sample store.
>
> HINT: You can force the deletion by adding the --force flag.

If this is expected, re-running the same command while adding the `--force` flag
according to the hint will perform the deletion.
