<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable no-duplicate-heading -->
<!-- markdownlint-disable ul-indent -->
<!-- markdownlint-disable first-line-h1 -->
# CLI Reference

!!! note

    This page provides documentation for the `ado` CLI tool, which needs to be
    installed. If this is not the case, follow the instructions provided in
    [Getting Started](../user-guide/getting-started.md#installing)

**ado** comes with a CLI utility that is designed to be familiar for users of
`kubectl` and `oc`. It allows creating and retrieving
[resources](../resources/index.md), managing multiple backends, executing
[actuators](../user-guide/actuators/working-with-actuators.md), and more.

This page provides documentation for every command that we support, presented in
alphabetical order. Refer to the navigation pane on the left to go to the
section you are more interested in.

## ado

**ado** supports a set of generic options that are passed down to all the other
commands.

```commandline
ado [--context | -c <context-file.yaml>] \
    [--log-level | -l <value>]
```

- `--context | -c` allows overriding the active context with one loaded from a
  file. This feature should only be used when running on remote Ray clusters.
- `--log-level | -l` configures the logging level. This does not affect child
  processes.

### Resource Type Shorthands

Many ado CLI commands accept resource types as arguments. To make commands more
concise, ado supports **shorthand aliases** for resource type names. You can use
either the full name or the shorthand interchangeably in any command.

#### Available Shorthands

> [!WARNING]
>
> Shorthands are case-sensitive and must be lowercase.

| Full Resource Type    | Shorthand | Example Usage                       |
| --------------------- | --------- | ----------------------------------- |
| actuatorconfiguration | ac        | `ado get ac`                        |
| context               | ctx       | `ado delete ctx my-context`         |
| datacontainer         | dcr       | `ado describe dcr container-123`    |
| discoveryspace        | space     | `ado create space -f space.yaml`    |
| document              | doc       | `ado describe doc document-abc123`  |
| experiment            | exp       | `ado get exp`                       |
| operation             | op        | `ado get op operation-456`          |
| samplestore           | store     | `ado get store`                     |

#### Usage Examples

```shell
# These commands are equivalent:
ado get discoveryspace space-abc123
ado get space space-abc123

# These commands are equivalent:
ado create actuatorconfiguration -f config.yaml
ado create ac -f config.yaml

# These commands are equivalent:
ado delete operation op-xyz789
ado delete op op-xyz789
```

## ado context

**ado** supports storing configuration and authentication details for multiple
backends, which in ado terms are called **contexts**.

The complete syntax of the `ado context` command is as follows:

```shell
ado context [CONTEXT_NAME]
```

### Examples

#### Getting the current context

Similar to `oc project`, users can see the name of the currently active context
by running:

```shell
ado context
```

#### Listing available contexts

Similar to `oc projects`, users can list available contexts by running:

```shell
ado contexts
```

It's also possible to output this information in multiple formats via the
`-o/--output` flag:

```shell
# List context names only
ado contexts -o name

# Export contexts as YAML
ado contexts -o yaml

# Export contexts as JSON
ado contexts -o json
```

#### Switching between contexts

To switch between the available contexts, specify the target context name to the
`ado context` command. In this example we assume that the `my-context` context
exists:

```shell
ado context my-context
```

## ado create

The **ado** CLI provides the _create_ command to create
[resources](../resources/index.md) given a YAML file with their
configuration.

The complete syntax of the `ado create` command is as follows:

```shell
ado create RESOURCE_TYPE [--file | -f <FILE.yaml>] \
                         [--set <jsonpath=json-value>] \
                         [--with <resource=value>] \
                         [--new-sample-store] \
                         [--use-default-sample-store] [--dry-run]
```

Where:

- `RESOURCE_TYPE` is one of the supported resource types for `ado create`. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _actuator_
    - _actuatorconfiguration_ (_ac_)
    - _context_ (_ctx_)
    - _document_ (_doc_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

    <!-- prettier-ignore-end -->

- `--file` or `-f` is a path to the resource configuration file in YAML format.
  It is mandatory in all scenarios, except when running
  `ado create samplestore --new-sample-store`.
- `--set` allows overriding fields in the provided resource configuration. It
  supports using JSONPath syntax. See the examples section for more information.
- `--with` enables you to create resources together with other resources they
  depend on, or to reference existing resource identifiers during creation. For
  example, you can create a space along with a sample store definition, or
  create an operation together with an actuator configuration and a space
  definition. See the Examples section for more details.
- `--use-latest` allows reusing the latest identifier of a certain resource kind
  from the current context's metastore. It is only supported for spaces and
  operations during `ado create`. Ignored if `--with` is used.
- `--new-sample-store` creates a new sample store. Only available when running
  `ado create` on `space` and `samplestore`. If running
  `ado create space --new-sample-store`, the `sampleStoreIdentifier` contained
  in the `DiscoverySpaceConfiguration` will be disregarded. It is ignored if
  `--with` or `--use-latest` are used.
- `--use-default-sample-store` uses the default sample store. Only available
  when running `ado create space`. Alias for
  `--set sampleStoreIdentifier=default`. It is ignored if --with, --use-latest,
  or --new-sample-store are used.
- `--dry-run` is an **optional** flag to only validate the resource
  configuration file provided and not actually creating the resource.

### Examples

#### Creating a Discovery Space

In this example, we assume that the file `ds.yaml` exists and contains a valid
[Discovery Space](../resources/discovery-spaces.md) definition.

```shell
ado create -f ds.yaml
```

#### Validating a Sample Store definition

In this example, we assume that the file `sample-store.yml` exists, but we make
no further assumptions on whether its content is a valid
[Sample Store](../resources/sample-stores.md) definition or not.

```shell
ado create -f sample-store.yml --dry-run
```

#### Creating a new sample store with no file

```shell
ado create samplestore --new-sample-store
```

#### Creating a space with a new sample store

Note that if the space definition `ds.yaml` contains an `sampleStoreIdentifier`,
it will be ignored, and a new one will be created.

```shell
ado create space -f ds.yaml --new-sample-store
```

#### Create a space overriding the sample store identifier

```shell
ado create space -f ds.yaml --set "sampleStoreIdentifier=abcdef"
```

Another option is to use:

```shell
ado create space -f ds.yaml --with store=abcdef
```

#### Create a space while providing a sample store definition

```shell
ado create space -f ds.yaml --with store=store_definition.yaml
```

#### Create a space reusing the latest sample store identifier

```shell
ado create space -f ds.yaml --use-latest samplestore
```

#### Create a space renaming a property identifier in the space

```shell
ado create space -f ds.yaml --set "entitySpace[0].identifier=abcdef"
```

## ado delete

The **ado** CLI provides the delete command to delete
[resources](../resources/index.md) given their unique identifier.

The complete syntax of the `ado delete` command is as follows:

```shell
ado delete RESOURCE_TYPE RESOURCE_ID [RESOURCE_ID ...] \
           [--force] \
           [--delete-local-db] [--no-delete-local-db]
```

Where:

- `RESOURCE_TYPE` is the type of resource you want to delete. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _actuatorconfiguration_ (_ac_)
    - _context_ (_ctx_)
    - _datacontainer_ (_dcr_)
    - _document_ (_doc_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

    <!-- prettier-ignore-end -->

- `RESOURCE_ID` is the unique identifier of the resource to delete. Multiple
  resource IDs can be provided to delete multiple resources of the same type in
  a single command.
- `--force` enables forced deletion of resources in the following cases:

    <!-- prettier-ignore-start -->

    - When attempting to delete operations while other operations are executing.
    - When attempting to delete sample stores that still contain data.

    <!-- prettier-ignore-end -->

- When deleting a local context, users can specify the flags `--delete-local-db`
  or `--no-delete-local-db` to explicitly delete or preserve a local DB when
  deleting its related context. If neither of these flags are specified, the
  user will be asked whether to delete the DB or not.

### Examples

#### Deleting a context

```shell
ado delete context my-context
```

#### Deleting a local context and preserving the local db

```shell
ado delete context my-local-context --no-delete-local-db
```

#### Deleting a single space

```shell
ado delete space space-abc123-456def
```

#### Deleting multiple operations

```shell
ado delete operation op-id-1 op-id-2 op-id-3
```

#### Deleting multiple operations with force flag

```shell
ado delete operation op-id-1 op-id-2 op-id-3 --force
```

#### Deleting multiple discovery spaces

```shell
ado delete space space-1 space-2 space-3
```

## ado describe

**ado** provides the `describe` command to retrieve readable information about
resources.

The complete syntax of the `ado describe` command is as follows:

```shell
ado describe RESOURCE_TYPE [RESOURCE_ID] [--file | -f <file.yaml>] \
             [--use-latest]
```

Where:

- `RESOURCE_TYPE` is the type of resource you want to describe. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _experiment_
    - _datacontainer_ (_dcr_)
    - _discoveryspace_ (_space_)
    - _document_ (_doc_)

    <!-- prettier-ignore-end -->

- `RESOURCE_ID` is the unique identifier of the resource to describe.
- The `--file` (or `-f`) flag is **currently only available for spaces** and
  allows getting a description of the space, given a space configuration file.
- `--use-latest` flag is **currently only available for spaces** and allows
  describing the most recently created space from the current context.
- When describing an experiment, `RESOURCE_ID` may be a bare experiment
  identifier, include an `@M.m.p` version suffix, a major version identifier
  such as `@v1`, or use the form `actuator_id.experiment_id` when multiple
  actuators implement the same experiment identifier.

### Examples

#### Describing a Discovery Space

```shell
ado describe space space-abc123-456def
```

#### Describing a document

```shell
ado describe document document-abc12345
```

## ado edit

**ado** automatically stores metadata in the backend for some of the resources
you can create. The fastest way to update these metadata is to use the
`ado edit` command.

The complete syntax of the `ado edit` command is as follows:

```shell
ado edit RESOURCE_TYPE RESOURCE_ID [-p | --patch <YAML>] \
    [--patch-file <FILE>] [--editor <NAME>]
```

Where:

- `RESOURCE_TYPE` is the type of resource you want to edit. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _actuatorconfiguration_ (_ac_)
    - _datacontainer_ (_dcr_)
    - _document_ (_doc_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

    <!-- prettier-ignore-end -->

- `RESOURCE_ID` is the unique identifier of the resource to edit.
- `-p` / `--patch` is an optional inline YAML/JSON string for non-interactive
  editing (similar to `oc` / `kubectl patch -p`). It is **merged** into the
  resource's existing stored metadata using a one-level strategic update:
  `labels` are merged as key–value maps; other top-level fields from the patch
  replace the previous values. You can use a patch string **or** a file, not
  both. When using `--patch`, the `--editor` flag is ignored.
- `--patch-file` is an optional path to a YAML/JSON file with the same merge
  behaviour as `--patch`. You may not use `--patch` and `--patch-file` together.
  When using `--patch-file`, the `--editor` flag is ignored.
- `--editor` is the name of the editor you want to use for **interactive**
  editing of metadata (ignored when `--patch` or `--patch-file` is specified).
  It must be one of the supported ones, which currently are:

    <!-- prettier-ignore-start -->

    - `vim`
    - `vi`
    - `nano` (_default_ if `ADO_EDITOR` is not set when using interactive edit)

    <!-- prettier-ignore-end -->

For interactive mode, you can set the default editor with the `ADO_EDITOR`
environment variable.

### Examples

#### Editing an operation's metadata

```shell
ado edit operation randomwalk-0.5.0-123abc
```

#### Editing a space's metadata using a different editor

```shell
ado edit space space-abc123-456def --editor nano
```

#### Editing a space's metadata using a different editor (set by environment variable)

```shell
ADO_EDITOR=nano ado edit space space-abc123-456def
```

#### Merging metadata with an inline patch (non-interactive, oc-style)

```shell
ado edit space space-abc123-456def -p "labels: { team: front }"
```

#### Merging metadata from a file (non-interactive)

```shell
ado edit space space-abc123-456def --patch-file extra-metadata.yaml
```

## ado get

**ado** allows getting resources in a similar way to `kubectl`. Users can choose
to either get all resources of a given type or specify a resource identifier to
restrict results to a single resource.

The complete syntax of the `ado get` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado get RESOURCE_TYPE [RESOURCE_ID] [--output | -o <default | yaml | json | config | raw | stats>] \
                                    [--output-file <path>] \
                                    [--use-latest] \
                                    [--exclude-default | --no-exclude-default] \
                                    [--exclude-unset | --no-exclude-unset ] \
                                    [--exclude-none | --no-exclude-none ] \
                                    [--minimize] \
                                    [--filter | -q <path=value>] \
                                    [--label | -l <key=value>] \
                                    [--details] [--show-deprecated] \
                                    [--matching-point <point.yaml>] \
                                    [--matching-space <space.yaml>] \
                                    [--matching-space-id <space-id>] \
                                    [--related-to <kind=id>]
```

<!-- markdownlint-enable line-length -->

Where:

- `RESOURCE_TYPE` is the type of resource you want to get. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _actuatorconfiguration_ (_ac_)
    - _actuator_
    - _context_ (_ctx_)
    - _datacontainer_ (_dcr_)
    - _document_ (_doc_)
    - _experiment_ (_exp_)
    - _operation_ (_op_)
    - _operator_
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

    <!-- prettier-ignore-end -->

- `RESOURCE_ID` is the optional unique identifier of the resource to get. If not
  specified, all resources of the given type are returned (unless `--use-latest`
  is used).
- `--use-latest` retrieves the most recently created resource of the specified
  type. This flag is ignored if a `RESOURCE_ID` is also provided (the explicit
  ID takes precedence).
- `--output` or `-o` determine the type of output that will be displayed:

    <!-- prettier-ignore-start -->

    - The `table` format shows the _identifier_, the _name_, and the _age_ of
      the matching resources.
    - The `name` format outputs only the resource identifiers, one per line
      (similar to `kubectl get -o name`).
    - The `yaml` format displays the full YAML document of the matching resources.
    - The `json` format displays the full JSON document of the matching resources.
    - The `config` format displays the `config` field of the matching resources.
    - The `raw` format displays the raw resource as stored in the database,
      performing no validation.
    - The `stats` format augments the default `table` output with additional
      statistics columns. Supported resource types and their columns are:
        - **Operations**: `TOTAL_RESULTS`, `SUCCESSFUL_RESULTS`,
          `FAILED_RESULTS`, `MEASURED_ENTITIES`.
        - **Discovery Spaces**: `EXPERIMENTS`, `OPERATIONS`,
          `EXPLORE_OPERATIONS`, `MEASURED_ENTITIES`.
        - **Sample Stores**: `ENTITIES`, `RESULTS`, `EXPERIMENTS`.
        - **Data Containers**: `TABLES`, `LOCATIONS`, `KEY_VALUES`,
          `DATA_BYTES`.

    <!-- prettier-ignore-end -->

- `--output-file` allows writing the output to a specified file instead of
  stdout. Avoids truncating columns when used with the `table` format.
- `--exclude-default` (set by default) allows excluding fields that use default
  values from the output. Alternatively, the `--no-exclude-default` flag can be
  used to show them.
- `--exclude-unset` (set by default) allows excluding from the output fields
  whose values have not been set. Alternatively, the `--no-exclude-unset` flag
  can be used to show them.
- `--exclude-none` (set by default) allows excluding fields that have null
  values from the output. Alternatively, the `--no-exclude-none` flag can be
  used to show them.
- `--exclude-field` allows the user to exclude fields from the output using
  JSONPath expressions. Documentation for creating these expressions can be
  found here:
  <https://github.com/h2non/jsonpath-ng?tab=readme-ov-file#jsonpath-syntax>.
  This flag is only supported when using the `yaml`, `json`, or `config` output
  format.
- `--minimize` minimizes the output. This might entail applying transformations
  on the model, changing it from the original. If set, it implies
  `--exclude-default`, `--exclude-unset`, and `--exclude-none`. This option is
  ignored when the output type is `table` or `raw`.
- When using the `--details` flag with the `table` output format, additional
  columns with the _description_ and the _labels_ of the matching resources are
  printed.
- The `--show-deprecated` flag is available **only for `ado get experiments`**
  and allows displaying experiments that have been deprecated. They are
  otherwise hidden by default.
- `--related-to` filters results to resources reachable from a given anchor
  resource. The value must be in the form `kind=id`
  (e.g. `samplestore=store-abc123`). Using shorthand aliases is supported
  (e.g. `store=store-abc123`). This flag:
    - is **not** supported for `actuator`, `experiment`, `operator`, or `context`
      resource types.
    - cannot be combined with a direct `RESOURCE_ID` argument or `--use-latest`.
    - can be combined with `--filter`, `--label`, `--matching-point`,
      `--matching-space`, and `--matching-space-id` to further narrow results.

### Searching and Filtering

See [searching the metastore](../resources/metastore.md#searching-the-metastore)
for detailed information on the following options, including syntax.

- By using (optionally multiple times) the `--filter` (or `-q`) flag, users can
  restrict the resources returned by requiring that a field in the resource
  contains a given value. This flag can be specified multiple times (even in
  conjunction with `-l` to further filter results).
- By using (optionally multiple times) the `--label` (or `-l`) flag, users can
  restrict the resources returned by means of the labels set in the resource's
  metadata. Labels must be specified in the `key=value` format. This flag can be
  specified multiple times (even in conjunction with `-q` to further filter
  results).
- The `--matching-point` option allows the user to search for spaces containing
  an entity with particular values for some properties along with a particular
  set of experiments applied to it
- The `--matching-space` option allows searching for `discoveryspaces` which
  match a given
  [configuration YAML](../resources/discovery-spaces.md#discovery-space-configuration-yaml).
- The `--matching-space-id` option works in the same way as `--matching-space`
  but allows the user to provide a space id instead of a configuration

### Examples

#### Getting all Discovery Spaces

```shell
ado get spaces
```

#### Getting all Discovery Spaces with additional details

```shell
ado get spaces --details
```

<!-- markdownlint-disable line-length -->

#### Getting all Discovery Spaces that include granite-7b-base in the property domain

<!-- markdownlint-enable line-length -->

!!! info

    More information on field-level filtering is provided in the
    [searching the metastore](../resources/metastore.md#searching-the-metastore)
    section

```shell
ado get space --filter 'config.entitySpace={"propertyDomain":{"values":["granite-7b-base"]}}'
```

#### Getting all Discovery Spaces with certain labels

```shell
ado get spaces -l key1=value1 -l key2=value2
```

#### Getting all discovery spaces matching a point

Assuming you have the following file saved as `point.yaml`:

```yaml
entity: # A key-value dictionary of constitutive property identifiers and values
  batch_size: 8
  number_gpus: 4
experiments: # A list of experiments
  - finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0
```

You can run:

```shell
ado get spaces --matching-point point.yaml
```

#### Getting all DiscoverySpaces and hiding fields

This example shows how to hide the `propertyDomain.variableType` and
`propertyDomain.domainRange` fields from the Discovery Space's entity space:

```shell
ado get space space-df8077-7535f9 -o yaml \
  --exclude-field "config.entitySpace[*].propertyDomain.variableType" \
  --exclude-field "config.entitySpace[*].propertyDomain.domainRange"
```

<!-- markdownlint-disable line-length -->

#### Getting an actuator configuration and hiding the status for the "created" event

<!-- markdownlint-enable line-length -->

```shell
ado get actuatorconfiguration actuatorconfiguration-myactuator-123456 -o yaml \
  --exclude-field 'status[?(@.event="created")]'
```

#### Getting a single Operation

```shell
ado get operation randomwalk-0.5.0-123abc
```

#### Getting the YAML of a single Operation

```shell
ado get operation randomwalk-0.5.0-123abc -o yaml
```

#### Getting only the identifiers of all Operations

```shell
ado get operations -o name
```

#### Getting the latest Discovery Space as YAML

```shell
ado get space --use-latest -o yaml
```

#### Getting the latest Operation as YAML

```shell
ado get operation --use-latest -o yaml
```

#### Displaying all current experiments

```shell
ado get experiments --details
```

#### Saving a Discovery Space configuration to a file

```shell
ado get space my-space-id -o yaml --output-file space.yaml
```

#### Saving all operations as JSON

```shell
ado get operations -o json --output-file operations.json
```

#### Saving resource identifiers to a file

```shell
ado get spaces -o name --output-file space-ids.txt
```

#### Saving table output to a file

```shell
ado get spaces --output-file spaces-table.txt
```

#### Getting measurement statistics for all Operations

```shell
ado get operations -o stats
```

#### Getting measurement statistics for a single Operation

```shell
ado get operation randomwalk-0.5.0-123abc -o stats
```

#### Getting statistics for all Discovery Spaces

```shell
ado get spaces -o stats
```

#### Getting statistics for a single Discovery Space

```shell
ado get space --use-latest -o stats
```

#### Getting statistics for all Sample Stores

```shell
ado get samplestores -o stats
```

#### Getting statistics for a single Sample Store

```shell
ado get samplestore --use-latest -o stats
```

#### Getting statistics for all Data Containers

```shell
ado get datacontainers -o stats
```

#### Getting statistics for a single Data Container

```shell
ado get datacontainer --use-latest -o stats
```

#### Getting all operations linked to a sample store

```shell
ado get operations --related-to samplestore=store-abc123-456def
```

#### Getting all discovery spaces linked to a sample store

```shell
ado get spaces --related-to samplestore=store-abc123-456def -o name
```

#### Getting operations linked to a space, filtered by name

```shell
ado get operations --related-to discoveryspace=space-abc123-456def \
  --filter config.metadata.name=my-operation-name
```

#### Getting operations linked to a sample store using the shorthand alias

```shell
ado get operations --related-to store=store-abc123-456def
```

## ado show

When interacting with resources, we might be interested in seeing some of their
details, entities measured, or related resources. `ado show` provides this with
the four following subcommands.

### ado show measurements

_show measurements_ supports displaying measurement data (entities with their
measured properties) that belong to a space or an operation.

The complete syntax of the `ado show measurements` command is as follows:

```shell
ado show measurements RESOURCE_TYPE [RESOURCE_ID] [--use-latest] [--file | -f <file.yaml>]\
                      [--property-format {observed | target}] \
                      [--output | -o {table | csv | json}] \
                      [--output-file <path>] \
                      [--property <property-name>] \
                      [--include {sampled | matching | missing | unsampled}] \
                      [--aggregate {mean | median | variance | std | min | max}]
```

Where:

- `RESOURCE_TYPE` is one of the supported resource types. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _discoveryspace_ (_space_)
    - _operation_ (_op_)

    <!-- prettier-ignore-end -->

- `RESOURCE_ID` is the unique identifier of the resource you want to see
  measurements for.
- `--use-latest` will use the identifier of the latest (i.e. most recent)
  resource of RESOURCE_TYPE from the current context. It is ignored if a
  RESOURCE_ID is provided.
- The `--file` (or `-f`) flag is **currently only available for spaces** and
  enables showing measurements that match the space defined in the configuration
  file. **NOTE**: using this flag forces `--include matching`.
- `--property-format` defines the naming format used for measured properties in
  the output, one of:

    <!-- prettier-ignore-start -->

    - `observed`: properties are named `$experimentid.$property_id`.
      There will be one row per entity.
    - `target`: properties are named `$property_id`.
      There will be one row per (entity, experiment) pair.

    <!-- prettier-ignore-end -->

- `--output` (or `-o`) is the format in which to display the measurement data.
  One of:

    <!-- prettier-ignore-start -->

    - `table` (print to stdout)
    - `csv` (write CSV to stdout, or to file if `--output-file` is specified)
    - `json` (write JSON to stdout, or to file if `--output-file` is specified)

    <!-- prettier-ignore-end -->

- `--output-file` specifies a file path to write the output to (except for table
  format which always prints to stdout).

- `--property` (can be specified multiple times) is used to filter what measured
  properties need to be output.
- `--include` (**exclusive to spaces**) determines what type of entities to
  include. One of:

    <!-- prettier-ignore-start -->

    - `sampled`: Entities that have been measured by explore operations on the
      `discoveryspace`
    - `unsampled`: Entities that have not been measured by an explore operation
      on the `discoveryspace`
    - `matching`: Entities in the `samplestore` the `discoveryspace` uses, that
      match the `discoveryspace`'s description
    - `missing`: Entities in the `discoveryspace` that are not in the
      `samplestore` the `discoveryspace` uses

    <!-- prettier-ignore-end -->

- `--aggregate` allows applying an aggregation to the result values in case
  multiple are present. One of:

    <!-- prettier-ignore-start -->

    - `mean`
    - `median`
    - `variance`
    - `std`
    - `min`
    - `max`

    <!-- prettier-ignore-end -->

#### Examples

<!-- markdownlint-disable line-length -->

##### Show matching measurements in a Space with target format and output them as CSV

<!-- markdownlint-enable line-length -->

Recommended approach using `--output-file` (ensures columns aren't truncated and
handles file write errors):

```shell
 ado show measurements space space-abc123-456def --include matching \
                                                 --property-format target \
                                                 -o csv --output-file entities.csv
```

Or to write CSV to stdout for piping:

```shell
 ado show measurements space space-abc123-456def --include matching \
                                                 --property-format target \
                                                 -o csv > entities.csv
```

<!-- markdownlint-disable line-length -->

##### Show a subset of the properties of measurements that are part of an operation and output them as JSON

```shell
ado show measurements operation randomwalk-0.5.0-123abc -o json \
                                                        --property my-property-1 \
                                                        --property my-property-2
```

<!-- markdownlint-enable line-length -->

##### Save table output of measurements to a file

```shell
ado show measurements space space-abc123-456def --output-file entities-table.txt
```

### ado show trace

_show trace_ allows inspecting in detail the trace of entity measurement
requests made during explore operations. It can provide crucial information for
debugging operation behaviour e.g. failed experiments or requests. Note,
multiple entities can be contained in a single measurement request depending on
the sampler used to explore and its settings.

`ado show trace` supports three resource types:

- `operation` — show the trace for a single operation.
- `discoveryspace` — show the aggregated trace for all operations linked to a
  discovery space.
- `samplestore` — show the aggregated trace for all operations linked to a
  sample store (across all its discovery spaces).

The complete syntax of the `ado show trace` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado show trace operation|discoveryspace|samplestore [RESOURCE_ID] [--use-latest] \
                         [--unroll-entities] \
                         [--filter <key=value>] \
                         [--output | -o <table | csv | json | yaml>] \
                         [--output-file <path>] \
                         [--hide <field>] \
                         [--no-trunc]
```

<!-- markdownlint-enable line-length -->

- `--use-latest` will use the identifier of the latest (i.e. most recent)
  resource of the selected type from the current context. It is ignored if a
  RESOURCE_ID is provided.
- `--unroll-entities` expands the table for output mode `table` or `csv` so each
  entity has its own row containing additional metadata on the result of
  applying the requested experiment to it.
- `--filter` filters based on field names from the underlying data model, not
  table column names. Can be used multiple times for AND logic.
- `--output` (or `-o`) determines the output format. Supports `table`, `csv`,
  `json`, and `yaml`. Output is written to stdout by default, or to a file if
  `--output-file` is specified.
- `--output-file` specifies a file path to write the output to. If not provided,
  output is written to stdout.
- `--hide` can be specified multiple times and allows hiding fields from the
  output.
- `--no-trunc` prevents truncation of table content (console output only).

#### Default Trace Output Table

The default output table shows the time-series of measurement requests with the
following columns:

- Index (auto-generated row number)
- Request ID
- Operation ID _(multi-operation views only: `discoveryspace`, `samplestore`)_
- Space ID _(samplestore view only)_
- Request Index
- Request type (measured/replayed)
- Timestamp
- Experiment ID
- Entity IDs (list)
- Status
- Measurements (count)
- Valid Measurements (count)
- Invalid Measurements (count)
- Metadata (request metadata)

#### Expanded Trace Output Table

Specifying `--unroll-entities` unrolls each request so each entity with a
request processed has its own row containing additional metadata on the result
of applying the requested experiment to it:

- Index (auto-generated row number)
- Request ID
- Operation ID _(multi-operation views only: `discoveryspace`, `samplestore`)_
- Space ID _(samplestore view only)_
- Request Index
- Request type (measured/replayed)
- Timestamp
- Experiment ID
- Result Index (per-request, 0-based)
- Result UID
- Entity ID (single, unrolled)
- Valid (boolean)
- Number of Properties
- Invalid Reason
- Request Metadata
- Result Metadata

### Filtering Output

The output of show trace can be filtered using the following fields:

- `requestIndex` - Request index number
- `requestid` - Request UUID
- `status` - Request status (Unknown, Success, Failed)
- `timestamp` - Request timestamp
- `metadata` - Request metadata (use dot notation for nested fields, e.g.,
  `metadata.key=value`)
- `experimentReference` - Stringified experiment reference

Filtering reduces the output to the requests matching the filters.

#### Examples

##### Show the trace for an operation as a table

```shell
ado show trace operation randomwalk-0.5.0-123abc
```

##### Show entity level information in the trace table

```shell
ado show trace operation randomwalk-0.5.0-123abc --unroll-entities
```

##### Show the YAML of a request

```shell
ado show trace operation randomwalk-0.5.0-123abc --filter requestid=abcdef -o yaml
```

##### Filter trace on multiple request fields

<!-- markdownlint-disable line-length -->

```shell
ado show trace operation randomwalk-0.5.0-123abc --filter status=Success --filter requestIndex=5
```

<!-- markdownlint-enable line-length -->

##### Hide specific columns

```shell
ado show trace operation randomwalk-0.5.0-123abc --hide metadata --hide timestamp
```

##### Show the aggregated trace for all operations in a discovery space

```shell
ado show trace discoveryspace my-space-123abc
```

##### Show the aggregated trace for all discovery spaces sharing a sample store

```shell
ado show trace samplestore my-store-456def
```

##### Export the aggregated trace for a discovery space to CSV

<!-- markdownlint-disable line-length -->

```shell
ado show trace discoveryspace my-space-123abc -o csv --output-file trace.csv
```

<!-- markdownlint-enable line-length -->

### ado show related

_show related_ supports displaying resources that are related to the one whose
id is provided (e.g., operations run on a space). By default the full resource
graph is traversed in both directions.

The complete syntax of the `ado show related` command is as follows:

```shell
ado show related RESOURCE_TYPE [RESOURCE_ID] [--use-latest] [--max-hops N]
```

- `RESOURCE_TYPE` is one of the supported resource types. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _actuatorconfiguration_ (_ac_)
    - _datacontainer_ (_dcr_)
    - _discoveryspace_ (_space_)
    - _document_ (_doc_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)

    <!-- prettier-ignore-end -->

- `RESOURCE_ID` is the unique identifier of the resource you want to see related
  resources for.
- `--use-latest` will use the identifier of the latest (i.e. most recent)
  resource of RESOURCE_TYPE from the current context. It is ignored if a
  RESOURCE_ID is provided.
- `--max-hops N` limits the traversal to at most `N` relationship hops in each
  direction (1-10). When omitted, the full graph depth is used.

#### Examples

##### Show resources related to a discovery space

```shell
ado show related space space-abc123-456def
```

##### Show only directly linked resources (1 hop)

```shell
ado show related space space-abc123-456def --max-hops 1
```

### ado show stats

_show stats_ supports displaying in-depth statistics for one or more resources.
It always outputs the same base columns as the corresponding `ado get` table
(e.g. `IDENTIFIER`, `NAME`, `AGE`) plus a richer set of statistics columns than
the `ado get -o stats` view. For operations, it also includes request-level
columns; for discovery spaces, it also includes full entity-space coverage
columns.

The complete syntax of the `ado show stats` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado show stats RESOURCE_TYPE [IDS...] [--use-latest] \
               [--query | -q <path=value>] \
               [--label | -l <key=value>] \
               [--details] \
               [--output | -o <table|md-table|csv|json|yaml>] \
               [--output-file <path>] \
               [--render]
```

<!-- markdownlint-enable line-length -->

Where:

- `RESOURCE_TYPE` is one of the supported resource types. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _discoveryspace_ (_space_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _datacontainer_ (_dcr_)

    <!-- prettier-ignore-end -->

- `IDS` is an optional list of one or more resource identifiers to show
  statistics for. If omitted, statistics are shown for all resources of the
  given type.
- `--use-latest` shows statistics for the most recently created resource of the
  selected type. Ignored if resource identifiers are also specified.
- `--filter` (or `-q`) and `--label` (or `-l`) filter which resources are
  included (same semantics as `ado get`). Cannot be used together with explicit
  IDs.
- `--details` appends `DESCRIPTION` and `LABELS` columns to the output,
  mirroring the behaviour of `ado get --details`.
- `--output` (or `-o`) selects the output format:

    <!-- prettier-ignore-start -->

    - `table` (**default**) — rich console table.
    - `md-table` — Markdown table.
    - `csv` — CSV format.
    - `json` — JSON.
    - `yaml` — YAML.

    <!-- prettier-ignore-end -->

- `--output-file` writes the output to the specified file instead of stdout.
- `--render` renders the output in the console. Only supported for `md-table`
  output.

The statistics columns produced per resource type are:

<!-- prettier-ignore-start -->

- **Operations**: `TOTAL_RESULTS`, `SUCCESSFUL_RESULTS`, `FAILED_RESULTS`,
  `MEASURED_ENTITIES`, `TOTAL_REQUESTS`, `FAILED_REQUESTS`,
  `SUCCESSFUL_REQUESTS`.
- **Discovery Spaces**: `EXPERIMENTS`, `OPERATIONS`, `EXPLORE_OPERATIONS`,
  `MEASURED_ENTITIES`, `SIZE_OF_ENTITY_SPACE`, `UNMEASURED_ENTITIES`,
  `MATCHING_ENTITIES`, `MATCHING_WITH_MEASUREMENTS`,
  `ENTITIES_WITH_ALL_MEASUREMENTS`, `ENTITIES_WITH_PARTIAL_MEASUREMENTS`,
  `MATCHING_ENTITIES_WITH_ALL_MEASUREMENTS`.
- **Sample Stores**: `ENTITIES`, `RESULTS`, `EXPERIMENTS`.
- **Data Containers**: `TABLES`, `LOCATIONS`, `KEY_VALUES`, `DATA_BYTES`.

<!-- prettier-ignore-end -->

!!! note

    `ado show stats discoveryspace` computes full entity-space coverage
    statistics including the heavy columns (`SIZE_OF_ENTITY_SPACE`,
    `UNMEASURED_ENTITIES`, `MATCHING_ENTITIES`, etc.). This is slower than
    `ado get spaces -o stats` because it instantiates each
    `DiscoverySpace` and queries the sample store. Use `ado get -o stats` for
    quick overviews.

#### Examples

##### Show full statistics for all operations

```shell
ado show stats operation
```

##### Show full statistics for a specific discovery space

```shell
ado show stats discoveryspace space-abc123-456def
```

##### Show full statistics for the latest operation as JSON

```shell
ado show stats operation --use-latest -o json
```

##### Show full statistics for the latest operation as YAML

```shell
ado show stats operation --use-latest -o yaml
```

##### Show full statistics for sample stores matching a label

```shell
ado show stats samplestore -l team=research
```

##### Show full statistics for all discovery spaces as a Markdown table

```shell
ado show stats discoveryspace -o md-table
```

##### Save full statistics to a file

```shell
ado show stats operation --output-file operations-stats.csv -o csv
```

## ado template

To assist in creating a resource configuration file, we typically start from a
reference file. The `ado template` command allows you to create template files
that you can edit to streamline the process.

The complete syntax of the `ado template` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado template RESOURCE_TYPE [--output-file <PATH>] \
                           [--include-schema] \
                           [--operator-name <NAME>] \
                           [--operator-type <TYPE>] \
                           [--actuator-identifier <NAME>] \
                           [--from-experiment | -e <experiment_id>] \
                           [--local-context] \
                           [--no-parameters-only-schema]
```

<!-- markdownlint-enable line-length -->

Where:

- `RESOURCE_TYPE` is one of the supported resource types. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _actuatorconfiguration_ (_ac_)
    - _context_ (_ctx_)
    - _discoveryspace_ (_space_)
    - _document_ (_doc_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)

    <!-- prettier-ignore-end -->

- `--output-file` can be used to specify a file path where the template will be
  saved. If not specified, the template will be written to stdout.
- `--include-schema`, if set, will also produce the JSON Schema of the resource
  the template was generated for.
- `--operator-name` (**exclusive for operations**) allows generating an
  operation template with a specific operator. If unset, a generic operation
  will instead be output.
- `--operator-type` (**exclusive for operations**) is the type of operator to
  generate a template for. Must be one of the supported operator types:

    <!-- prettier-ignore-start -->

    - `characterize`
    - `explore`
    - `compare`
    - `modify`
    - `study`
    - `fuse`
    - `learn`

    <!-- prettier-ignore-end -->

- `--actuator-configuration` (**exclusive for actuatorconfigurations**) is the
  identifier of the actuator to output. If unset, a generic actuator
  configuration will be output.
- `--from-experiment` (**exclusive for spaces**) is the identifier of the
  experiment you want in your space. It may include an `@M.m.p` version suffix,
  a major version identifier such as `@v1`, or use the form
  `actuator_id.experiment_id` when multiple actuators implement the same
  experiment identifier. If unset, a generic space will be output.
- `--local-context` (**exclusive for contexts**) creates a template using SQLite
  instead of MySQL.
- `--no-parameters-only-schema` (**exclusive for operations**) when used with
  `--include-schema`, outputs a generic operation schema. By default (when not
  specifying this flag), the schema will be operator-specific.

### Examples

#### Creating a template for a context

```shell
ado template context
```

#### Creating a template for a space that uses a specific experiment

```shell
ado template space --from-experiment finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0
```

<!-- markdownlint-disable line-length -->

#### Creating a template for a space that uses a specific experiment from a specific actuator

<!-- markdownlint-enable line-length -->

```shell
ado template space --from-experiment SFTTrainer.finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0
```

#### Creating a template for a Discovery Space with the schema

```shell
ado template space --include-schema
```

#### Creating an operation template for the Rifferla operator

```shell
ado template operation --operator-name rifferla
```

## ado upgrade

!!! tip

    **`ado` will detect automatically when resource upgrades are required**
    and will print the exact command to run as a warning. In all other cases, there
    is no need to run this command.

Sometimes the models that are used in ado undergo changes that require updating
stored representations of them in the [metastore](../resources/metastore.md).
When required, you can run this command to update all resources of a given kind
in the database.

```shell
ado upgrade RESOURCE_TYPE [--upgrade-entities-and-results]
```

Where:

- `RESOURCE_TYPE` is one of the supported resource types. See
  [Resource Type Shorthands](#resource-type-shorthands) for shorthand aliases.
  Currently supported:

    <!-- prettier-ignore-start -->

    - _actuatorconfiguration_ (_ac_)
    - _datacontainer_ (_dcr_)
    - _discoveryspace_ (_space_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)

    <!-- prettier-ignore-end -->

- `--upgrade-entities-and-results` is an **optional** flag, only applicable to
  `samplestore`. When passed, also upgrades stored entities and measurement
  results in each sample store. Omit this flag for a fast metadata-only upgrade;
  pass it only when entity or measurement result schemas have changed, as it can
  take a long time for large stores.

### Examples

#### Upgrade all operation resources

```shell
ado upgrade operations
```

#### Upgrade all sample store resources (metadata only)

```shell
ado upgrade samplestores
```

#### Upgrade all sample store resources including entities and measurement results

```shell
ado upgrade samplestores --upgrade-entities-and-results
```

## ado version

When unsure about what ado version you are running, you can get this information
with:

```shell
ado version
```

## What's next

<!-- prettier-ignore-start -->
<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Let's get started!**

    ---

    Jump into our examples

    [Our how-tos :octicons-arrow-right-24:](../user-guide/examples/index.md)

- :octicons-workflow-24:{ .lg .middle } **Learn more about the built-in Operators**

      ---

      Learn what `ado`'s built-in operators can offer you

      [Follow the guide :octicons-arrow-right-24:](../user-guide/operators/working-with-operators.md)

</div>
<!-- markdownlint-enable line-length -->

<!-- prettier-ignore-end -->
