<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable no-duplicate-heading -->
<!-- markdownlint-disable ul-indent -->
<!-- markdownlint-disable-next-line first-line-h1 -->
!!! note

    This page provides documentation for the `ado` CLI tool, which needs to be
    installed. If this is not the case, follow the instructions provided in
    [Installation](install.md)

**ado** comes with a CLI utility that is designed to be familiar for users of
`kubectl` and `oc`. It allows creating and retrieving
[resources](../resources/resources.md), managing multiple backends, executing
[actuators](../actuators/working-with-actuators.md), and more.

This page provides documentation for every command that we support, presented in
alphabetical order. Refer to the navigation pane on the left to go to the
section you are more interested in.

## CLI reference

### ado

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

### ado context

**ado** supports storing configuration and authentication details for multiple
backends, which in ado terms are called **contexts**.

The complete syntax of the `ado context` command is as follows:

```shell
ado context [CONTEXT_NAME]
```

#### Examples

##### Getting the current context

Similar to `oc project`, users can see the name of the currently active context
by running:

```shell
ado context
```

##### Listing available contexts

Similar to `oc projects`, users can list available contexts by running:

```shell
ado contexts
```

The default context will also be printed out.

##### Switching between contexts

To switch between the available contexts, specify the target context name to the
`ado context` command. In this example we assume that the `my-context` context
exists:

```shell
ado context my-context
```

### ado create

The **ado** CLI provides the _create_ command to create
[resources](../resources/resources.md) given a YAML file with their
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

- `RESOURCE_TYPE` is one of the supported resource types for `ado create`,
  currently:

    - _actuator_
    - _actuatorconfiguration_ (_ac_)
    - _context_ (_ctx_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

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
- `--use-latest` allows reusing the previous identifier of a certain resource
  kind. It is only supported for spaces and operations. The latest identifiers
  are updated every time an `ado create` command is successful. The stored
  identifiers are not per-context, meaning that, for example running
  `ado create samplestore`, changing context, and running
  `ado create --use-latest samplestore` will raise an error. Ignored if `--with`
  is used.
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

#### Examples

##### Creating a Discovery Space

In this example, we assume that the file `ds.yaml` exists and contains a valid
[Discovery Space](../resources/discovery-spaces.md) definition.

```shell
ado create -f ds.yaml
```

##### Validating a Sample Store definition

In this example, we assume that the file `sample-store.yml` exists, but we make
no further assumptions on whether its content is a valid
[Sample Store](../resources/sample-stores.md) definition or not.

```shell
ado create -f sample-store.yml --dry-run
```

##### Creating a new sample store with no file

```shell
ado create samplestore --new-sample-store
```

##### Creating a space with a new sample store

Note that if the space definition `ds.yaml` contains an `sampleStoreIdentifier`,
it will be ignored, and a new one will be created.

```shell
ado create space -f ds.yaml --new-sample-store
```

##### Create a space overriding the sample store identifier

```shell
ado create space -f ds.yaml --set "sampleStoreIdentifier=abcdef"
```

Another option is to use:

```shell
ado create space -f ds.yaml --with store=abcdef
```

##### Create a space while providing a sample store definition

```shell
ado create space -f ds.yaml --with store=store_definition.yaml
```

##### Create a space reusing the latest sample store identifier

```shell
ado create space -f ds.yaml --use-latest samplestore
```

##### Create a space renaming a property identifier in the space

```shell
ado create space -f ds.yaml --set "entitySpace[0].identifier=abcdef"
```

### ado delete

The **ado** CLI provides the delete command to delete
[resources](../resources/resources.md) given their unique identifier.

The complete syntax of the `ado delete` command is as follows:

```shell
ado delete RESOURCE_TYPE RESOURCE_ID \
           [--force] \
           [--delete-local-db] [--no-delete-local-db]
```

Where:

- `RESOURCE_TYPE` is the type of resource you want to delete. Currently, the
  only supported types are:

    - _actuatorconfiguration_ (_ac_)
    - _context_ (_ctx_)
    - _datacontainer_ (_dcr_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

- `RESOURCE_ID` is the unique identifier of the resource to delete.
- `--force` enables forced deletion of resources in the following cases:
    - When attempting to delete operations while other operations are executing.
    - When attempting to delete sample stores that still contain data.
- When deleting a local context, users can specify the flags `--delete-local-db`
  or `--no-delete-local-db` to explicitly delete or preserve a local DB when
  deleting its related context. If neither of these flags are specified, the
  user will be asked whether to delete the DB or not.

#### Examples

##### Deleting a context

```shell
ado delete context my-context
```

##### Deleting a local context and preserving the local db

```shell
ado delete context my-local-context --no-delete-local-db
```

##### Deleting a space

```shell
ado delete space space-abc123-456def
```

### ado describe

**ado** provides the `describe` command to retrieve readable information about
resources.

The complete syntax of the `ado describe` command is as follows:

```shell
ado describe RESOURCE_TYPE [RESOURCE_ID] [--file | -f <file.yaml>] \
             [--use-latest] [--actuator-id <actuator>]
```

Where:

- `RESOURCE_TYPE` is the type of resource you want to describe. Currently, the
  supported resource types are:

    - _experiment_
    - _datacontainer_ (_dcr_)
    - _discoveryspace_ (_space_)

- `RESOURCE_ID` is the unique identifier of the resource to describe.
- The `--file` (or `-f`) flag is **currently only available for spaces** and
  allows getting a description of the space, given a space configuration file.
- `--use-latest` flag is **currently only available for spaces** and allows
  describing the latest space created locally. It is not context aware.
- `--actuator-id` (**optional**) can be used only when the resource type is
  experiment and is used to indicate what actuator the experiment belongs to.

#### Examples

##### Describing a Discovery Space

```shell
ado describe space space-abc123-456def
```

### ado edit

**ado** automatically stores metadata in the backend for some of the resources
you can create. The fastest way to update these metadata is to use the
`ado edit` command.

The complete syntax of the `ado edit` command is as follows:

```shell
ado edit RESOURCE_TYPE RESOURCE_ID [--editor <NAME>]
```

Where:

- `RESOURCE_TYPE` is the type of resource you want to edit. Supported types are:

    - _actuatorconfiguration_ (_ac_)
    - _datacontainer_ (_dcr_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

- `RESOURCE_ID` is the unique identifier of the resource to edit.
- `--editor` is the name of the editor you want to use for editing metadata. It
  must be one of the supported ones, which currently are:

    - `vim` (_default_)
    - `vi`
    - `nano`

  Alternatively, you can also set the value for this flag by using the
  environment variable `ADO_EDITOR`.

#### Examples

##### Editing an operation's metadata

```shell
ado edit operation randomwalk-0.5.0-123abc
```

##### Editing a space's metadata using a different editor

```shell
ado edit space space-abc123-456def --editor nano
```

##### Editing a space's metadata using a different editor (set by environment variable)

```shell
ADO_EDITOR=nano ado edit space space-abc123-456def
```

### ado get

**ado** allows getting resources in a similar way to `kubectl`. Users can choose
to either get all resources of a given type or specify a resource identifier to
restrict results to a single resource.

The complete syntax of the `ado get` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado get RESOURCE_TYPE [RESOURCE_ID] [--output | -o <default | yaml | json | config | raw>] \
                                    [--exclude-default | --no-exclude-default] \
                                    [--exclude-unset | --no-exclude-unset ] \
                                    [--exclude-none | --no-exclude-none ] \
                                    [--minimize] \
                                    [--query | -q <path=value>] \
                                    [--label | -l <key=value>] \
                                    [--details] [--show-deprecated] \
                                    [--matching-point <point.yaml>] \
                                    [--matching-space <space.yaml] \
                                    [--matching-space-id <space-id>] \
                                    [--from-sample-store <sample-store-id>] \
                                    [--from-space <space-id>] [--from-operation <operation-id>]
```

<!-- markdownlint-enable line-length -->

Where:

- `RESOURCE_TYPE` is the type of resource you want to get. Currently, the only
  supported types are:

    - _actuatorconfiguration_ (_ac_)
    - _actuator_
    - _context_ (_ctx_)
    - _datacontainer_ (_dcr_)
    - _operation_ (_op_)
    - _operator_
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

- `RESOURCE_ID` is the optional unique identifier of the resource to get.
- `--output` or `-o` determine the type of output that will be displayed:

    - The `default` format shows the _identifier_, the _name_, and the _age_ of
      the matching resources.
    - The `yaml` format displays the full YAML document of the matching resources.
    - The `json` format displays the full JSON document of the matching resources.
    - The `config` format displays the `config` field of the matching resources.
    - The `raw` format displays the raw resource as stored in the database,
      performing no validation.

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
  ignored when the output type is `default` or `raw`.
- The `--from-sample-store`, `--from-space`, `--from-operation` flags are
  available **only for `ado get measurementrequests`** and allow specifying what
  samplestore/space/operation the measurement request belongs to.
- When using the `--details` flag with the `default` output format, additional
  columns with the _description_ and the _labels_ of the matching resources are
  printed.
- The `--show-deprecated` flag is available **only for
  `ado get actuators --details`** and allows displaying experiments that have
  been deprecated. They are otherwise hidden by default.

#### Searching and Filtering

See [searching the metastore](../resources/metastore.md#searching-the-metastore)
for detailed information on the following options, including syntax.

- By using (optionally multiple times) the `--query` (or `-q`) flag, users can
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

#### Examples

##### Getting all Discovery Spaces

```shell
ado get spaces
```

##### Getting all Discovery Spaces with additional details

```shell
ado get spaces --details
```

<!-- markdownlint-disable line-length -->
##### Getting all Discovery Spaces that include granite-7b-base in the property domain
<!-- markdownlint-enable line-length -->

!!! info

    More information on field-level querying is provided in the
    [searching the metastore](../resources/metastore.md#searching-the-metastore)
    section

```shell
ado get space -q 'config.entitySpace={"propertyDomain":{"values":["granite-7b-base"]}}'
```

##### Getting all Discovery Spaces with certain labels

```shell
ado get spaces -l key1=value1 -l key2=value2
```

##### Getting all discovery spaces matching a point

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

##### Getting all DiscoverySpaces and hiding fields

This example shows how to hide the `propertyDomain.variableType` and
`propertyDomain.domainRange` fields from the Discovery Space's entity space:

```shell
ado get space space-df8077-7535f9 -o yaml \
  --exclude-field "config.entitySpace[*].propertyDomain.variableType" \
  --exclude-field "config.entitySpace[*].propertyDomain.domainRange"
```

<!-- markdownlint-disable line-length -->
##### Getting an actuator configuration and hiding the status for the "created" event
<!-- markdownlint-enable line-length -->

```shell
ado get actuatorconfiguration actuatorconfiguration-myactuator-123456 -o yaml \
  --exclude-field 'status[?(@.event="created")]'
```

##### Getting a single Operation

```shell
ado get operation randomwalk-0.5.0-123abc
```

##### Getting the YAML of a single Operation

```shell
ado get operation randomwalk-0.5.0-123abc -o yaml
```

##### Displaying all current experiments

```shell
ado get actuators --details
```

##### Displaying all experiments for the st4sd actuator

```shell
ado get actuator st4sd --details --show-deprecated
```

##### Getting the yaml of a MeasurementRequest from an operation

<!-- markdownlint-disable line-length -->

```shell
ado get measurementrequest measurement-request-123 --from-operation randomwalk-0.5.0-123abc -o yaml
```

<!-- markdownlint-enable line-length -->

### ado show

When interacting with resources, we might be interested in seeing some of their
details, entities measured, or related resources. `ado show` provides this with
the four following subcommands.

#### ado show details

_show details_ supports displaying aggregate details about resources and related
resources.

The complete syntax of the `ado show details` command is as follows:

```shell
ado show details RESOURCE_TYPE [RESOURCE_ID] [--use-latest]
```

Where:

- `RESOURCE_TYPE` is one of the supported resource types:

    - _operation_ (_op_)
    - _discoveryspace_ (_space_)

- `RESOURCE_ID` is the unique identifier of the resource you want to see details
  for.
- `--use-latest` will use the identifier of the latest (i.e. most recent)
  resource of RESOURCE_TYPE created locally. It is not context aware. It is
  ignored if a RESOURCE_ID is provided.

##### Examples

###### Show details for a space

```shell
ado show details space space-abc123-456def
```

###### Show details for the latest space

```shell
ado show details space --use-latest
```

#### ado show entities

_show entities_ supports displaying entities that belong to a space or an
operation.

The complete syntax of the `ado show entities` command is as follows:

```shell
ado show entities RESOURCE_TYPE [RESOURCE_ID] [--use-latest] [--file | -f <file.yaml>]\
                  [--property-format {observed | target}] \
                  [--output-format {console | csv | json}] \
                  [--property <property-name>] \
                  [--include {sampled | matching | missing | unsampled}] \
                  [--aggregate {mean | median | variance | std | min | max}]
```

Where:

- `RESOURCE_TYPE` is one of the supported resource types:

    - _operation_ (_op_)
    - _discoveryspace_ (_space_)

- `RESOURCE_ID` is the unique identifier of the resource you want to see
  entities for.
- `--use-latest` will use the identifier of the latest (i.e. most recent)
  resource of RESOURCE_TYPE created locally. It is not context aware. It is
  ignored if a RESOURCE_ID is provided.
- The `--file` (or `-f`) flag is **currently only available for spaces** and
  enables showing entities that match the space defined in the configuration
  file. **NOTE**: using this flag forces `--include matching`.
- `--property-format` defines the naming format used for measured properties in
  the output, one of:

    - `observed`: properties are named `$experimentid.$property_id`.
      There will be one row per entity.
    - `target`: properties are named `$property_id`.
      There will be one row per (entity, experiment) pair.

- `--output-format` is the format in which to display the entity data. One of:

    - `console` (print to stdout)
    - `csv` (output as CSV file)
    - `json` (output as JSON file)

- `--property` (can be specified multiple times) is used to filter what measured
  properties need to be output.
- `--include` (**exclusive to spaces**) determines what type of entities to
  include. One of:

    - `sampled`: Entities that have been measured by explore operations on the
      `discoveryspace`
    - `unsampled`: Entities that have not been measured by an explore operation
      on the `discoveryspace`
    - `matching`: Entities in the `samplestore` the `discoveryspace` uses, that
      match the `discoveryspace`'s description
    - `missing`: Entities in the `discoveryspace` that are not in the
      `samplestore` the `discoveryspace` uses

- `--aggregate` allows applying an aggregation to the result values in case
  multiple are present. One of:
    - `mean`
    - `median`
    - `variance`
    - `std`
    - `min`
    - `max`

##### Examples

###### Show matching entities in a Space with target format and output them as CSV

```shell
 ado show entities space space-abc123-456def --include matching \
                                             --property-format target \
                                             --output-format csv
```

<!-- markdownlint-disable-next-line line-length -->
###### Show a subset of the properties of entities that are part of an operation and output them as JSON

```shell
ado show entities operation randomwalk-0.5.0-123abc --output-format json \
                                                    --property my-property-1 \
                                                    --property my-property-2
```

#### ado show requests

_show requests_ supports displaying the `MeasurementRequest`s that were part of
an operation.

The complete syntax of the `ado show requests` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado show requests operation [RESOURCE_ID] [--use-latest] \
                            [--output-format | -o <console | csv | json>] \
                            [--hide <field>]
```

<!-- markdownlint-enable line-length -->

- `--use-latest` will use the identifier of the latest (i.e. most recent)
  operation created locally. It is not context aware. It is ignored if a
  RESOURCE_ID is provided.
- `--output-format` determines whether the output will be printed to console or
  saved to a file.
- `--hide` can be specified multiple times and allows hiding fields from the
  output.

##### Examples

###### Show measurement requests for an operation and save them as csv

```shell
ado show requests operation randomwalk-0.5.0-123abc -o csv
```

###### Show measurement requests for an operation and hide certain fields

<!-- markdownlint-disable line-length -->

```shell
ado show requests operation randomwalk-0.5.0-123abc --hide type --hide "experiment id"
```

<!-- markdownlint-enable line-length -->

#### ado show results

_show results_ supports displaying the `MeasurementResult`s that were part of an
operation.

The complete syntax of the `ado show results` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado show results operation [RESOURCE_ID] [--use-latest] \
                           [--output-format | -o <console | csv | json>] \
                           [--hide <field>]
```

<!-- markdownlint-enable line-length -->

- `--use-latest` will use the identifier of the latest (i.e. most recent)
  operation created locally. It is not context aware. It is ignored if a
  RESOURCE_ID is provided.
- `--output-format` determines whether the output will be printed to console or
  saved to a file.
- `--hide` can be specified multiple times and allows hiding fields from the
  output.

##### Examples

###### Show measurement results for an operation

```shell
ado show results operation randomwalk-0.5.0-123abc -o csv
```

###### Show measurement results for an operation and hide certain fields

```shell
ado show results operation randomwalk-0.5.0-123abc --hide uid
```

#### ado show related

_show related_ supports displaying resources that are related to the one whose
id is provided (e.g., operations run on a space).

The complete syntax of the `ado show related` command is as follows:

```shell
ado show related RESOURCE_TYPE [RESOURCE_ID] [--use-latest]
```

- `RESOURCE_TYPE` is one of the supported resource types:

    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

- `RESOURCE_ID` is the unique identifier of the resource you want to see related
  resources for.
- `--use-latest` will use the identifier of the latest (i.e. most recent)
  resource of RESOURCE_TYPE created locally. It is not context aware. It is
  ignored if a RESOURCE_ID is provided.

##### Examples

###### Show resources related to a discovery space

```shell
 ado show related space space-abc123-456def
```

#### ado show summary

_show summary_ supports generating overviews about discovery spaces. The content
can be provided in the following formats:

- Markdown table (for high level overviews).
- Markdown text (for an easy to read and more in-depth format)
- CSV

The complete syntax of the `ado show summary` command is as follows:

```shell
ado show summary RESOURCE_TYPE [RESOURCE_IDS...] [--use-latest] \
                 [--query | -q <path=candidate>] \
                 [--label | -l <LABEL> ] \
                 [--with-property | -p <PROPERTY> ] \
                 [--format | -o <md | table | csv>]
```

Where:

- `RESOURCE_TYPE` is always _discoveryspace_ (_space_)
- `RESOURCE_IDS` are one or more space-separated space identifiers.
- `--use-latest` will add the identifier of the latest (i.e. most recent) space
  created locally to the RESOURCE_IDS. It is not context aware.
- By using (optionally multiple times) the `--query` (or `-q`) flag, users can
  restrict the resources returned by requiring that a field in the resource is
  equal to a provided value or that the content of a JSON document appear in the
  resource. This flag can be specified multiple times (even in conjunction with
  `-l` to further filter results).
- By using (optionally multiple times) the `--label` (or `-l`) flag, users can
  restrict the resources returned by means of the labels set in the resource's
  metadata. Labels must be specified in the `key=value` format. This flag can be
  specified multiple times (even in conjunction with `-q` to further filter
  results).
- `--with-property | -p` displays values for a subset of the constitutive
  properties. Cannot be used when the output format is `md`.
- `--format | -o` allows choosing the output format in which the information
  should be displayed. Can be one of either:
    - `md` - for Markdown text.
    - `table` (**default**) - for Markdown tables.
    - `csv` - for a comma separated file.

##### Examples

###### Get the summary of a space as a Markdown table

```shell
ado show summary space space-abc123-456def
```

<!-- markdownlint-disable line-length -->
###### Get the summary of a space as a Markdown table and include the constitutive property MY_PROPERTY
<!-- markdownlint-enable line-length -->

```shell
ado show summary space space-abc123-456def -p MY_PROPERTY
```

###### Get the summary of multiple spaces as a Markdown table via identifiers

```shell
ado show summary space space-abc123-456def space-ghi789-123jkl
```

###### Get the summary of multiple spaces as a Markdown table via key-value labels

```shell
ado show summary space -l issue=123
```

###### Get the summary of a space as a Markdown text

```shell
ado show summary space space-abc123-456def -o md
```

###### Get the summary of a multiple spaces as a CSV file via key-value labels

```shell
ado show summary space -l issue=123 -o csv
```

###### Get the summary of spaces that include granite-7b-base in the property domain

!!! info

    More information on field-level querying is provided in the
    [searching the metastore](../resources/metastore.md#searching-the-metastore)
     section

```shell
ado show summary space -q 'config.entitySpace={"propertyDomain":{"values":["granite-7b-base"]}}'
```

### ado template

To assist in creating a resource configuration file, we typically start from a
reference file. The `ado template` command allows you to create template files
that you can edit to streamline the process.

The complete syntax of the `ado template` command is as follows:

<!-- markdownlint-disable line-length -->

```shell
ado template RESOURCE_TYPE [--output | -o <PATH>] \
                           [--include-schema] \
                           [--operator-name <NAME>] \
                           [--operator-type <TYPE>] \
                           [--actuator-identifier <NAME>] \
                           [--from-experiment | -e <experiment_id | actuator_id:experiment_id>] \
                           [--local-context] \
                           [--no-parameters-only-schema]
```

<!-- markdownlint-enable line-length -->

Where:

- `RESOURCE_TYPE` is one of the supported resource types:

    - _actuator_
    - _actuatorconfiguration_ (_ac_)
    - _context_ (_ctx_)
    - _operation_ (_op_)
    - _discoveryspace_ (_space_)

- `--output` or `-o` can be used to point to a location where to save the
  template. By default, the template will be saved in the current directory with
  an autogenerated name.
- `--include-schema`, if set, will also produce the JSON Schema of the resource
  the template was generated for.
- `--operator-name` (**exclusive for operations**) allows generating an
  operation template with a specific operator. If unset, a generic operation
  will instead be output.
- `--operator-type` (**exclusive for operations**) is the type of operator to
  generate a template for. Must be one of the supported operator types:

    - `characterize`
    - `search`
    - `compare`
    - `modify`
    - `study`
    - `fuse`
    - `learn`

- `--actuator-configuration` (**exclusive for actuatorconfigurations**) is the
  identifier of the actuator to output. If unset, a generic actuator
  configuration will be output.
- `--from-experiment` (**exclusive for spaces**) can either be the identifier of
  the experiment you want to have in your space or, in case multiple actuators
  implementing the same experiment identifier, the identifier of the actuator
  and that of the experiment in the form `actuator_id:experiment_id`. If unset,
  a generic space will be output.
- `--local-context` (**exclusive for contexts**) creates a template using SQLite
  instead of MySQL.
- `--no-parameters-only-schema` (**exclusive for operations**) when used with
  `--include-schema`, outputs a generic operation schema. By default (when not
  specifying this flag), the schema will be operator-specific.

#### Examples

##### Creating a template for a context

```shell
ado template context
```

##### Creating a template for a space that uses a specific experiment

```shell
ado template space --from-experiment finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0
```

<!-- markdownlint-disable-next-line line-length -->
##### Creating a template for a space that uses a specific experiment from a specific actuator

```shell
ado template space --from-experiment SFTTrainer:finetune-gptq-lora-dp-r-4-a-16-tm-default-v1.1.0
```

##### Creating a template for a Discovery Space with the schema

```shell
ado template space --include-schema
```

##### Creating an operation template for the Rifferla operator

```shell
ado template operation --operator-name rifferla
```

### ado upgrade

!!! tip

    **`ado` will detect automatically when resource upgrades are required**
    and will print the exact command to run as a warning. In all other cases, there
    is no need to run this command.

Sometimes the models that are used in ado undergo changes that require updating
stored representations of them in the [metastore](../resources/metastore.md).
When required, you can run this command to update all resources of a given kind
in the database.

```shell
ado upgrade RESOURCE_TYPE
```

Where:

- `RESOURCE_TYPE` is one of the supported resource types:

    - _actuatorconfiguration_ (_ac_)
    - _datacontainer_ (_dcr_)
    - _operation_ (_op_)
    - _samplestore_ (_store_)
    - _discoveryspace_ (_space_)

#### Examples

##### Upgrade all operation resources

```shell
ado upgrade operations
```

### ado version

When unsure about what ado version you are running, you can get this information
with:

```shell
ado version
```

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Let's get started!**

    ---

    Jump into our examples

    [Our how-tos :octicons-arrow-right-24:](../examples/examples.md)

- :octicons-workflow-24:{ .lg .middle } **Learn more about the built-in Operators**

      ---

      Learn what `ado`'s built-in operators can offer you

      [Follow the guide :octicons-arrow-right-24:](../operators/working-with-operators.md)

</div>
<!-- markdownlint-enable line-length -->
