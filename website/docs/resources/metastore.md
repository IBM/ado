<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
`ado` uses a SQL database to store
[resource definitions](https://ibm.github.io/ado/resources/resources/#common-features-of-resources)
and [SQLSampleStores](sample-stores.md#sqlsamplestore). When you execute `ado`
commands like `get` or `describe` they are interacting with this metastore.

By hosting a metastore on a dedicated server `ado` can be used by multiple
distributed users.

!!! info end

    The `ado` CLI can create local metastore instances. Shared metastores require
    [separately provisioning the database server](/ado/getting-started/installing-backend-services/#using-the-distributed-mysql-backend-for-ado).

## Contexts and Projects

An instance of the metastore can host one or more `projects`. To access a
`project` you create a `context` which contains location information, and
optionally access credentials, for the `project`.

### Contexts for local projects

Local projects are stored in local metastores. Local metastores use SQLite. A
local metastore can hold a single project. Hence, there is one database per
local metastore instance that contains the resources associated with this
project.

A context for a local metastore looks like:

```yaml
project: local-test
metadataStore:
  path: $HOME/Library/Application Support/ado/databases/local-test.db
  sslVerify: false
```

### Contexts for remote projects

Remote projects are stored in remote metastores. Remote metastores use MySQL. A
remote metastore can host multiple projects. Each project is associated with an
access-controlled database that contains the project's resources.

!!! info end

    Everyone with access to the same remote project can see and interact with all
    the resources in it

A context for a remote metastore looks like:

```yaml
project: ft-prod
metadataStore:
  host: 192.168.0.1
  password: XXXXXXXXXXX
  port: 32001
  sslVerify: false
```

## Working with Contexts

### Creating a context

To create a local or remote context in `ado`, create a file with the
corresponding YAML definition (see above) and run:

```commandline
ado create context -f $YAML_FILE
```

If the context refers to a local project (a local context), a SQLite database is
created for the project if it doesn't exist. If the context refers to a remote
project (a remote context), the MySQL database for the project must have been
created separately.

### Listing available contexts

To see a list of contexts do

```commandline
ado get contexts
```

This will output something like

```commandline
                  CONTEXT DEFAULT
0              finetuning
1              ap-testing
2       developer-testing
3             mascots2024
4      caikit-testharness
5    materials-evaluation
6                 ft-prod       *
7            unit-testing
8       your-project-name
9  resource-store-testing
```

Note, the name of the context is the name of the associated project.

### The active context

To use a context you activate it with:

```commandline
ado context $CONTEXTNAME
```

and it becomes the "Active Context". All `ado` commands that interact with the
metastore, like `get`, `show`, will be directed to the project associated with
the active context.

Example:

```commandline
$ ado context materials-evaluation
Success! Now using context materials-evaluation
```

To remind yourself what the active context is run

```commandline
ado context
```

The active context is also denoted by a "\*" in the output of `ado get contexts`
(see output above).

> [!NOTE]
>
> Although `context` appears _like_ a resource in `ado` e.g. you can `get`
> contexts, the definition is not stored in the metastore, so it is purely
> local.

### Deleting contexts

You can delete a context using

```commandline
ado delete context $CONTEXT_NAME
```

For remote contexts the delete operation only deletes the context YAML. The
underlying MySQL database remains and must be deleted separately.

For local contexts, the delete operation prompts if you want to delete the
underlying SQLite database, and thus the project. If you opt to delete the
project, the data cannot be retrieved. In this case, if you recreate the context
a new local database will be created.

If you just delete the context, the underlying SQLite database, and hence the
project data, remains. In this case, if you recreate the context it will use the
existing database.

## Searching the Metastore

The [`ado get`](../getting-started/ado.md#ado-get) CLI command lets you easily
retrieve and search
[resource definitions](https://ibm.github.io/ado/resources/resources/#common-features-of-resources)
in the metastore in a variety of ways.

### Searching for similar spaces

A common use case is searching for spaces that are "similar" to a reference
space. A space is considered similar only if **both** of the following hold:

- They include **exactly the same base experiments** as the reference space
- Their **entity space** is in a **hierarchical relationship** with the
  reference space: subspace, equal or superspace

This search can be performed in two ways:

- Using as reference an existing discovery space identifier via the flag
  `--matching-space-id`
- Providing a
  [DiscoverySpace configuration YAML](discovery-spaces.md#discovery-space-configuration-yaml)
  to the flag `--matching-space`. This is useful to find similar spaces without
  actually creating one first.

The output of this command will include the hierarchical relationship between
the spaces, meaning that a column will say whether the matching space is a
subspace, a superspace, or an exact match.

### Searching for spaces containing a point

If you're looking for discovery spaces that **contain** a specific
[entity](../core-concepts/entity-spaces.md#entities) and (optionally) a list of
experiments, you can use the `--matching-point` option.

This option accepts a YAML file with the following structure:

> [!IMPORTANT]
>
> The match condition is not **equals** but **contains**. That is, any entity
> that **at least** has the provided properties will match.

```yaml
entity: # A key-value dictionary of constitutive property identifiers and values
  batch_size: 8
  number_gpus: 4
experiments: # (OPTIONAL) A list of experiments
  - finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0
```

### Searching for resources with a given label

If you're using **labels** to tag your resources, you can quickly retrieve
matching resources using the `--label` option (or its shorthand `-l`), providing
each label in the format: `key=value`.

You can specify this option **multiple times** to filter resources that match
**all** the given labels. For example:

```commandline
ado get operations -l labelone=valueone -l label_two=value_two
```

will retrieve all operations that have the label `labelone` with the value
`valueone` and `label_two` with the value `value_two`.

### Searching against resource fields

The `-q` (`--query`) option to `ado get` allows searching for resources where a
particular field **contains** a particular value. The `-q` option can be
specified multiple times allowing search conditions on multiple fields. It can
also be specified with `-l`.

The syntax is:

```commandline
-q $KEY_PATH=value
```

In the simplest case `$KEY_PATH` is a `.`-separated sequence of fieldnames that
lead to a target field. For example, in this case:

```yaml
config:
  metadata:
    name: "my_name"
    description: "some description"
    tags:
      - "test"
      - "fine_tuning"
```

`config.metadata.name` is the key path to the `name` field.

> [!IMPORTANT]
>
> The key path and values are case-sensitive.

#### Searching for fields with scalar values

For fields whose values are scalars (integers, floats, and strings), to match
the value at `$KEY_PATH` must be equal to the given value. In the example above
the value of the `name` field is a string, so the query

```commandline
-q config.metadata.name=my_name
```

asks whether the value of `config.metadata.name` is equal to `my_name`.

<!-- markdownlint-disable no-blanks-blockquote -->

> [!NOTE]
>
> Be careful when searching for fields whose values are strings which are
> numbers. If the field is an integer it will not be matched by the string
> equivalent and vice versa. For example, "947" will not match 947.

> [!NOTE]
>
> You do not need to quote non-numeric strings to search them c.f. my_name
> above.

<!-- markdownlint-enable no-blanks-blockquote -->

#### Searching for fields with dictionary values

For fields whose values are dictionaries, the search value must also be a JSON
dictionary. The query asks if the dictionary at `$KEY_PATH` contains the set of
key:values in the given JSON dictionary.

```commandline
-q 'config.metadata={"name":"my_name"}'
```

asks whether the dictionary at `config.metadata` contains the key `name` with
value `my_name`.

<!-- markdownlint-disable no-blanks-blockquote -->
> [!IMPORTANT]
>
> The matching criteria is not equality but _**contains**_.

> [!NOTE]
>
> The dictionary keys (strings) must be quoted and string values **must** be
> quoted. This is different from when a string is used on its own as a value.

<!-- markdownlint-enable no-blanks-blockquote -->

#### Searching in arrays

For fields whose values are arrays, the equality operation asks if the array at
`$KEY_PATH` contains the value. In this case `value` can be any valid JSON
object. For example:

```commandline
-q config.metadata.tags=fine_tuning
```

checks if the string "fine_tuning" is in the list (TBD: Should the string be
quoted? c.f. dictionary or scalar?)

<!-- markdownlint-disable no-blanks-blockquote -->
> [!NOTE]
>
> The matching criteria is not equality but _**contains**_.

> [!NOTE]
>
> If the value being searched for is a non-scalar JSON object, strings **must**
> be properly quoted.

<!-- markdownlint-enable no-blanks-blockquote -->

#### More complex key paths

The JSON Path follows
[MYSQL JSON Path syntax](https://www.mysqltutorial.org/mysql-json/mysql-json-path/),
with some important differences.

First, the root element `$.` is added automatically as we've found this is
intuitively how users expect the statement `X=Y` to work. That is, the key path
`metadata.name` is translated to `$.metadata.name`.

**TBD: The above page states that indexing arrays [N] indexes element [N-1] but
this does not seem to be the case.**

> [!IMPORTANT]
>
> Finally, the select-all operator, `*`, is not supported. You can often
> leverage the _contains_ matching to replicate the same behaviour.

### Examples

If you want to query operations that use the RayTune operator you can do it
with:

```commandline
ado get operations -q config.operation.module.moduleClass=RayTune
```

To query all spaces that contain the
`finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0` experiment:

<!-- markdownlint-disable line-length -->
```commandline
ado get spaces -q 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}'
```
<!-- markdownlint-enable line-length -->

To also include those using `NVIDIA-A100-SXM4-80GB` for `gpu_model` and
`mistral-7b-v0.1` for `model_name`:

<!-- markdownlint-disable line-length -->
```commandline
ado get spaces -q 'config.entitySpace={"identifier": "model_name", "propertyDomain":{"values":["mistral-7b-v0.1"]}}' \
              -q 'config.entitySpace={"identifier": "gpu_model", "propertyDomain":{"values":["NVIDIA-A100-SXM4-80GB"]}}' \
              -q 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}'
```
<!-- markdownlint-enable line-length -->

Note, if you know a value is only used in a particular domain you can leave out
`identifier` above.

<!-- markdownlint-disable line-length -->

```commandline
ado get spaces -q 'config.entitySpace={"propertyDomain":{"values":["mistral-7b-v0.1"]}}' \
              -q 'config.entitySpace={"propertyDomain":{"values":["NVIDIA-A100-SXM4-80GB"]}}' \
              -q 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}'
```
<!-- markdownlint-enable line-length -->
