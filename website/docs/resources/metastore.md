<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
`ado` uses a SQL database to store
[resource definitions](https://ibm.github.io/ado/resources/resources/#common-features-of-resources)
and [SQLSampleStores](sample-stores.md#sqlsamplestore).
When you execute `ado`
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

Remote projects are stored in remote metastores. Remote metastore uses MySQL. A
remote metastore can host multiple projects. Each project is associated with an
access-controlled databases that contains the projects resources.

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

If the context refers to a local project (a local context), a SQLite database
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
> Although `context` appears _like_ resource in `ado` e.g. you can `get`
> contexts, the definition is not stored in the metastore, so it is purely
> local.

### Deleting contexts

You can delete a context using

```commandline
ado delete context $CONTEXT_NAME
```

For remote contexts the delete operation only deletes the context yaml. The
underlying MySQL database remains and must be deleted separately.

For local contexts, the delete operation prompts if you want to delete the
underlying SQLite database, and thus the project. If you opt to delete the
project, the data cannot be retrieved. In this case, if you recreate the context
a new local database will be created.

If you just delete the context, the underlying SQLite database, and hence the
project data, remains. In this case, if you recreate the context it will use the
existing database.

## Searching the Metastore

The [`ado get`](../getting-started/ado.md#ado-get) CLI command allows fetching [resource
definitions](https://ibm.github.io/ado/resources/resources/#common-features-of-resources)
from the metastore.
In addition, `ado get` provides the ability to search the data in the metastore
in various ways.

### Searching for spaces like X

Use the `--matching-space-id` option to `ado get` to finding `discoveryspace`s similar
to another `discoveryspace` (the input space). Spaces will match if:

- They include **exactly** the same **base experiments** as the input space
- Their entity space is in a hierarchical relationship with the input space :
subspace, equal or superspace

The hierarchical relationship will be output i.e., a column will say whether the
matching space is a subspace, a superspace, or an exact match with the input
space.

The `--matching-space` option to `ado get` works in the same way, but
allows the user to provide a
[discoveryspace configuration YAML](discovery-spaces.md#discovery-space-configuration-yaml)
to search against.
This allows searching against spaces without the need to create them.

### Searching for spaces containing a point

 The `--matching-point` option allows finding spaces which contain a particular
 [entity](../core-concepts/entity-spaces.md#entities) and experiment combination.
For example:

  ```yaml
  entity: # A key-value dictionary of constitutive property identifiers and values
    batch_size: 8
    number_gpus: 4
  experiments: # A list of experiments
    - finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0
  ```

>[!IMPORTANT]
> The match condition is not **equals** but **contains**.
> That is, any entity with the listed properties will match - it may have others
> also.

### Searching for resources with a given label

The `-l` (`--label`) option to `ado get` allows searching for resources
with a particular value for a label.
The `-l` option can be specified multiple times allowing search
conditions on multiple fields.

```commandline
ado get spaces -l mylabel=somevalue
```

### Searching against resource fields

The `-q` (`--query`) option to `ado get` allows searching for resources
where a particular field **contains** a particular value.
The `-q` option can be specified multiple times allowing search conditions
on multiple fields. It can also be specified with `-l`.

The syntax is:

```commandline
-q $KEY_PATH=value
```

In the simplest case `$KEY_PATH` is a `.` separated sequence of fieldnames that
lead to a target field. For example, in this case

```yaml
config:
  metadata:
    name: 'my_name'
    description: 'some description'
    tags:
      - 'test'
      - 'fine_tuning'
```

`config.metadata.name` is the key path to the `name` field.

>[!IMPORTANT]
> The key path and values are case-sensitive.

#### Searching for fields with scalar values

For fields whose values are scalars i.e., integers, floats and strings, to match
the value at `$KEY_PATH` must be equal to given value.
In the above example the value of the `name` field is a string (scalar) so the query

```commandline
-q config.metadata.name=my_name
```

asks if the value of `config.metadata.name` is equal to `my_name`.

<!-- markdownlint-disable MD028 -->
>[!NOTE]
> Be careful when searching for fields whose values are strings which are numbers.
> If the field is an integer it will not be matched by the equivalent and vice versa.
> For example, "947" will not match 947

>[!NOTE]
> You do not need to quote non-numeric strings to search them c.f. my_name above
<!-- markdownlint-enable MD028 -->

#### Searching for fields with dictionary values

For fields whose values are dictionaries, the value search for
should also be a JSON dictionary.
The query asks if the (dictionary) value at `$KEY_PATH` contains
the set of key:values in the given JSON dictionary.
For example, the query:

```commandline
-q 'config.metadata={"name":"my_name"}'
```

asks if the dictionary at `config.metadata` contains the key "name" with value "my_name".

<!-- markdownlint-disable MD028 -->
> [!IMPORTANT]
> The matching criteria is not equality but _**contains**_.

> [!NOTE]
> The dictionary keys (strings) must be quoted and string values **must** be quoted.
> This is different
> [to when a string is used on its own as a value.](#searching-for-fields-with-scalar-values)
<!-- markdownlint-enable MD028 -->

### Searching in arrays

For fields whose values are arrays,
the equality operation asks if the array at `$KEY_PATH` _contains_ the value.
In this case `value` can be any valid JSON object.
For example:

```commandline
-q config.metadata.tags=fine_tuning
```

checks if the string "fine_tuning" is in the list (**TBD: Does this string
need to be quoted c.f. dictionary or not c.f. scalar?**)

> [!NOTE]
> The matching criteria is not equality but _**contains**_.
> [!NOTE]
> If the value being searched for is a non-scalar JSON object,
> strings **must** be properly quoted

#### More complex key paths

The JSON Path follows [MYSQL JSON Path syntax](https://www.mysqltutorial.org/mysql-json/mysql-json-path/),
with some important differences.

First, the root element `$.` is added automatically as we've found this is
intuitively how users expect the statement `X=Y` to work.
That is, the key path `metadata.name` is translated to `$.metadata.name`.

**TBD: The above page states that indexing arrays [N] indexes element [N-1]
but this does not seem to be the case.**

>[!IMPORTANT]
> Finally, the select all operator, `*`, is not supported.
> You can often leverage the _contains_ matching to replicate the same behaviour.

### Examples

If you want to query operations that use the RayTune operator you can do it with:

```commandline
ado get operations -q config.operation.module.moduleClass=RayTune
```

To query all spaces that contain the
`finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0` experiment:

<!-- markdownlint-disable line-length -->
```commandline
ado get space -q 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}' 
```
<!-- markdownlint-enable line-length -->

To also include those use `NVIDIA-A100-SXM4-80GB` for `gpu_model`
and `mistral-7b-v0.1` for `model_name`:

<!-- markdownlint-disable line-length -->
```commandline
ado get space -q 'config.entitySpace={"identifier": "model_name", "propertyDomain":{"values":["mistral-7b-v0.1"]}}' \
              -q 'config.entitySpace={"identifier": "gpu_model", "propertyDomain":{"values":["NVIDIA-A100-SXM4-80GB"]}}' \
              -q 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}'
```
<!-- markdownlint-enable line-length -->

Note, if you know a value  is only used in a particular domain you can leave
out `identifier` above

<!-- markdownlint-disable line-length -->
```commandline
ado get space -q 'config.entitySpace={"propertyDomain":{"values":["mistral-7b-v0.1"]}}' \
              -q 'config.entitySpace={"propertyDomain":{"values":["NVIDIA-A100-SXM4-80GB"]}}' \
              -q 'config.experiments={"experiments":{"identifier":"finetune-lora-fsdp-r-4-a-16-tm-default-v2.0.0"}}'
```
<!-- markdownlint-enable line-length -->