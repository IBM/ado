<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
`ado` uses a SQL database to store
[resource definitions](https://ibm.github.io/ado/resources/resources/#common-features-of-resources)
and [SQLSampleStores](sample-stores.md#sqlsamplestore). When you execute `ado`
commands like `get` or `describe` they are interacting with this metastore.

By hosting a metastore on a dedicated server, `ado` can be used by multiple
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

### Searching inside resource representations

For more advanced searches, `ado` provides the `--query` option to find
resources based on the contents of their
[stored representation](resources.md/#common-features-of-resources). This option
can be specified multiple times and in conjunction with the `-l` option to find
resources that match all the specified filters.

The syntax is:

```commandline
-q 'path=candidate'
```

!!! warning inline end

    We suggest using single quotation marks around the candidate passed to the -q
    option. **This is required when dealing with strings, dictionaries, and
    arrays**.

Where:

- **Path** is a dot-separated string that defines how to locate a specific field
  within a resource (represented by a dictionary or a JSON object). Each segment
  in the path represents a key at a particular level of the structure. By
  following the path from left to right, you traverse through the nested fields
  until you reach the target field. An example is `config.metadata.labels`.
- **Candidate** is a valid JSON value (dictionary, array, string, number,
  boolean).

#### Matching rules

??? tip end "The examples below will reference this example YAML"

    ```yaml
    config:
      metadata:
        description: Perform a random walk on all points in a space
        name: randomwalk-all
      operation:
        parameters:
          batchSize: 1
          numberEntities: 48
          samplerConfig:
            mode: random
            samplerType: generator
          singleMeasurement: true
      spaces:
      - space-3904a4-96dd91
    created: '2025-09-24T15:11:36.334913Z'
    identifier: randomwalk-1.0.2.dev30+a96d4d1.dirty-b54866
    metadata:
      entities_submitted: 48
      experiments_requested: 74
    operationType: search
    operatorIdentifier: randomwalk-1.0.2.dev30+a96d4d1.dirty
    status:
    - event: created
      recorded_at: '2025-09-24T15:10:48.301104Z'
    - event: added
      recorded_at: '2025-09-24T15:11:39.692664Z'
    - event: started
      recorded_at: '2025-09-24T15:11:43.266627Z'
    - event: updated
      recorded_at: '2025-09-24T15:11:43.266640Z'
    - event: finished
      exit_state: success
      recorded_at: '2025-09-24T15:16:54.422497Z'
    - event: updated
      recorded_at: '2025-09-24T15:16:54.429667Z'
    ```

Resources are returned if they **_contain_ the candidate value at the specified
path**. The matching behavior depends on the type of the candidate value.
Examples for different value types are shown below.

##### The path points to a scalar

!!! warning inline end

    `ado` converts boolean
    [property values](/ado/resources/discovery-spaces/#defining-the-domains-of-constitutive-properties-in-the-entityspace)
    to integers. For more details on how this works in practice, refer to the
    [additional examples](#additional-examples).

If the candidate is a scalar (a string in double quotes, a number, `true`,
`false`, `null`), the value at the specified path will match if it is of the
**same type** (treating integers and floats as equivalent) and has the **same
value**.

- ✅ `ado get operations -q 'config.operation.parameters.batchSize=1'` (1 == 1)
- ✅ `ado get operations -q 'config.operation.parameters.batchSize=1.0'` (1.0 ==
  1, floats and ints count as same type)
- ❌ `ado get operations -q 'config.operation.parameters.batchSize="1"'` ("1" !=
  1, type mismatch)
- ✅ `ado get operation -q 'config.operation.parameters.singleMeasurement=true'`
- ❌ `ado get operation -q 'config.operation.parameters.singleMeasurement=True'`
  (`true` and `false` are the only boolean values in JSON)

##### The path points to an array

**If the candidate is an array**, the array at the specified path will match if
it **contains all elements of the candidate**.

- ✅
  `ado get operation -q 'status=[{"event": "finished", "exit_state": "success"}]'`
  (all keys are contained)
- ❌
  `ado get operation -q 'status=[{"exit_state": "success"}, {"exit_state": "failure"}]'`
  (no operation can have both a `success` and a `failure` exit state)

**If the candidate is not an array**, the array at the specified path will match
if it **contains the candidate**.

- ✅ `ado get operation -q 'status={"exit_state": "success"}'` (status has an
  element that contains the key `exit_state` and value `success`)
- ✅ `ado get operation -q 'config.spaces=space-3904a4-96dd91'` (the string is
  part of the array)
- ❌ `ado get operation -q 'status={"event": "fake"}'` (no element of the
  `status` array has the value `fake` for the `event` key)

##### The path points to a dictionary (JSON object)

**If the candidate is a dictionary**, the dictionary at the specified path will
match if it contains all keys and corresponding values from the candidate.

<!-- markdownlint-disable line-length -->
- ✅
  `ado get operation -q 'config.operation.parameters={"batchSize": 1, "samplerConfig": {"mode": "random"}}'`
- ❌
  `ado get operation -q 'config.operation.parameters={"batchSize": 1, "samplerConfig": {"mode": "smart"}}'`
  (the operation's `samplerConfig.mode` is `random`)
<!-- markdownlint-enable line-length -->

!!! tip

    The field-searching behavior described above is powered by MySQL's
    `JSON_CONTAINS` function. For a more formal and complete definition of
    containment logic, read
    [the official MySQL documentation](https://dev.mysql.com/doc/refman/8.4/en/json-search-functions.html#function_json-contains).

#### Additional examples

##### Finding operations using a certain operator

If you want to query operations that use the RayTune operator you can do it
with:

```commandline
ado get operations -q config.operation.module.moduleClass=RayTune
```

##### Finding spaces with a boolean parameterization

To query all spaces are parameterized with the `bf16` property with the boolean
value `true`, you will have to query using the value `1` instead. This is
because `ado` applies a type conversion to boolean values in
[properties](/ado/resources/discovery-spaces/#defining-the-domains-of-constitutive-properties-in-the-entityspace):

<!-- markdownlint-disable line-length -->
```commandline
ado get spaces -q 'config.experiments={"experiments":{"parameterization":[{"property":{"identifier":"bf16"},"value":1}]}}
```
<!-- markdownlint-enable line-length -->

##### Finding spaces with a specific experiment

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
