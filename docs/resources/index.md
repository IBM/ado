# Resources

<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->

With the **ado** CLI
you work with _resources_, _actuators and operators_, and
_contexts_.

**Actuators and operators**
provide capabilities, and **resources** define how you use them.
You store the resources in projects along with the measurement data.
To access a particular project you have a **context** which describes
its location and access credentials.

This section covers the different resource types, writing
resource definitions, and using the CLI to create resources from
the definitions and work with the results.

For more on actuators, operators and contexts see their dedicated sections

- [Working with Actuators](../user-guide/actuators)
- [Working with Operators](../user-guide/operators)
- [Projects & Contexts](metastore.md#contexts-and-projects)

## Resource Types

You use the `ado` CLI to create, list, inspect or delete resources.
To create a resources you first defined it in a YAML files. On
creation the resource definition is stored in a database (the
[metastore](metastore.md)).

The current `ado` resources are:

- **[samplestore](sample-stores.md)**: A database for storing entities and
  measurement results.
- **[discoveryspace](discovery-spaces.md)**: Describes a set of entities along
  with the experiment protocols that should be applied to them
- **[operation](operation.md)**: An instance of applying an operator to a
  `discoveryspace`. For example, running an optimization
- **[datacontainer](datacontainer.md)**: A collection of string, tabular or
  location data. Used to store arbitrary output from `operation`s.
- **[actuatorconfigurations](actuatorconfig.md)**: A configuration for an
  actuator.
- **[document](document.md)**: A markdown or HTML report or note stored in the
  metastore, optionally linked to related resources.

Creating some resource triggers other events to occur. In particular

- creating a `samplestore` resource results in specific
piece of storage being created.
- creating an operation resource causes an operation to execute.
The operation may sample and measure entities in a space or
could perform an analysis, potentially returning new resources

> [!NOTE]
>
> Some resources take other resources as input, for example `operations` take
> `discoveryspaces` as input.

### Naming Conventions: Concepts versus Resources

`ado` resources are directly related to `ado`
[concepts](../concepts/core-concepts.md) and usually have the same name. To
differentiate a concept and the associated resource in the documentation we
adopt the following conventions.

When we refer to concepts, upper case nouns like "Sample Store", "Actuator" are
used. However, for the corresponding resources lower case is used, with no
spaces, so `samplestore` and `actuator`.

## Common CLI commands for interacting with resources

Here is a list of common `ado` CLI commands for interacting with resources. See
the [ado CLI guide](../cli-reference/index.md) for more details

<!-- markdownlint-disable MD007 -->
- `ado get [resource type]`
    - Lists all resources of the requested type
- `ado get [resource type] [$identifier] -o yaml`
    - Outputs the YAML of resource `$identifier`
- `ado create [resource type] -f [YAMLFILE]`
    - Creates the resource of the specified type from the definition in "YAMLFILE"
- `ado delete [resource type] [$identifier]`
    - Deletes the resource of the specified type with the provided identifier from
      the database. See the [deleting resources](#deleting-resources) section for
      more information and considerations to keep in mind.
- `ado describe [resource type] [$identifier]`
    - Outputs a human-readable description of resource `$identifier`
- `ado show related [resource type] [$identifier]`
    - List ids of resources related to resource `$identifier`
- `ado template [resource type] --include-schema`
    - Outputs a default YAML for the given resource along with a schema file
      explaining the fields.`
<!-- markdownlint-enable MD007 -->

### Deleting resources

>[!TIP]
>
> Refer to the following documentation for detailed information on
> specific use cases:
>
> - [Deleting sample stores](sample-stores.md#deleting-sample-stores)
> - [Deleting operations](operation.md#deleting-operations)

In **ado** you can delete resources, but there is an important constraint: a
resource cannot be deleted if it has dependent (child) resources.

If you attempt to delete a resource that still has children, you will encounter
an error similar to the following:

```terminaloutput
ERROR:  Cannot delete discoveryspace space-3fbaad-c3a5f6 as it has children resources:

                                          IDENTIFIER       TYPE
0  raytune-1.0.2.dev11+1c62218-bayesopt-b7f779  operation

HINT:   You must delete each of them first.
```

To proceed, ensure that all child resources are deleted (using the `ado delete`
command on them) before attempting to remove the parent resource.

## Common features of resources

All resources have a YAML or JSON representation which is what is stored in the
metastore. The schema of this YAML has a common structure.

```yaml
config: ... # The configuration of the resource - different for each different type
created: "2024-10-03T12:42:35.786484Z" # Creation date
identifier: space-8f1cfb-91ecfb # Resource identifier
kind: discoveryspace # Resource kind
metadata: {} # Metadata dictionary
provenance: ... # Package provenance frozen at resource creation time
status: [] # A list of status objects describing notable events on the resource
version: v1 # The version of this resource
```

### Resource status

The status field of a resource contains an ordered sequence of status updates to
it. The most recent update is last. Each status update is associated with an
event that occurred to the resource, and this event is captured in the `event`
field. A status update will also have a timestamp, which is when the event was
recorded (usually right after it occurred). It can also contain additional event
dependent fields.

All resources have status updates recorded for the following events:

- **created**: When the resource is created
- **added**: When the resource is added to the metastore
- **updated**: Whenever the resource is updated in the metastore

Here is an example:

```yaml
config: ...
created: "2026-07-09T14:35:23.264745Z"
identifier: 88341a
kind: samplestore
metadata: {}
provenance:
  ado:
    distributionName: ado-core
    distributionVersion: 2.0.0
status:
  - event: created
    recorded_at: "2026-07-09T14:35:23.264754Z"
  - event: added
    recorded_at: "2026-07-09T14:35:23.265368Z"
  - event: updated
    recorded_at: "2026-07-24T13:56:36.053942Z"
version: v1
```

### Programmatic view of resources

Programmatically each resource type is represented by a Python pydantic model
class. All these classes inherit their basic structure from the root resource
class `ADOResource`.

You can load and validate any `ado` resource YAML with the following code
snippet: replace `discoveryspace` with the name of the resource as shown in
above list

```python
import yaml
from ado.core import kindmap

with open("resource.yaml") as f:
    resource = kindmap["discoveryspace"].model_validate(yaml.safe_load(f))
```

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

      ---

      Explore working with resources via our [examples](../user-guide/examples/index.md).

      [Our examples :octicons-arrow-right-24:](../user-guide/examples/index.md)

- :octicons-rocket-24:{ .lg .middle } **Learn about the ado CLI tool**

    ---

    Learn more about the [ado CLI tool](../cli-reference/index.md) for interacting with resources.

    [ado CLI :octicons-arrow-right-24:](../cli-reference/index.md)

</div>
<!-- markdownlint-enable line-length -->
