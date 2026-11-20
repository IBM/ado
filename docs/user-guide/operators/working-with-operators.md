<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable first-line-h1 -->

An `operator` is a code module that provides a capability to perform an
`operation` on a `discoveryspace`. For example the `RandomWalk` operator
provides the capability to perform a random walk `operation` on a
`discoveryspace`.

The pages in this section give details about some of the operators available in
`ado`: what they are for, what they do and how to use them.

!!! info end

    The [examples](../examples/index.md) section contains worked
    examples of using some of these operators.

## `operator` types

Operators are grouped into the following types:

- **explore**: sample and measure entities from a `discoveryspace`
- **characterize**: analyse a `discoveryspace`
- **modify**: create a new `discoveryspace` by changing the entityspace or
  measurementspace of an input `discoveryspace`
- **compare**: compare one or more `discoveryspaces`
- **fuse**: create a new `discoveryspace` from a set of input `discoveryspaces`
- **summarize**: summarize a batch of resources, optionally versus a baseline

[This page](explore-operators.md) describes **explore** operators in more detail
as they are the only operators that sample and measure entities.

## Listing the available operators

The following CLI command will list the available `operators`

```commandline
ado get operators
```

This example shows the operators in one installation. Your list depends on the
plugins you have installed.

```commandline
Available operators by type:
┌───────┬─────────────────────────┬─────────┬──────────────┐
│ INDEX │ OPERATOR                │ VERSION │ TYPE         │
├───────┼─────────────────────────┼─────────┼──────────────┤
│ 0     │ detect_anomalous_series │ 1.0.5   │ characterize │
│ 1     │ flora                   │ 0.2.0   │ characterize │
│ 2     │ pkh                     │ 0.2.0   │ characterize │
│ 3     │ profile                 │ 2.0.4   │ characterize │
│ 4     │ trim                    │ 2.2.0   │ characterize │
│ 5     │ random_walk             │ 2.1.0   │ explore      │
│ 6     │ ray_tune                │ 2.1.0   │ explore      │
│ 7     │ rifferla                │ 2.0.6   │ modify       │
└───────┴─────────────────────────┴─────────┴──────────────┘
```

## Using operators

Using an operator involves the following steps:

1. Save a template for an installed operator, for example `random_walk`.

      ```shell
      ado template operation --operator-name random_walk --output-file operation.yaml
      ```

2. Edit `operation.yaml` to set the discovery space identifier and parameters.
3. Create the operation:

      ```shell
      ado create operation -f operation.yaml
      ```

4. Use the identifier printed by `ado create operation` to inspect that run.

      ```shell
      OPERATION_IDENTIFIER="<your operation identifier>"
      ado show related operation "$OPERATION_IDENTIFIER"
      # For explore operations
      ado show measurements operation "$OPERATION_IDENTIFIER"
      ```

These steps are covered in detail in [operations](../../resources/operation.md).

The keys under `inputs` in an operation YAML file are the operator function's
resource parameter names. A parameter `discoverySpace: DiscoverySpace` is set
as `inputs.discoverySpace`.

`ado create operation --with` and `--use-latest` can supply a discovery space,
data container, or actuator configuration. This can fill a single
unset input of that kind. When several inputs of the same kind are unset, set
them in the operation YAML.

If the operator has a list input of that kind, as `summarize` operators do,
the resource is appended to that list. A scalar input of the same kind, such as
a baseline, is not filled this way. Set it in the operation YAML.

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

      ---

      Explore using some of these operators with our [examples](../examples/index.md).

      [Our examples :octicons-arrow-right-24:](../examples/index.md)

- :octicons-rocket-24:{ .lg .middle } **Learn about Actuators**

    ---

    Learn about extending ado with new [Actuators](../actuators/working-with-actuators.md).

    [Creating new Actuators :octicons-arrow-right-24:](../actuators/working-with-actuators.md)

</div>
<!-- markdownlint-enable line-length -->
