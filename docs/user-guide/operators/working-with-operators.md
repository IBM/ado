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

[This page](explore-operators.md) describes **explore** operators in more detail
as they are the only operators that sample and measure entities.

## Listing the available operators

The following CLI command will list the available `operators`

```commandline
ado get operators
```

Example output:

```commandline
┌───────┬─────────────────────────┬─────────┬──────────────┐
│ INDEX │ OPERATOR                │ VERSION │ TYPE         │
├───────┼─────────────────────────┼─────────┼──────────────┤
│ 0     │ detect_anomalous_series │ 1.0.5   │ characterize │
│ 1     │ profile                 │ 2.0.4   │ characterize │
│ 2     │ trim                    │ 2.0.3   │ characterize │
│ 3     │ random_walk             │ 2.0.0   │ explore      │
│ 4     │ ray_tune                │ 2.0.6   │ explore      │
│ 5     │ rifferla                │ 2.0.6   │ modify       │
└───────┴─────────────────────────┴─────────┴──────────────┘
```

## Using operators

Using an operator involves the following steps:

1. Generate an operation template for the operator with default parameter values:

      ```shell
      ado template operation --operator-name $OPERATOR_NAME
      ```

2. Edit the generated YAML to configure the parameters to your liking.
3. Create the operation:

      ```shell
      ado create operation -f $OPERATION_FILE
      ```

4. Retrieve the results of the operation:

      ```shell
      ado show related operation --use-latest
      # For explore operations
      ado show measurements operation --use-latest
      ```

These steps are covered in detail in [operations](../../resources/operation.md).

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
