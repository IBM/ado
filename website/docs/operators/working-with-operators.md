<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
An `operator` is a code module that provides a capability to perform an
`operation` on a `discoveryspace`. For example the `RandomWalk` operator
provides the capability to perform a random walk `operation` on a
`discoveryspace`.

The pages in this section give details about some of
the operators available in `ado`: what they are for, what they do and how to use
them.

!!! info end

    The [examples](../examples/examples.md) section contains worked
    examples of using some of these operators.

## `operator` types

Operators are grouped into the following types:

- **explore**: sample and measure entities from a `discoveryspace`
- **characterize**: analyse a `discoveryspace`
- **modify**: create a new `discoveryspace` by changing the entityspace or
  measurementspace of an input `discoveryspace`
- **compare**: compare one or more `discoveryspaces`
- **fuse**: create a new `discoveryspace` from a set of input `discoveryspaces`

[This page](explore_operators.md) describes **explore** operators in more detail
as they are the only operators that sample and measure entities.

## Listing the available operators

The following CLI command will list the available `operators`

```commandline
ado get operators
```

Example output:

```commandline
                                       OPERATOR          TYPE
1                       detect_anomalous_series  characterize
0                                       profile  characterize
2                                   random_walk       explore
3                                      ray_tune       explore
9                       export_to_llm_lakehouse        export
8  integrate_and_export_to_llm_lakehouse_format        export
4                                add_experiment        modify
7              generate_representative_subspace        modify
5                                learning_split        modify
6                                      rifferla        modify
```

## Using operators

Using an operator involves the following steps:

1. Find the input parameters for the operator:
   - `ado template operation --operator-name $OPERATOR_NAME`
2. Write an operation YAML for applying the operator.
   - This involves setting specific values for its input parameters.
3. Create the operation:
   - `ado create operation -f $YAML`
4. Retrieve the results of the operation:
   - `ado show related $OPERATION_IDENTIFIER`
   - in addition `ado show entities $OPERATION_IDENTIFIER` for explore operations

These steps are covered in detail in [operations](../resources/operation.md).

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

      ---

      Explore using some of these operators with our [examples](../examples/examples.md).

      [Our examples :octicons-arrow-right-24:](../examples/examples.md)

- :octicons-rocket-24:{ .lg .middle } **Learn about Actuators**

    ---

    Learn about extending ado with new [Actuators](../actuators/working-with-actuators.md).

    [Creating new Actuators :octicons-arrow-right-24:](../actuators/working-with-actuators.md)

</div>
<!-- markdownlint-enable line-length -->