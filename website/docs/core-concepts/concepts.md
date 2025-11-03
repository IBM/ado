<!-- markdownlint-disable-next-line first-line-h1 -->
## Discovery Space

The core concept in `ado` is called a _Discovery Space_. In `ado` you are often
creating and performing operations on Discovery Spaces.

For users familiar with `pandas` and `dataframes`, a Discovery Space combines:

- the schema of a `dataframe` i.e. the columns and what they mean
- instructions on how to fill the `dataframe` rows
- the current data in the `dataframe` (and what's missing!)

A Discovery Space expresses the hidden metadata and contextual
information necessary to understand and extend a dataframe. See
[Discovery Space](discovery-spaces.md) for more details.

A Discovery Space is built from:

- [Entities and Entity Spaces](entity-spaces.md): The set of things in a
  Discovery Space
- [Measurement Spaces](actuators.md#measurement-space): The set of experiments
  in a Discovery Space
- [Experiments and Actuators](actuators.md): The available experiments and the
  tools that execute them

## Sample Store

In `ado`, data on sampled entities, and the results of experiments on them, are
kept in a **sample store**.

A single sample store can be used by multiple Discovery Spaces, allowing them to
share data. This means, for example, if an experiment has already been run,
`ado` can reuse the existing results instead of running the experiment again,
saving time and computational resources.

This ability to transparently share and reuse data is a core feature of `ado`.
See [Shared Sample Stores](data-sharing.md) for more details.

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Learn about resources**

    ---

    Next go to [resources](../resources/resources.md) to learn more about working with these core-concepts in `ado`.

    [ado resources :octicons-arrow-right-24:](../resources/resources.md)

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

      ---

      Try some of our [examples](../examples/examples.md) if you want to dive straight in.

      [Our examples :octicons-arrow-right-24:](../examples/examples.md)

</div>
<!-- markdownlint-enable line-length -->