# The core ado concepts

`ado` is a tool for systematically exploring, measuring, and analysing a space
of entities - for example, configurations, systems and substances. It is built
on three core concepts: Discovery Space, Operations and Sample Store. In brief,
you define a Discovery Space, apply Operations to it, and store the results in a
Sample Store.

## Discovery Space

A **Discovery Space** defines how to answer the following questions: three
questions:

| Question                            | Concept                                             | Description                                                                                                                                |
| ----------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **How are measurements performed?** | [Measurement Space](actuators.md#measurement-space) | A Discovery Space defines a set of [Experiments](actuators.md) to use. Each Experiment takes defined inputs and produces measured outputs. |
| **What do you want to measure?**    | [Entity Space](entity-spaces.md)                    | A Discovery Space defines the specific set of things, called _Entities_, you want to measure with the Experiments.                         |
| **What have you measured so far?**  | [Sample Store](#sample-store)                       | A Discovery Space uses a shared database to read and store measurement results.                                                            |

For users familiar with `pandas`, a Discovery Space is like a DataFrame that
knows its own schema, knows how to fill in missing values, and shares data
transparently with other DataFrames. See [Discovery Spaces](discovery-spaces.md)
for more.

## Operations

To explore or analyse and Discovery Space you define an Operation. For example,
you might define to randomly sample and measure 40 points in the DiscoverySpace.

Defining an operation involves specifying the `operator` to use (the python
module the implements the operations) and the parameter values to set.

There are two broad classes of Operations: explore operations sample and measure
points from a Discovery Space; analysis operations process the data currently
collected in a Discovery Space to provide insights.

You can run multiple explore operations on the same space. Each one can select
and measure new points, increasing the total amount of information available on
entities in the space. You can always view the entities sampled by each
operation independently.

The fact that the Discovery Space exists independently from any particular
operations on it is a central innovation of `ado`.

## Sample Store

In `ado`, Entities and the results of Experiments on them are kept in a **Sample
Store** — a shared database that multiple Discovery Spaces can use.

If an Experiment has already been run on an Entity, `ado` can reuse the result
rather than running it again. This transparent data sharing is a core feature of
`ado`. See [Shared Sample Stores](data-sharing.md) for more details.

## What's next

<!-- prettier-ignore-start -->

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Learn about resources**

    ---

    Go to [resources](../resources/index.md) to learn more about working
    with these core concepts in `ado`.

    [ado resources :octicons-arrow-right-24:](../resources/index.md)

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

    ---

    Try some of our [examples](../user-guide/examples/index.md) if you want to
    dive straight in.

    [Our examples :octicons-arrow-right-24:](../user-guide/examples/index.md)

</div>
<!-- markdownlint-enable line-length -->

<!-- prettier-ignore-end -->
