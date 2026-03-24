<!-- markdownlint-disable-next-line first-line-h1 -->
## Discovery Space

`ado` is a tool for systematically exploring, measuring, and analysing a space of
entities - for example, configurations, systems and substances.
The core concept enabling this is a
**Discovery Space**. It answers three questions:

- **How are measurements performed?** A Discovery Space defines
  a set of [Experiments](actuators.md). Each Experiment
  takes defined inputs and produces measured outputs. The collection of Experiments
  is called a [Measurement Space](actuators.md#measurement-space).
- **What do you want to measure?** A Discovery Space defines an
  [Entity Space](entity-spaces.md) — the
  specific set of things, called _Entities_, you want to measure.
- **What have you measured so far?** A Discovery Space uses
  a **Sample Store**, a shared database, to read and store measurement
  results.

For users familiar with `pandas`, a Discovery Space is like a DataFrame that
knows its own schema, knows how to fill in missing values, and shares data
transparently with other DataFrames. See [Discovery Spaces](discovery-spaces.md)
for more.

## Sample Store

In `ado`, Entities and the results of Experiments on them are kept in a
**Sample Store** — a shared database that multiple Discovery Spaces can use.

If an Experiment has already been run on an Entity, `ado` can reuse the result
rather than running it again. This transparent data sharing is a core feature of
`ado`. See [Shared Sample Stores](data-sharing.md) for more details.

## What's next

<!-- markdownlint-disable line-length -->
<!-- markdownlint-disable-next-line no-inline-html -->
<div class="grid cards" markdown>

- :octicons-rocket-24:{ .lg .middle } **Learn about resources**

    ---

    Go to [resources](../resources/resources.md) to learn more about working
    with these core concepts in `ado`.

    [ado resources :octicons-arrow-right-24:](../resources/resources.md)

- :octicons-workflow-24:{ .lg .middle } **Try our examples**

    ---

    Try some of our [examples](../examples/examples.md) if you want to dive
    straight in.

    [Our examples :octicons-arrow-right-24:](../examples/examples.md)

</div>
<!-- markdownlint-enable line-length -->