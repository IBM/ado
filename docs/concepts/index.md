# Concepts

This section explains the core concepts behind `ado`. Reading these pages will
give you a mental model of how `ado` structures, performs, and shares
measurements before you start configuring resources or running experiments.

The concepts build on each other in the order they appear below.

## [Properties and Domains](properties-and-domains.md)

Everything in `ado` is described through **Properties** — named identifiers
such as `batch-size` or `gpu-model`. A **Property Domain** constrains the
values a property can take (categorical, discrete, continuous, binary, or open
categorical) and controls how values are sampled.

Properties play three roles: *constitutive* (inputs that identify an entity),
*target* (what an experiment intends to measure), and *observed* (the
namespaced value actually recorded by a specific experiment).

## [Experiments and Actuators](actuators.md)

An **Experiment** measures the values of a set of output properties given a
set of input properties. Experiments declare required and optional inputs
(each with a Property Domain) and the target properties they produce as output.

**Actuators** are plugins that group and provide Experiments for a particular
domain — for example, foundation model fine-tuning or robotic biology. A
**Measurement Space** is the collection of Experiments used in a Discovery
Space.

## [Entities and Entity Spaces](entity-spaces.md)

An **Entity** is the thing you want to measure. It is fully described by a set
of constitutive property values — for example, a specific combination of
`gpu-model`, `batch-size`, and `model-name`.

An **Entity Space** defines the full set of Entities you want to explore: a set
of constitutive properties, each with a Property Domain. The space is the
cartesian product of those domains — every valid combination is a potential
Entity to measure.

## [Discovery Spaces](discovery-spaces.md)

A **Discovery Space** combines an Entity Space and a Measurement Space. It
answers three questions: *what* to measure (Entity Space), *how* to measure it
(Measurement Space), and *where* results are stored (Sample Store).

A Discovery Space is a *view*, not a container — data is fetched from a shared
Sample Store on demand. An explore operation selects Entities from the space,
applies the Experiments, and stores the results.

## [Shared Sample Stores](data-sharing.md)

Entities and measurement results are stored in a **Sample Store** — a shared
database that multiple Discovery Spaces can use. If an Entity has already been
measured (even by a different Discovery Space using the same store), `ado` can
reuse the result rather than re-running the Experiment. This transparent
**memoization** is a core feature of `ado`.

---

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
