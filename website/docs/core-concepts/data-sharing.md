# Shared Sample Stores

In `ado`, Entities and measurement results are stored in a database called a
**Sample Store**. For more on how Sample Stores are configured and managed see
[their dedicated page](../resources/sample-stores.md).

Two principles underpin data reuse in `ado`:

- **A Sample Store can be shared across multiple Discovery Spaces.** This allows
  any Discovery Space to access Entities and measurements recorded by operations
  on other Discovery Spaces that use the same store.
- **Each Entity has exactly one record in a Sample Store.** If two Discovery
  Spaces both include the same Entity, they reference the same record — there is
  no duplication.

> [!NOTE]
>
> To maximise the chance of data reuse, similar Discovery Spaces should use the
> same Sample Store. However, any Discovery Spaces can share a store regardless
> of how similar they are.

## How `ado` matches shared data

### Entities

In `ado` two Entities are the same if they have identical sets of
[constitutive property](properties-and-domains.md#property-types) values. Hence,
`ado` uses [constitutive property](properties-and-domains.md#property-types)
values to check if an Entity is present in the Sample Store, regardless of which
Discovery Space originally recorded them.

> [!IMPORTANT]
>
> The match does not use Entity identifiers as there are situations where they
> could be different even for Entities with same constitutive properties.

### Measurements

When an Entity is retrieved from the Sample Store, it carries the results of all
Experiments that have been applied to it. `ado` checks whether any of those
experiment identifiers match an Experiment in the current Measurement Space — if
so, the result can be reused.

## Data retrieval modes

When retrieving data from a Discovery Space (e.g. via `ado show measurements`),
there are two modes that control whether shared data is included:

<!-- markdownlint-disable line-length -->

| Mode         | What is returned                                                                                                                             |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **measured** | Only Entities and measurements recorded by operations run directly on _this_ Discovery Space. Compatible data from other spaces is excluded. |
| **matching** | All Entities and measurements in the Sample Store that are compatible with this Discovery Space, regardless of which space produced them.    |

<!-- markdownlint-enable line-length -->

Use **measured** when you want to see only the results your operations have
produced. Use **matching** when you want the full picture including any
compatible data from other spaces.

## Memoization

> [!IMPORTANT]
>
> Each explore operator provides a way to turn memoization on and off. See the
> [random walk](../operators/random-walk.md) and
> [ray tune](../operators/optimisation-with-ray-tune.md) operator documentation
> for examples of how this setting is controlled.

_Memoization_ is the name for data reuse that happens automatically during an
explore operation. When an operation samples an Entity it proceeds as follows:

- The Entity is sampled from the Entity Space
- The Entity's record is retrieved from the Sample Store if present (_matching
  on constitutive property values_).
- If **memoization is on**, `ado` checks for existing results for each
  Experiment in the Measurement Space using the Experiment’s unique identifier
  and reuses all matching results if any are found.
- If **memoization is off**, `ado` ignores existing results, reapplying each
  Experiment in the Measurement Space to the Entity. The new results are added
  to any that already exist.

See
[explore operators](../operators/explore_operators.md#memoization-replaying-measurements)
for how memoization fits into the explore execution loop, and
[Discovery Spaces](discovery-spaces.md#sampling-and-measurement) for how
operations and spaces relate.
