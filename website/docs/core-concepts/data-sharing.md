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

Each Entity has a unique identifier derived from its
[constitutive property](properties-and-domains.md#property-types) values.
For example, an Entity with properties `X=4` and `Y=10` gets the id
`X.4-Y.10`. `ado` uses these identifiers to look up Entities in the Sample
Store, regardless of which Discovery Space originally recorded them.

### Measurements

Each Experiment also has a unique identifier (its name plus any explicitly set
optional inputs). When an Entity is retrieved from the Sample Store, it carries
the results of all Experiments that have been applied to it. `ado` checks
whether any of those result identifiers match an Experiment in the current
Measurement Space — if so, the result can be reused.

## Data retrieval modes

When retrieving data from a Discovery Space (e.g. via `ado show entities`),
there are two modes that control whether shared data is included:

<!-- markdownlint-disable line-length -->
| Mode | What is returned |
| --- | --- |
| **measured** | Only Entities and measurements recorded by operations run directly on *this* Discovery Space. Compatible data from other spaces is excluded. |
| **matching** | All Entities and measurements in the Sample Store that are compatible with this Discovery Space, regardless of which space produced them. |
<!-- markdownlint-enable line-length -->

Use **measured** when you want to see only the results your operations have
produced. Use **matching** when you want the full picture including any
compatible data from other spaces.

## Memoization

> [!IMPORTANT]
>
> Each explore operator should provide a way to turn memoization on and off.
> Check the operator documentation.

*Memoization* is the name for data reuse that happens automatically during an
explore operation. It's recommended you also check the documentation on
[operations](../resources/operation.md) and
[explore operators](../operators/explore_operators.md).

When an operation samples an Entity it proceeds as follows:

- The Entity is sampled from the Entity Space
- The Entity's record is retrieved from the Sample Store if present (via its
  unique identifier)
- If **memoization is on**
    - for each Experiment in the Measurement Space, `ado` checks
      if a result for it already exists (via the Experiment's unique identifier)
        - if it does, the result is reused. If there is more than one result,
          they are all reused
- If **memoization is off**
    - existing results are ignored. Each Experiment in the Measurement Space is
      applied again to the Entity. The new results are added to any existing.
