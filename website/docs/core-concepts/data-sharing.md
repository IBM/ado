# Shared Sample Stores

In `ado` Entities and measurement results are stored in a database called a
**Sample Store**. This document describes how Sample Stores enable sharing of
data. For more general information about these databases see
[their dedicated page](../resources/sample-stores.md).

There are two key points that underpin data reuse in `ado`:

- You can **share** a Sample Store between multiple Discovery Spaces
    - This allows a Discovery Space to (re)use relevant Entities and Measurements
       stored in the Sample Store by operations on other Discovery Spaces
- **Entities are always shared**. There is only one entry in a Sample Store for
  an Entity

> [!NOTE]
>
> To maximize the chance of data-reuse, similar Discovery Spaces should use the
> same Sample Store. However, Discovery Spaces do not have to be similar to use
> the same Sample Store.

## When data can be shared in `ado`

There are two situations where data can be shared between Discovery Spaces in
`ado`:

- **Data Retrieval**: retrieving data about entities and measurements from the
  Discovery Space e.g. `ado show entities space`
- **Data Generation**: When performing an explore operation on a Discovery
  Space - this type of data reuse is called `memoization`

## How `ado` determines what data can be shared

As a quick recap, a Discovery Space is composed of:

- an [Entity Space](entity-spaces.md) which describes a set of Entities (points)
  to be measured
- a [Measurement Space](actuators.md#measurement-space) which describes a set of
  Experiments to apply to the points

### Entities

Each Entity in the Entity Space has a unique identifier, usually determined by
its set of constitutive property values. For example, if an Entity has two
constitutive properties `X` an `Y` with values 4 and 10, its id will be
'X:4-Y:10'. Since the identifiers of all the Entities in the Entity Space are
known, the Sample Store can be searched to see if it contains a record for any
of the Entities.

### Measurements

Each experiment in a Measurement Space has a unique identifier, determined from
its base name plus any optional properties that have been explicitly set. When
an Entity is retrieved from the Sample Store, it contains results of all the
experiments that have been applied to it. If the identifier of a result matches
the identifier of an Experiment in the Measurement Space, `ado` determines it
can be reused.

## Data sharing and data retrieval

When retrieving data from a Discovery Space, e.g. via `ado show entities`, you
are actually retrieving data from the Sample Store that matches the Discovery
Space. When determining what data to retrieve there are two situations to
consider:

- **measured**: retrieve only Entities and measurements that were sampled via an
  operation on the given Discovery Space
    - this can be considered the "no sharing" mode. If an Entity or measurement
      exists in the Sample Store that's compatible with the Discovery Space, but
      no operation on the Discovery Space ever visited it, the "measured" mode
      will not show it
- **matching**: retrieve all Entities and measurements that match the Discovery
  Space
    - this can be considered the "sharing" mode.

## Data sharing and memoization

> [!IMPORTANT]
>
> Each explore operator should provide a way to turn memoization on and off.
> Check the operator documentation.

This section explains how data sharing and reuse works during an explore
operation - a feature called _memoization_. It's recommended you check the
documentation on [operations](../resources/operation.md) and
[explore operators](../operators/explore_operators.md).

Briefly, an explore operation samples a point in the Entity Space of a Discovery
Space and applies the experiments in the Measurement Space to it. In detail, the
sampling process is as follows:

- An Entity is sampled from the Entity Space
- The Entity's record is retrieved from the Sample Store if present (via its
  unique identifier)
- If **memoization is on**
    - for each experiment in the MeasurementSpace, `ado` checks
      if a result for it already exists (via the experiment's unique identifier)
        - if it does, the result is reused. If there is more than one result, they
          are all reused
- if **memoization is off**
     - Existing results are ignored. Each experiment in the Measurement Space is
       applied again to the Entity. The new results are added to any existing.
