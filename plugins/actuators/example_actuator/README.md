# Example Actuator

This repository contains an example for creating an actuator for `ado`.

For more about actuators - what they represent, how to create them, etc. - see
the
[`ado` documentation](https://ibm.github.io/ado/actuators/working-with-actuators/).

This example defines an actuator called `robotic_lab` with one experiment called
`peptide_mineralization`. The example is fully installable and works as-is,
the only caveat being that it uses made up properties for measurement.

## Installing

To install, run the following command in the directory containing this README:

```bash
pip install .
```

You can then confirm it's installed with:

```bash
ado get experiments --details
```

You will see:

<!-- markdownlint-disable line-length -->

```commandline
┌─────────────┬────────────────────────┬────────────────────────────────────────────────────────┐
│ ACTUATOR ID │ EXPERIMENT ID          │ DESCRIPTION                                            │
├─────────────┼────────────────────────┼────────────────────────────────────────────────────────┤
│ mock        │ test-experiment        │                                                        │
│ mock        │ test-experiment-two    │                                                        │
│ robotic_lab │ peptide_mineralization │ Measures adsorption of peptide lanthanide combinations │
└─────────────┴────────────────────────┴────────────────────────────────────────────────────────┘
```

<!-- markdownlint-enable line-length -->

On the last line, you can see the new actuator and its experiment.

## Create a `discoveryspace` and Operation

You can create a `discoveryspace` and run `operations` on it using this example
actuator. The files in `yamls/` provide some examples.

```bash
ado create samplestore --new-sample-store
```

A `samplestore` is a database for storing entities and measurement results. It
can be reused with multiple `discoveryspaces`.

1. Create a
   [discoveryspace](https://ibm.github.io/ado/resources/discovery-spaces/):

   ```bash
   ado create space -f yamls/discoveryspace.yaml --use-latest samplestore
   ```

   You can use `ado get` or `ado describe` on the `discoveryspace` using the
   identifier output by the command above.

2. Create a random walk
   [operation](https://ibm.github.io/ado/resources/operation/):

   ```bash
   ado create operation -f yamls/random_walk_operation.yaml --use-latest space
   ```

At this point, you can try `ado show measurements` to get sampled entities or
apply other operators. The actuator is already fully integrated with `ado`: all
you need to do is have it perform "real" experiments.

## Parameterizable Experiments

This actuator demonstrates how to define parameterizable experiments. There are
experiment that define optional properties with default values. Users can then
create different variants of the base experiment by changing defaults or moving
optional properties into the entity space.

Some examples of how parameterizable experiments can be used:

- [`discoveryspace_override_defaults.yaml`](yamls/discoveryspace_override_defaults.yaml)

  - Demonstrates changing the default of one of the optional properties.

- [`discoveryspace_optional_parameter_in_entity_space.yaml`](yamls/discoveryspace_optional_parameter_in_entity_space.yaml)

  - Demonstrates using one of the optional parameters to define the entities in
    the entity space.

- [`discoveryspace_multiple_parameterizations.yaml`](yamls/discoveryspace_multiple_parameterizations.yaml)
  - Demonstrates using two parameterizations of the same base experiment.

## The Actuator Package: Key Files

The actuator package is located under `src/robotic_lab_actuator/`. This plugin
uses the src layout, which is the recommended structure for Python packages.

Key files include:

- `actuator.py` (**Required**, but can have any name)

  - Contains the `robotic_lab` actuator in this example.
  - Each actuator plugin must have at least one Python module containing one
    actuator.
  - The actuator class must be registered via the entry point in
    `pyproject.toml`.

- `experiments.yaml` (**Optional**)

  - Contains YAML definitions of the experiments defined by the actuator.
  - This list could also be created via code in a Python module.

- `experiment_executor.py` (**Optional**)
  - Contains code that:
    - Determines the values for experiment parameters from the passed entity and
      experiment.
    - Sends the measured property values back to the orchestrator.

## Renaming the Actuator

There are three components you can rename independently:

- **Package name installed via pip** (currently `'robotic_lab'`)

  - Change the `name` field in `pyproject.toml`.

- **Python module name** (currently `robotic_lab_actuator`)

  - Rename the `robotic_lab_actuator` directory under `src/` to your desired
    name.
  - Update the package name under `[tool.setuptools.package-data]` if used.
  - Update the entry point in `[project.entry-points."ado.actuators"]` to
    reflect the new module path.

- **Actuator identifier seen by users**
  - Change the `identifier` field of `RoboticLab` in `actuator.py`.
  - Change the `identifier` fields of the experiments in `experiments.yaml`.
