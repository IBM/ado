# Example actuator

This repository contains an example for creating an actuator for `ado`.

For more about Actuators, what they represent, how to create them etc., see the
`ado` [docs](https://ibm.github.io/ado/actuators/working-with-actuators/).

This example defines an actuator called "robotic_lab" with one experiment called
"peptide_mineralization". The example is fully installable and works as is - the
only caveat being it makes up properties it measures.

## Installing

To install run in the directory with this README

```commandline
pip install .
```

You can then confirm its installed with

```commandline
ado get actuators --details
```

You will see

```commandline
   ACTUATOR ID   CATALOG ID           EXPERIMENT ID  SUPPORTED
0         mock         mock         test-experiment       True
1         mock         mock     test-experiment-two       True
2  robotic_lab  robotic_lab  peptide_mineralization       True
```

On the last line you can see the new actuator and the experiment

## Create a `discoveryspace` and operation

You can create a `discoveryspace` and run `operations` on it with this example
actuator. The files in `yamls/` give some examples.

```commandline
ado create samplestore --new-sample-store
```

A `samplestore` is a database for storing entities and measurement results. It
can be reused with multiple `discoveryspaces`. The above command will output an
identifier, record this for the next step. We will refer to it as
`$SAMPLE_STORE_IDENTIFIER`.

1. Create a [discoveryspace](https://ibm.github.io/ado/resources/discovery-spaces/)

    ```commandline
    ado create space -f yamls/discoveryspace.yaml --set "sampleStoreIdentifier=$SAMPLE_STORE_IDENTIFIER"
    ```

    Record the id output by above. We will refer to it as
    `$DISCOVERY_SPACE_IDENTIFIER`. At this point you can also `ado get` or
    `ado describe` the `discoveryspace`

2. Create a random walk [operation](https://ibm.github.io/ado/resources/operation/)

    ```commandline
    ado create operation -f yamls/random_walk_operation.yaml --set "spaces[0]=$DISCOVERY_SPACE_IDENTIFIER"
    ```

At this point you can try `ado show entities` to get entities sampled, or apply
other Operators. The actuator is already fully integrated in `ado` - all you
need to do is have it perform "real" experiments.

## Parameterizable Experiments

This actuator demonstrates how to define parameterizable experiments. There are
experiment that define optional properties with default values. Users can then
create different variants of the base experiment by changing defaults or moving
optional properties into the entity space.

Some examples of how parameterizable experiment can be used are:

- [discoveryspace_override_defaults.yaml](yamls/discoveryspace_override_defaults.yaml)
  - Shows changing the default of one of the optional properties
- [discoveryspace_optional_parameter_in_entity_space.yaml](yamls/discoveryspace_optional_parameter_in_entity_space.yaml)
  - Shows using one of the optional parameters to define the Entities in the
    entity space
- [discoveryspace_multiple_parameterizations](yamls/discoveryspace_multiple_parameterizations.yaml)
  - Shows using two parameterizations of the same base experiment

## The Actuator Package: Key Files

The actuator package is under `ado_actuators/robotic_lab_actuator`. Note all
actuator packages must be under a directory called `ado_actuators` as this is
the name of the namespace package that contains all `ado` actuator plugins.

The key files are:

- actuator_definitions.yaml (REQUIRED)
  - This defines which classes in which modules of your package contain
    Actuators.
- actuators.py (REQUIRED but can have any name)
  - In this example this python module contains our one actuator, `robotic_lab`.
  - Each Actuator plugin has at least one python module containing one actuator,
    however the name can be anything.
  - It just needs to be the same name as in `actuator_definitions.yaml`
- experiments.yaml (OPTIONAL)
  - In this example this file contains the definitions of the experiments the
    actuator defines as YAML
  - This list could also be created via code in a python module
- experiement_executor.py (OPTIONAL)
  - In this example this file contains the code that
    - determines the values for the experiment parameters from the passed Entity
      and Experiment
    - sends the measured property values back to the orchestrator

## Renaming the Actuator

There are three different things you can rename independently if you like

- Change the name pip installs the package as (currently 'robotic_lab') ->
  Change the name field in pyproject.toml
- Change the python module name to $NAME i.e instead of
  `import ado_actuators.robotic_lab_actuator`
  - Change the name of the `robotic_lab_actutor` directory under
    'ado_actuators/' to $NAME
  - Change the package name under [tool.setuptools.package-data], if using, to
    $NAME
  - Change the `actuator_definitions.yaml` to use $NAME
- Change the identifier of the actuator as seen by the user to $ID i.e. what
  they see in `get actuators` and use when creating `discoveryspace` with the
  actuators experiments.
  - Change the `identifier` field of RoboticLabActuator in `actuator.py` to $ID
  - Change the `identifier` fields of the experiments in `experiment.yaml` to
    $ID
