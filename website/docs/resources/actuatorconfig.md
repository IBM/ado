<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
Some actuators expose parameters that can be configured. For these actuators an
`actuatorconfiguration` resource define a particular set of values for these
parameters. When creating an `operation` resource you can then optionally
specify the `actuatorconfiguration` resources to retrieve to configure the
actuators used in the operation.

!!! info end

    See
    [enabling custom configuration of an actuator](../actuators/creating-actuator-classes.md#enabling-custom-configuration-of-an-actuator)
    for how to enable this configuration capability for an actuator

Some examples of configuration parameters are:

- location of storage e.g. pvc name
- A rest end-point
- what compute resources are available

Often these parameters capture things that change depending on the system the
actuator is running on. For example, one system has A100 GPUs but another
doesn't, and the configuration allows the actuator to know before trying to
submit an experiment that it can't be executed on the given system.

!!! important end

    Not all actuators can be configured

!!! important end

    By convention the value of actuator configuration parameters should not change
    the results of a given experiment on a given entity.

## creating an `actuatorconfiguration`

Similar to `operation` resources creating an `actuatorconfiguration` involves
writing a YAML containing values for the configurable actuator parameters.

### Getting input options and potential values

The available parameters can be found by checking the actuator documentation or
a default configuration can be obtained using

```commandline
ado template actuatorconfiguration --actuator-identifier $ACTUATORNAME
```

The output file will be called
"$ACTUATORNAME_actuatorconfiguration_template_$UID.yaml"

The schema of the configuration YAML, which includes documentation on each
field, can additionally be output using

```commandline
ado template actuatorconfiguration --actuator-identifier $ACTUATORNAME --include-schema
```

The schema will be output to a file called
"$ACTUATORNAME_actuatorconfiguration_template_$UID_schema.yaml"

### Creation and validation

Once the values you want are set and saved in a file called
`$ACTUATOR_CONFIGURATION_FILE` then the `actuatorconfiguration` resource can be
created with:

```commandline
ado create actuatorconfiguration -f $ACTUATOR_CONFIGURATION_FILE
```

This will return an identifier with the format
"actuatorconfiguration-$ACTUATORNAME-$UID" e.g.

```commandline
> Success! Created actuator configuration with identifier `actuatorconfiguration-robotic_lab-3edc9cd3`
```

On `create` the provided parameters are first passed to the actuator for
validation. If it detects any errors you will get a warning indicating the issue
and the resource will not be created.

## Using an `actuatorconfiguration`

`actuatorconfiguration`'s are used when creating `operation` resources. See
[specifying actuator parameters](operation.md#passing-actuator-parameters) in
the `operation` resource documentation for details.

### Other ado commands that work with actuatorconfiguration

- `ado get actuatorconfigurations`
  - list stored `actuatorconfiguration`s or retrieve their representations
- `ado show related actuatorconfiguration ID`
  - show operations using an `actuatorconfiguration`
- `ado edit actuatorconfiguration ID`
  - set the name, description, and labels for an `actuatorconfiguration`
- `ado delete actuatorconfiguration ID`
  - delete an `actuatorconfiguration`
