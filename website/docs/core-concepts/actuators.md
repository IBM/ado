<!-- markdownlint-disable-next-line first-line-h1 -->
## Experiments

An **Experiment**
measures the values of a set of output properties given a set of input
properties. Each time an Experiment is applied to an
[Entity](entity-spaces.md) it produces a measurement result.

### Inputs and Outputs

Experiments define two things:

- **Inputs** — the values an Experiment needs in order to run. Each input
  restricts the values it accepts through a **Property Domain** (for example,
  a list of allowed model names, or any integer within a range). See
  [Properties and Domains](properties-and-domains.md) for the full list of
  domain types.
- **Outputs** — the properties the Experiment measures and records. Because
  many Experiments may target the same concept (e.g. `tokens_per_second`),
  each output is namespaced to the Experiment that produced it — see
  [Target and Observed Properties](#target-and-observed-properties).

### Example

Below is the description of `robotic_lab.peptide_mineralization`, an Experiment
that measures the adsorption of peptide and lanthanide combinations in a
robotic biology lab:

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: robotic_lab.peptide_mineralization
Description: Measures adsorption of peptide lanthanide combinations


Required Inputs:

   Constitutive Properties:
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: peptide_identifier
     Description: The identifier of the peptide to use

     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['test_peptide', 'test_peptide_new']

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: peptide_concentration
     Description: The concentration of the peptide

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: lanthanide_concentration
     Description: The concentration of lanthanide

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Values: [0.1, 0.4, 0.6, 0.8]

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

Optional Inputs and Default Values:

    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: temperature
     Description: The temperature at which to execute the experiment

     Domain:

        Type: CONTINUOUS_VARIABLE_TYPE
        Range: [0, 100]

     Default value: 23
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: replicas
     Description: How many replicas to average the adsorption_timeseries over

     Domain:

        Type: DISCRETE_VARIABLE_TYPE
        Interval: 1
        Range: [1, 4]

     Default value: 1
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
     Identifier: robot_identifier
     Description: The identifier of the robot to use to perform the experiment

     Domain:

        Type: CATEGORICAL_VARIABLE_TYPE
        Values: ['harry', 'hermione']

     Default value: 'hermione'
    ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

Outputs:
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   peptide_mineralization-adsorption_timeseries
   peptide_mineralization-adsorption_plateau_value
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
<!-- markdownlint-enable line-length -->

The example shows:

- **Required inputs** — `peptide_identifier`, `peptide_concentration`, and
  `lanthanide_concentration` must always be provided. Each declares a domain
  that restricts the valid values.
- **Optional inputs** — `temperature`, `replicas`, and `robot_identifier` each
  have a default value and can be overridden.
- **Outputs** — two properties are measured and recorded:
  `adsorption_timeseries` and `adsorption_plateau_value`.

### Required Inputs

Values must be provided for all required inputs before the Experiment can run.
Providing a value outside the declared domain is an error.

Most required inputs are **constitutive properties** — values that describe the
Entity being measured, such as a model name or a concentration. However, an
input can also be an **observed property** produced by another Experiment: if
Experiment `B` requires a value that Experiment `A` produces, Experiment `A`
must have been run on the Entity first.

See [Properties and Domains](properties-and-domains.md) for a full description
of constitutive and observed properties and all domain types.

### Optional Inputs

Experiments can also declare optional inputs that have default values.
The defaults can be overridden to create **parameterized experiments** — useful
when you want to fix certain settings while exploring others. This is described
further in the
[`discoveryspace` resource documentation](../resources/discovery-spaces.md).

### Target and Observed Properties

Experiments declare the properties they intend to measure — these are called
**target properties**. However, many different Experiments might target the
same property (e.g. `tokens_per_second`) measured in different ways. To
distinguish them, the actual value recorded by Experiment `A` for target
property `X` is called an **observed property**, named `A-X`.

In the example above:

- `adsorption_plateau_value` is the **target property** — the concept being
  measured.
- `peptide_mineralization-adsorption_plateau_value` is the **observed property**
  — that value as recorded by this specific Experiment.

For a full description of property types see
[Properties and Domains](properties-and-domains.md).

## Measurement Space

A Measurement Space is a collection of [Experiments](#experiments).
As a result a Measurement Space also defines a set of observed properties and target
properties as follows

Property Type | Measurement Space Definition
--- | ---
Observed | Union of the observed property sets of it Experiments
Target | Union of the target property sets of it Experiments

When combined with an Entity Space, a Measurement Space forms a
[Discovery Space](discovery-spaces.md).

## Actuators

Experiments are grouped and provided by **Actuators**. An Actuator typically
covers a particular domain - for example, foundation model fine-tuning,
computational chemistry, or robotic biology - and provides a collection of
related Experiments for that domain.

A primary way to extend `ado` is by developing new Actuators to support
Experiments in a new domain.

The [Actuator documentation](../actuators/working-with-actuators.md)
has more detail including how to see the Actuators and Experiments
available in your deployment.
