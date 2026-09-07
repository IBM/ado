---
name: run-experiment
description: >-
  Use the run_experiment tool to execute one experiment on a single point
  (entity) without creating a discoveryspace or operation. Use when smoke
  testing or functionally validating an actuator, custom experiment, or actuator
  configuration; debugging experiment execution or entity validation; or when a
  single measurement is wanted without ado metastore tracking. For campaigns
  over many entities, see define-experiment-campaign; for the ado CLI itself,
  see using-ado-cli.
---

# Running an experiment on a single point

`run_experiment` is a **separate CLI entry point**, not an `ado` subcommand.
There is no `ado run_experiment`. It runs one entity through one or more
experiments and prints the result — nothing is written to the metastore.

Use it for:

- Smoke testing an actuator or custom experiment after installing a plugin
- Debugging experiment execution or entity validation
- Taking a single measurement when tracking is not needed

For measuring multiple entities, create a `discoveryspace` and an
`operation` instead — see
[define-experiment-campaign](../define-experiment-campaign/SKILL.md).

## Usage

```bash
uv run run_experiment PATH_TO_POINT_YAML
```

Ray is started locally and shut down automatically; no metastore or remote
cluster is required.

## Point YAML

The point.yaml has two top level fields:

- `entity` is a map of constitutive property name/value pairs.
These are the input parameter name/value combinations to the experiment.
- `experiments` is a
list of experiment references to run on that entity.

```yaml
entity:
  mass: 8
  volume: 4
experiments:
  - actuatorIdentifier: custom_experiments
    experimentIdentifier: calculate_density
    experimentVersion: 1.0.0 # Omit if the experiment has no version
```

Get the identifiers to use with:

```bash
# Actuators and the experiments they provide
uv run ado get actuators --details

# Required and optional consitutive properties of one experiment
uv run ado describe experiment ACTUATOR_ID.EXPERIMENT_ID
```

## Options

Verify with `uv run run_experiment --help` before writing a command.

<!-- markdownlint-disable line-length -->

| Option                                 | Purpose                                                                                           |
|----------------------------------------| ------------------------------------------------------------------------------------------------- |
| `--no-validate`                        | Skip entity validation. Needed when the experiment is not installed locally                       |
| `--actuator-config-id ID`              | Apply an actuatorconfiguration from the active context's metastore. Repeat for multiple actuators |
| `--remote ENDPOINT`                    | Execute via an ado REST API endpoint instead of locally                                           |
| `--timeout SECONDS`                    | Timeout for a remote experiment (default 300)                                                     |
| `--request-timeout SECONDS`            | Timeout for individual web requests to a remote endpoint (default 60)                             |
| `--verify-certs` / `--no-verify-certs` | SSL certificate verification of remote hosts (default `--no-verify-certs`)                        |

<!-- markdownlint-enable line-length -->

## Interpreting the output

`run_experiment` prints the point, whether the entity validated, and then the
result as a pandas Series holding the request and entity identifiers, the
entity's constitutive property values, and the experiment's target output
properties.

- **"Entity is not valid"**: the entity is missing a required constitutive
  property, or a value is outside the property's domain. Check the requirements
  with `uv run ado describe experiment ACTUATOR_ID.EXPERIMENT_ID`.
- **"Failed to initialize actuator"**: the actuator raised during startup —
  usually a missing dependency or bad actuator configuration.
- **"Measurement request failed unexpectedly"**: the experiment itself raised.
  Re-run with `LOGLEVEL=10` set to see the actuator traceback.

## Related Resources

- [define-experiment-campaign](../define-experiment-campaign/SKILL.md) — running
  experiments over many entities with a discoveryspace and operation
- [using-ado-cli](../using-ado-cli/SKILL.md) — the `ado` CLI: syntax, flags, and
  shortcuts
- [plugin-development](../plugin-development/SKILL.md) — plugin testing
  checklist, where `run_experiment` is the experiment execution check
- [docs/user-guide/advanced/run-experiment.md](../../../docs/user-guide/advanced/run-experiment.md)
  — user documentation for this tool
