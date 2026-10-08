---
name: conduct-empirical-study
description: >-
  End-to-end workflow for using ado to conduct an empirical study — systematic exploration
  of entity spaces/configuration spaces via experiment campaigns and analysis
  of the results. Covers problem formulation, implementation of
  custom experiments or analysis operators, campaign execution (local or remote), and analysing
  results. Use when the user wants to run a multi-entity study, answer research
  questions empirically, benchmark systems, or collect data across a parameter
  space.
---

# Conducting an Empirical Study with ado

ado can perform any study involving systematic exploration and analysis of a space
of entities: executing experiments to answer research questions, benchmarking, or
any task where data must be collected across a parameter space.

For measuring **one** entity/point (functional check), use
`run_experiment` — see [run-experiment](../run-experiment/SKILL.md) — not this
workflow.

## Basic Skill Requirements

Ensure you read the following basic skills first:

- [using-ado-cli](../using-ado-cli/SKILL.md)
- [Querying ado data](../query-ado-data)

## Workflow Overview

Six sequential steps. Steps 3 is optional.

1. Check Prior Work
2. Define/Refine Study
3. (Optional) Implement Experiments
4. Define Experiment Campaign
5. Execute Experiment Campaign
6. Analyse

---

## Step 1: Check Prior Work

To perform this step read:

- [Create Research Study Document](../create-research-study-document)

List existing study documents

```commandline
ado get documents --details --no-trunc --output-file studies.txt
```

Identify any relevant ones. For the ones identified run

```commandline
ado get document $DOCUMENTID -o yaml --output-file $ID.yaml
```

Find the relevant study labels and then query for resources that have those labels.
[See the examples](../create-research-study-document/SKILL.md#query).

Form a picture of the current study state. In particular:

- What and how much has been explored
- What experiments have been used
- What relevant analysis operators exists

---

## Step 2: Define or Refine the Study

At the start of a named study (and when objectives or next steps change), create
or refresh a study document and apply its study labels to spaces/operations —
see [create-research-study-document](../create-research-study-document/SKILL.md).

Decide if this constitutes a new study or is an extension of an existing
one (matches the scope of an existing study).

If it is an extension of an existing study decide if the study
document should be refreshed with additional information.
In any case use that studies labels where possible in the subsequent steps.

If it constitutes a new study write a study document for it.

---

## Step 3: (Optional) Implementation Phase

Read [define-experiment-campaign](../define-experiment-campaign/SKILL.md)

Decide if any new experiments or analysis operators are required for the
study.

Follow [plugin-development](../plugin-development/SKILL.md) to implement
the needed components.

Gather user input on implementation details:

- Whether experiments should be actuators or custom experiments
- Which parameters should be required vs. optional in a custom experiment
- Fields needed in actuator configurations or operator parameters
- Details on analysis operators

---

## Step 4: Define Experiment Campaign

Define an initial experiment campaign following [define-experiment-campaign](../define-experiment-campaign/SKILL.md).

---

## Step 5: Execute the Experiment Campign

Execute the experiment campaign.

**Local execution**: create and start the operation from the repo root (verify
flags with `uv run ado create operation --help`):

```bash
uv run ado create space -f space.yaml
uv run ado create operation -f operation.yaml --use-latest space
```

**Remote execution**: follow [remote-execution](../remote-execution/SKILL.md).

Prefer remote execution when the study requires:

- Many experiments in parallel
- Computationally expensive experiments or analysis
- Accelerators (GPUs, etc.)

Gather user input on execution details:

- Local vs. remote execution
- Project context to use
- Remote execution context details (cluster, environment)

---

## Step 6: Analyse Results

After the operation has produced data, use:

- [examining-ado-operations](../examining-ado-operations/SKILL.md), to
examine the data produced
- [examining-discovery-spaces](../examining-discovery-spaces/SKILL.md),
to inspect the space operated on, especially when the space
has been explored by multiple operations e.g.
  - between phases of a multi-step study
  - when there was pre-existing data in the space

---

## Guidelines

### Complex multi-step studies

Some studies require results from an initial round of data collection before the
next steps can be determined (e.g. an exploratory phase followed by focused
analysis). In these cases:

- Formulate only the first set of independent steps
- Execute, then apply Step 5 to analyse those results
- Report the potential follow-on steps and revisit the workflow once initial
  results are available

Do not attempt to formulate the full study upfront if later steps depend on
earlier empirical findings.
