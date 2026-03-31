---
name: conduct-empirical-study
description: End-to-end workflow for conducting empirical studies with ado — systematic
  exploration and analysis of entity spaces. Covers problem formulation, optional
  implementation of custom experiments/operators, execution (local or remote), and
  analysing results. Use when the user wants to run experiments, answer research
  questions empirically, benchmark systems, or perform any study involving systematic
  data collection across a parameter space.
---

# Conducting an Empirical Study with ado

ado can perform any study involving systematic exploration and analysis of a space
of entities: executing experiments to answer research questions, benchmarking, or
any task where data must be collected across a parameter space.

## Workflow Overview

Five sequential steps. Steps 2 and 3 are optional, but if step 2 is performed,
step 3 must follow.

```text
1. Formulate  →  2. (Optional) Implement  →  3. (Optional) Complete
  →  4. Execute  →  5. Analyse
```

---

## Step 1: Formulate the Discovery Problem

Follow the [formulate-discovery-problem](../formulate-discovery-problem/SKILL.md)
skill to frame the problem with ado.

Gather user input on formulation details before deciding the next step:

- Preferred exploration technique (random search, space-filling, multi-objective
  optimization, etc.)
- Values for actuator configuration

**Outcome A** — All required experiments and analysis tools exist:
skip to Step 4.

**Outcome B** — Required experiments or analysis tools are missing: proceed to
Step 2.

---

## Step 2: (Optional) Implementation Phase

Required only when Step 1 identified missing experiments, actuators, or operators.

Follow [plugin-development.mdc](../../rules/plugin-development.mdc) to implement
the needed components.

Gather user input on implementation details:

- Whether experiments should be actuators or custom experiments
- Which parameters should be required vs. optional in a custom experiment
- Fields needed in actuator configurations or operator parameters

---

## Step 3: (Optional) Complete Problem Formulation

Required if Step 2 was performed. Return to the
[formulate-discovery-problem](../formulate-discovery-problem/SKILL.md) skill and
complete the formulation, now incorporating the new components created in Step 2.

---

## Step 4: Execute the Empirical Study

Execute the plan from Step 1 or Step 3.

**Local execution**: create and start the operation from the repo root (verify
flags with `uv run ado create operation --help`):

```bash
uv run ado create space -f space.yaml
uv run ado create operation -f operation.yaml --use-latest space
```

For CLI conventions, shortcuts, and debugging, see
[using-ado-cli](../using-ado-cli/SKILL.md).

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

## Step 5: Analyse Results

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
