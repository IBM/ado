---
name: examining-ado-project
description: >-
  Builds a picture of work in an ado project: activity volume,
  spaces and operations created over time, experiments and operation configs 
  used etc. Use to create a project/context overview report,
  summarize what the team has been doing in an ado project, report trends across
  spaces/operations, or to onboard onto an ado project.
---

# Examining an ado Project

End-to-end workflow to summarize **all** discoveryspaces, operations, and
related metadata in the ado project associated to the active context.

- Run all commands from the **repository root** with `uv run` (see
  [using-ado-cli](../using-ado-cli/SKILL.md)).
- See [projects and contexts](../../../website/docs/resources/metastore.md#contexts-and-projects)
  for details on what projects and contexts are
- NOTE: Users may refer to an ado project using either the term "project" or "context"

## Tips

- Prefer metastore listing and YAML dumps before heavy `uv run ado show` data pulls;
  see [query-ado-data](../query-ado-data/SKILL.md).
- For one space in depth: [examining-discovery-spaces](../examining-discovery-spaces/SKILL.md).
- For one operation in depth: [examining-ado-operations](../examining-ado-operations/SKILL.md).

## Pre-requisites - Check active context is associated with expected project

Context names and project names are identical by construction.

If asked to examine a specific named project:

1. uv run ado context - see if it has the correct name
   a. If yes continue to [next step](#1-overview-activity-and-types)
   b. if no, execute `uv run ado get contexts` - see if there is another context
      with matching name
    i. if there is switch to that context: `uv run ado context $NAME`
    ii. if not inform user a context connecting to the project the specified
        cannot be found

## 1. Overview: activity and types

Goal: volume of work, recency, and which spaces attract the most operations.

1. **Spaces (tabular, with metadata)**

   ```bash
   uv run ado get spaces --details
   ```

   Use **age** (list is age-sorted, most recent last), **name**, **description**,
   and **labels** to infer themes and activity.

2. **Operations (tabular, with metadata)**

   ```bash
   uv run ado get operations --details
   ```

   Relates operations to target spaces; age and labels summarize recent work.

    **Heuristic — operation IDs:** many ids encode the operator and a version
    segment, e.g. `OPERATOR_NAME-VERSION-...-UID` (exact shape varies). Use this
    together with `uv run ado get operators --details` to understand more about
    the operators used.

**Synthesis:** cluster mentally (or in notes) by **creation time** to see bursts
of activity; count operations **per space** from the operations listing to see
which spaces are busiest.

## 2. Deeper pass: full YAML and experiments

Goal: experiments/actuators in use, operation parameters, and how much was
**submitted** for measurement (entities selected by explore-style operators—may
differ from completed measurements if runs failed).

Dump full resource documents for easier scripted or batched reading:

```bash
uv run ado get spaces -o yaml > spaces.yaml
uv run ado get operations -o yaml > operations.yaml
```

On **large metastores**, these files can be very large—prefer `uv run ado get …
-q …` filters, scripts, or the SQL store API (see
[query-ado-data](../query-ado-data/SKILL.md)) before dumping everything.

When interpreting YAML fields, confirm paths against resource schemas:

```bash
uv run ado template discoveryspace --include-schema
uv run ado template operation --include-schema
```

From **space** YAML: note **experiment** and **actuator**  
identifiers referenced by each space.

Gain further information on the experiments using

```bash
# Outputs description of actuators used
uv run ado get actuators --details 
# Outputs description of experiments used 
uv run ado get experiments --details 
# Outputs detailed information on an experiment 
# Use to drill down into most used experiments
uv run ado describe experiment $EXPERIEMENTID
```

From **operation** YAML: note actuatorconfigurations used (if any),
read **parameters** (operator-specific), target
**discoveryspace** references, and any fields that indicate **entity count /
batch / sample** configuration for explore operations.

**Synthesize:**

- What is being **measured** and with which experiments (domain of the project).
- Which **explore** (or similar) operations drove the largest submitted entity
  sets and on **which spaces**.
- Whether **experiment definitions or parameters** shift over time (versions,
  configs).

## 3. Space relationships and entity-space shape

From step 2, pick a handful of **high-traffic or recent** spaces. For each,
inspect its fragment in `spaces.yaml`, focusing on **entity space** structure
(dimensions, bounds, representation).

Find **Spaces that match these spaces** (refinement, expansion, or parallel
configurations)

```bash
uv run ado get spaces --matching-space-id SPACE_ID
```

to build a picture of how they are related.

Optionally complement with one-hop links:

```bash
uv run ado show related space SPACE_ID
```

## 4. Report template

Write a concise markdown report

- Write the report to `reports/<ado_context_name>/` (create the directory if
  needed), where `ado_context_name` is the **active ado metastore context**
  (`uv run ado context`).
- Write the report as `project_<YYYY-MM-DD>_report.md`.
- If a report already exists check if there has been any activity
in the project since it was written - if not ask user if they want
to replace it.

### Project summary

- Domains or problems implied by experiments, actuators, and space descriptions.
- Dominant **operation** and **experiment** patterns.

### Latest activity

- Most recent spaces and operations (from `--details` listings).
- What the latest work seems focused on (labels, names, target spaces).

### Spaces overview

- Which spaces are most used and most analyzed.
- How **entity spaces** and **matching-space** relationships evolve: expanding,
  narrowing, or shifting configuration.

### Operations overview

- Operator mix and whether **parameters** or **operator choice** evolve over time.
- Which operations **submitted** the most entities (from operation YAML/config).
- What **analysis**-style operations ran (infer from operator names and
  parameters).
