---
name: ado-write-research-report
description: >-
  Use when the user asks to synthesise a large study, or a group of related
  studies, into a self-contained report or paper for readers outside the
  study: one that explains a subject (for example a problem class, its
  instances, a workload or a system) by drawing insights from many
  experiments with one or more methods (solvers, algorithms,
  configurations). Also use when asked to explain why a subject is hard for a
  class of methods, or to compare methods across a study's experiments.
  Covers report structure, evidence tagging and claim verification. For a
  report on a single operation, discoveryspace or project, use
  ado-examine-operation, ado-examine-space or ado-examine-project.
---

# Writing a Research Report from Empirical Data

A research report explains a **subject** (what was studied) using evidence
from experiments with **methods** (what was applied to it). Its readers do
not know the study, its phases or its operations.

## Workflow

1. **Gather sources.** Read the study documents (see
   [ado-create-study-document](../ado-create-study-document/SKILL.md)),
   summary CSVs and the measurement data of the operations they name (see
   [ado-examine-operation](../ado-examine-operation/SKILL.md) and
   [ado-examine-space](../ado-examine-space/SKILL.md)).
2. **Find external context.** Locate the set the subject comes from (for
   example a benchmark library): its size, how the studied items were
   selected, published baselines, and external reference URLs.
3. **Check the subject directly.** Compute the structural properties the
   report will claim (sizes, spectra, symmetries, trivial solutions) from the
   input files. Record this as experiment `E0`.
4. **Write the report.** Read [assets/template.md](assets/template.md) before
   drafting and follow its section order and insight labels.
5. **Verify** every claim against the checklist below. Fix the report, not
   only the reply to the user.
6. **Save** as a markdown file next to the study's materials. Store it as an
   ado `document` resource only if the user asks (see
   [ado-create-resource-yaml](../ado-create-resource-yaml/SKILL.md)).

## Rules

### Framing

- Write about the subject. The study's own question (for example "can
  method A beat method B") goes in the Appendix as the source of the
  evidence, not in headings or the summary headline.
- Make the Summary readable on its own by someone who has not seen the
  study, and keep it to about one page. It states:
  - the subject, the wider set it belongs to, and which items and parameter
    values were studied
  - the goal of the report and the methods evaluated (for example two named
    solvers)
  - the main characteristics of the subject and their consequences for the
    methods
  - how the methods compare, and the most promising next direction
  - a few headline numbers, each with what it measures
- Distinguish structural facts from derived insights.
- Never name study phases, internal run labels or campaign steps in the main
  text. Group results by topic ("a search-control screen of X, Y and Z
  found ...") and cite experiments with tags such as `[E7]`.
- Replace specialist jargon with plain words where one exists.
- State what the conclusions are expected to generalise to beyond the
  studied items, and tag that generalisation `[Inferred]`.

### Evidence

- Tag every claim and proposal: `[Measured]` (with an experiment reference),
  `[Inferred]` (explanation consistent with the data) or `[Proposed]` (not
  tested). Statements about the internals of a third-party method whose runs
  you only have results for are `[Inferred]`.
- List every experiment in an Appendix catalogue: tag, what it varied,
  items, run length, and metastore operation identifiers.
- Say what cannot be compared, for example a baseline published only as
  final values has no trajectory.

### Content

- **Three levels of insight.** Number each insight with its level's prefix
  and have it cite the insights it depends on from the levels above:
  1. **Subject insights** (`S1`, `S2`, ...): characteristics of the subject
     itself, independent of how it is encoded or tackled.
  2. **Formulation insights** (`F1`, ...): characteristics of a particular
     encoding of the subject.
  3. **Method insights** (`M1`, ...): method traits and the behaviour they
     cause on a formulation. A **method trait** is a property of how a
     method, or a class of methods, works that governs how it responds to
     the subject. For example, for MIP solvers: "they must linearise or
     convexify a non-convex objective before they can bound it", "better
     solutions arrive as rare jumps", "convexified relaxations are
     numerically fragile".
- **Define terms where they belong.** Define each symbol, parameter and term
  in the section that introduces the thing it belongs to, not where it is
  first used. A formulation's parameters belong in that formulation's
  definition, even if they first matter in a later section.
- **State the consequence.** For every characteristic or trait, state its
  practical consequence for the methods or for the user.
- **Justify parameters.** State why each experimental boundary (run length,
  parameter range, item selection) or method adjustment was chosen.
- **Traits before evidence.** Open each method section with the traits that
  matter and the subject or formulation insights they interact with, then
  give the measurements. Introduce each method comparison with what is
  compared and why.
- Omit sections that do not apply. Do not add sentences that only describe
  the report's structure.

## Verification checklist

- [ ] The Summary, read alone, states the subject, scope, goal, methods
      evaluated and main conclusions, and fits on about one page.
- [ ] Reference links to the wider set and published baselines are given.
- [ ] Every symbol, parameter and term is defined where the thing it belongs
      to is introduced.
- [ ] Every run length or experimental parameter has a stated justification.
- [ ] Every characteristic or trait states its practical consequence.
- [ ] Every insight carries an `S`, `F` or `M` label and cites the insights
      it depends on.
- [ ] Every effect size uses one stated, comparable metric, for example the
      fraction of the remaining gap closed rather than percentage change of a
      value near zero.
- [ ] No spread among the best configurations is reported as the effect of
      tuning against defaults.
- [ ] Each blanket claim ("tuning barely matters") was checked for
      counterexamples among all varied parameters.
- [ ] Each observation is generalised only as far as it was checked: an
      observation on one item or parameter value is stated for that item, and
      checked on the others before being stated more widely.
- [ ] Each table compares like with like (same run length, same baseline);
      differences are footnoted, and every duration states its unit.
- [ ] Each proposal states why it is valid, that it is untested, and how to
      validate it before use.
- [ ] The Summary contains no run identifiers or configuration detail.
- [ ] Every figure in the text matches the source data.
- [ ] Every path and identifier in the report exists.
