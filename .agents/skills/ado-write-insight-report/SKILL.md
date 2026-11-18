---
name: ado-write-insight-report
description: >-
  Describes how to write an insight report from empirical data: a
  self-contained document that characterises a subject (for example an
  optimization problem instance, workload or system) using results from
  experiments run with one or more methods (solvers, algorithms,
  configurations). Covers structure, evidence tagging, claim verification and
  common mistakes. Use when the user asks to write up, summarise or report the
  findings of a study or experiment campaign; to explain why a problem is hard
  for a solver; to compare solvers or methods on an instance; or to turn a
  study document into a report for readers outside the study.
---

# Writing an Insight Report from Empirical Data

An insight report explains a **subject** (what was studied) using evidence
from experiments with **methods** (what was applied to it). Its readers do
not know the study, its phases or its operations.

## Workflow

1. **Gather sources.** Read the study document (see
   [ado-create-study-document](../ado-create-study-document/SKILL.md)),
   summary CSVs and the measurement data of the operations it names (see
   [ado-examine-operation](../ado-examine-operation/SKILL.md) and
   [ado-examine-space](../ado-examine-space/SKILL.md)).
2. **Find external context.** Locate the set the subject comes from (for
   example a benchmark library): its size, how the studied items were
   selected, published baselines, and external reference URLs.
3. **Check the subject directly.** Compute the structural properties the
   report will claim (sizes, spectra, symmetries, trivial solutions) from the
   input files. Record this as experiment `E0`.
4. **Write the report** with the outline in [template.md](template.md).
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
- The Summary MUST state the ultimate goals of the report, the exact scope
  of the experiments, and the specific solvers evaluated (e.g., CPLEX vs. Gurobi).
- Distinguish cleanly between structural facts and derived insights.
- Never name study phases, internal run labels or campaign steps in the main
  text. Group results by topic ("a search-control screen of X, Y and Z
  found ...") and cite experiments with tags such as `[E7]`.
- Define every symbol and parameter before first use (for example the
  quadratic matrix and the weight parameter in an objective). Replace
  specialist jargon with plain words where one exists.
- State the scope: which items of the wider set were studied, which
  parameter values, and what the conclusions are expected to generalise to.
  Tag that generalisation `[Inferred]`.

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

- **Separate insights into three distinct levels**:
  1. **Problem Class Insights:** Characteristics of the subject/problem
     itself, independent of how it is tackled.
  2. **Formulation Insights:** Characteristics specific to how the problem
     is encoded or described
  3. **Solver Insights:** Behavior specific to applying a particular solver
     or method to a formulation. Number them (`H1`, `F1`) and have each
     lower level cite the level above.
- **Upstream Completeness:** Define all terminology used in analysis in the
  problem class or formulation definition sections.
- **Mandate the "So What?":** When stating characteristics of problem,
  formulation or solver/method explicitly state the practical consequence of
  that characteristic.
- **Justify Parameters:** Explicitly state the rationale for experimental
  boundaries or algorithmic tweaks.
- **Decouple Theory from Empirical Results:** When reviewing method or solver
  mechanics include a clear introductory sentence bridging the theoretical
  friction to the specific problem instances being solved. Avoid an abrupt
  mix of general theory and explicit experimental data. Provide introductory
  framing before directly comparing solvers or methods.
- Omit sections that do not apply. Do not add sentences that only describe
  the report's structure.

## Verification checklist

- [ ] Goals, scope, baseline solvers, and reference links are explicitly
      stated upfront.
- [ ] All variables, costs, and terms are fully defined in the model section
      before use.
- [ ] Every time limit or technical parameter includes a stated justification.
- [ ] Every mathematical property claimed includes an explicit statement of
      its practical impact on the solver.
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
      differences are footnoted, and time formatting is completely unambiguous.
- [ ] Each proposal states why it is valid, that it is untested, and how to
      validate it before use.
- [ ] The summary contains the conclusion and at most a few headline numbers,
      and no run identifiers or configuration detail.
- [ ] Every figure in the text matches the source data.
- [ ] Every path and identifier in the report exists.

## Example

`examples/cplex_mip_qoblib_portfolio/insight_report.md` applies this skill
to two QOBLIB portfolio instances evaluated with CPLEX and Gurobi.
