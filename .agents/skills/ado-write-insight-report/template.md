# Insight report template

Replace each section's guidance with content. Omit any section or
subsection that does not apply.

```markdown
# Insight report: <subject>

## Summary

<Context: the wider set the subject belongs to and how the studied items
were selected. Two to four bullets on the characteristics that make the
subject hard. Consequences for the methods. One or two sentences on how the
methods compare, pointing to the comparison section. The most promising
next direction.>

**Evidence.** <Where the data came from, one sentence, pointing to the
Appendix. Key to the tags [Measured], [Inferred], [Proposed].>

## 1. Subject anatomy and structural hardness

### 1.1 The model
<What the subject is, in plain terms. Define every symbol and parameter.>

### 1.2 Items studied
<Table of the wider set. Which items and parameter values were studied, and
what "instance" means in the rest of the report.>

### 1.3 Characteristics that make it hard
<H1, H2, ...: properties of the subject itself, independent of any method,
each with evidence tags.>

## 2. Reformulations

### 2.1 Why reformulate
<Why changing the formulation helps on this subject.>

### 2.2 <Tested reformulation>
<What it is, validity check, measured effect.>

### 2.3 <Proposed reformulation>
<Why it is valid, dependencies on other proposals, how to validate it.>

## 3. Behaviour of <class of methods>

### 3.1 Friction common to <class>
<F1, F2, ...: each cites the H it comes from. Effect-size table using one
comparable metric.>

### 3.2 <Method A>
**Algorithmic friction** <specific to A, citing F and H>
**Concrete evidence** <grouped by topic, with experiment tags>
**Recommendations for <method A> on these instances**

### 3.3 <Method B>
<Same structure. State the limits of the evidence for B.>

## 4. How the methods compare

<Like-for-like tables (for example bound and solution quality by
formulation). What each method is stronger at, and what cannot be
compared.>

## Appendix: evidence provenance

### Source of the evidence
<The study's question, hardware, software versions, seeds, time limits,
paths to the input files and tools.>

### Experiment catalogue
| Ref | Experiment | Items | Operations |
|---|---|---|---|
| E0 | Direct analysis of the input files for this report | ... | none |
```
