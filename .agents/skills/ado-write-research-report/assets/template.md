# Research report: `<subject>`

## Summary

<Self-contained, about one page. The subject, the wider set it belongs to,
how the studied items and parameter values were selected. The goal of the
report and the methods evaluated. Two to four bullets on the main subject
and formulation insights. Their consequences for the methods. One or two
sentences on how the methods compare, pointing to the comparison section.
A few headline numbers, each saying what it measures. The most promising
next direction.>

**Evidence.** <Where the data came from, one sentence, pointing to the
Appendix. Key to the tags [Measured], [Inferred], [Proposed].>

## 1. The subject

### 1.1 The model

<What the subject is, in plain terms. Define every symbol and parameter of
the subject.>

### 1.2 Items studied

<Table of the wider set. Which items and parameter values were studied, and
what "instance" means in the rest of the report.>

### 1.3 Subject insights

<S1, S2, ...: characteristics of the subject itself, independent of any
encoding or method, each with its consequence and evidence tags.>

## 2. Formulations

### 2.1 Why the formulation matters

`<How the choice of encoding affects the methods on this subject.>`

### 2.2 `<Tested formulation>`

<Definition, including all of its parameters. Validity check. F1, ...:
its formulation insights, each citing the S insights it depends on, with
measured effect.>

### 2.3 `<Proposed formulation>`

<Definition, why it is valid, dependencies on other proposals, how to
validate it.>

## 3. Behaviour of `<class of methods>`

### 3.1 Traits common to `<class>`

<M1, M2, ...: method traits shared by the class, each citing the S and F
insights it interacts with. Effect-size table using one comparable metric.>

### 3.2 `<Method A>`

**Method traits** <specific to A, citing M, F and S>
**Evidence** <grouped by topic, with experiment tags>
**Recommendations for `<method A>` on these items**

### 3.3 `<Method B>`

`<Same structure. State the limits of the evidence for B.>`

## 4. How the methods compare

<What is compared and why. Like-for-like tables (for example bound and
solution quality by formulation). What each method is stronger at, and what
cannot be compared.>

## Appendix: evidence provenance

### Source of the evidence

<The studies' questions, hardware, software versions, seeds, run lengths,
paths to the input files and tools.>

### Experiment catalogue

| Ref | Experiment                                         | Items | Run length | Operations |
| --- | -------------------------------------------------- | ----- | ---------- | ---------- |
| E0  | Direct analysis of the input files for this report | ...   | n/a        | none       |
