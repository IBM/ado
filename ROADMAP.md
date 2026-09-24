# Project Roadmap

## 📅 Overview

The **ado** roadmap outlines the planned direction and milestones for the next
major versions of the project. This is a living document that will be updated
regularly as the project evolves.

## 🚀 Key Goals

- **Production Ready**: Ensure all core operators, actuators and examples can be
  executed robustly
- **Performant and Human-Centred CLI**: A CLI that is responsive and provides
  helpful feedback on error
- **Agent-Driven Automation**: Provide a structured environment for AI agents to
  seamlessly assist and automate research workflows.
- **Seamless Scaling**: Make it easy to scale from single-person working on
  their laptop, to a distributed team on executing on remote infrastructure.
- **Community & Ecosystem Development**: Respond to user needs and empower
  developers to extend `ado`

---

## Upcoming Releases

### **September 2026** Version 2.2.0

Version 2.2.0 will be a maintenance release. It will address a variety of issues
found in day-to-day use, along with updates to the bundled plugins and
documentation. Some highlights will include:

- **Clearer provenance**:
  - Store the experiments and actuators an operation used, including minor and
    patch version differences.
  - Record the Ray job submission id so logs can be found and the job interacted
    with.
- **CLI cleanup**:
  - Delete an operation's datacontainers with the operation
  - Make samplestore measurements and space properties easier to inspect
- **Improved performance**:
  - Load and dump YAML with the C implementation when it is available,
  - Remove performance issues when working spaces with large discrete domains
    (via ado get space etc.). This includes waiting O(10)s for operations to
    complete.
- **Plugin updates**: Fixes and small capability updates across the plugins

### **October 2026** Version 2.3.0

Version 2.3.0 will focus on increasing the capabilities of operator plugins.
We've noticed that many scripts created for results analysis can't be expressed
as operators and hence can't be easily distributed for reuse and fall out of the
provenance system.

- **Multi-input, multi-resource-type operators**: Enable operator functions to
  take more that a single DiscoverySpace and also to process resources other
  than a DiscoverySpace e.g. operation, datacontainer
- **Operator Packaging**: Enable bundling operators specific to a given
  experiment
  - _See which operators process outputs of e.g. `solve_mip`_
  - _Enable operators to flag that they require certain output metrics to work_
- **Batch Operator Application**: Provide mechanisms to simplify executing
  operators multiple times

### **November 2026** Version 2.4.0

Currently `ado` only supports cartesian product spaces (full-factorial designs).
It also only supports scalar dimensions (scalar experiment input types).

Version 2.4.0 will expand the type of configuration spaces that can be explored.
This also means richer experiment definitions (more types, input relationships)

- **Constraints**: e.g. Allow the value of one property to be conditional on
  - _Allows fractional factorial designs_
  - _Enables experiments to forbid certain combinations of parameter values_
- **Treatments**: Allow a space to specify list of explicit points to explore,
  giving maximum flexibility in experiment design.
  - _Full: Specify the exact entities to sample_
  - _Partial: Set specific value pairs for some dimensions, while allow full
    cartesian product for others_
- **Vector Properties**: Allow easily specifying a dimension that expands to a
  vector space
  - _Currently, you would have to specify each dimension separately_
- **Property Groups**: Allow grouping properties/dimensions in experiments
  - _Enables an experiment be passed a nested dictionary of their parameters
    rather than flat dictionary**
  - _This will the need to group input parameters after being passed them**

## 📆 Recent Milestones

### **August 2026**: [Version 2.1.0](https://github.com/IBM/ado/releases#release-2.1.0)

As we use AI agents to drive research via ado more we've noticed we're creating
far more spaces, operations etc., including many failed experiments and trials
which we don't need to keep. We're also inundated with agent reports and
analysis scripts piling up in our filesystems.

In this release we were adding some features to address these issues:

- **New document resource type** for storing agent reports, plans etc.
  - _Store reports written for operations or spaces so they can be accessed by
    collaborators_
  - _Associate research plans and todos with projects/contexts_
- **Expanded operator interface** allowing operations on any resource types, in
  any number and combination
  - _Allow easily package analysis scripts for a project as an operator bundle
    that can be distributed_
  - _Leverage ado provenance and storage for the data produced by these scripts_
- **Agent skills for project maintenance**
  - Improve ability of Agents to correctly version plugins, manage their
    life-cycle and identify versioning related issues
  - Improve ability of Agents to recognise and delete superseded resources,
    failed and test operations, unused spaces etc. functionality

### **July 2026**: [Version 2.0.0](https://github.com/IBM/ado/releases#release-2.0.0)

In this release we are making a number of breaking changes in order to address
known issues we've encountered since 1.0 and provide a stable platform

- **Refactoring of the CLI** to make it more intuitive for humans and agents
  - _Include ability to get common stats via `ado get`_
  - _Simplified `ado show` subcommand structure_
  - _Updated naming and harmonized functions_
- **Increased performance of stats** **queries**
- **New StandardActuator baseclass**
  - _reduces amount of custom code_
  - _enables synchronous/non-ray execution patterns_
- **Enhanced plugin versioning and provenance**
  - _Improve ability of Agents to correctly version plugins, manage their
    life-cycle and identify versioning related issues_
  - _Track plugin versions used in resources_

---

## 💬 How You Can Help

- **Contribute**: Submit pull requests for new features, bug fixes, or
  documentation improvements.
- **Open Issues**: Report bugs, request features, or provide feedback.
- **Spread the Word**: Share the project with others who could benefit
