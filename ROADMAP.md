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

## 📆 Milestones

### **July 2026**: Version 2.0.0 (Released)

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

### **August 2026**: Version 2.1.0

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

---

## 💬 How You Can Help

- **Contribute**: Submit pull requests for new features, bug fixes, or
  documentation improvements.
- **Open Issues**: Report bugs, request features, or provide feedback.
- **Spread the Word**: Share the project with others who could benefit
