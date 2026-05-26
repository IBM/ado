# Project Roadmap

## 📅 Overview

The **ado** roadmap outlines the planned direction and milestones for the next
major versions of the project. This is a living document that will be updated
regularly as the project evolves.

## 🚀 Key Goals

- **Production Ready**: Ensure all core operators, actuators and examples can
be executed robustly
- **Performant and Human-Centred CLI**: A CLI that is responsive and provides
  helpful feedback on error
- **Agent-Driven Automation**: Provide a structured environment for AI agents
  to seamlessly assist and automate research workflows.
- **Seamless Scaling**: Make it easy to scale from single-person working on
  their laptop, to a distributed team on executing on remote infrastructure.
- **Community & Ecosystem Development**: Respond to user needs and empower
  developers to extend `ado`

---

## 📆 Milestones

### **June 2026**: Version 2.0.0

In this release we are making a number of breaking changes in order to
address known issues we've encountered since 1.0 and provide a stable platform

- Refactoring of the CLI to make it more intuitive for humans and agents
  - Include ability to get common stats via `ado get`
  - Simplified `ado show` subcommand structure
  - Updated naming and harmonized functions
- Increase performance of stats queries via rewrite of sql sample store
- Ability to upgrade the schema used for sample store data
- Update actuators to use entrypoints
- New StandardActuator baseclass
  - reduces amount of custom code
  - enables synchronous/non-ray execution patterns

### **July/August 2026**: Version 2.1.0

As we use AI agents to drive research via ado more we've noticed we're creating
far more spaces, operations etc.,
including many failed experiments and trials which we don't need to keep.
We're also inundated with agent reports and analysis scripts piling up in
our filesystems.

In this release we were adding some features to address these issues:

- New document resource type for storing agent reports, plans etc.
  - Store reports written for operations or spaces so they can be accessed by collaborators
  - Associate research plans and todos with projects/contexts
- Expanded operator interface allowing operations on any resource types, in any
number and combination
  - Allow easily package analysis scripts for a project as an operator bundle that
    can be distributed
  - Leverage ado provenance and storage for the data produced by these scripts
- Enhanced actuator,operator and experiment versioning support including skills
  - Improve ability of Agents to correctly version plugins, manage their life-cycle
    and identify versioning related issues
- Agent skill for project maintenance, coupled with enhanced delete functionality

---

## 💬 How You Can Help

- **Contribute**: Submit pull requests for new features, bug fixes, or
  documentation improvements.
- **Open Issues**: Report bugs, request features, or provide feedback.
- **Spread the Word**: Share the project with others who could benefit
