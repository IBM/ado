# Project Roadmap

## 📅 Overview

The **ado** roadmap outlines the planned direction and milestones for the next
major versions of the project. This is a living document that will be updated
regularly as the project evolves.

## 🚀 Key Goals

- **Production Ready**: Ensure the project robustly executes all core operators,
  actuators and examples.
- **Performant and Human-Centred CLI**: A CLI that is responsive and provides
  helpful feedback on error
- **Seamless Scaling**: Make it easy to scale from single-person working on
  their laptop, to a distributed team on executing on remote infrastructure.
- **Community & Ecosystem Development**: Respond to user needs and empower
  developers to extend `ado`

---

## 📆 Milestones

### **Q3 2025**

#### Version 1.0.0 (Initial Release)

- Complete core feature set
- Production actuator for fine-tuning performance measurement
- Reference actuators for inference performance measurement and model evaluation
- Focus on bug fixing, stability, and documentation
- Initial performance optimizations

### **Q4 2025**

- Property grouping and conditional properties
- Actuator and operator for building and applying predictive performance models
- REST API via `ado serve`

### **Q1 2026**

- Improved features for interacting (creating and running on) remote ray
  clusters
  - `ado create operation --target=$REMOTE_CLUSTER` can spin up a RayCluster,
    based on selected actuators needs, install code, start operation etc.
- Database performance improvements
- Agentic extensions - use natural language to create spaces and operations.

---

## 💬 How You Can Help

- **Contribute**: Submit pull requests for new features, bug fixes, or
  documentation improvements.
- **Open Issues**: Report bugs, request features, or provide feedback.
- **Spread the Word**: Share the project with others who could benefit
