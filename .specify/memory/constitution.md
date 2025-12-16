<!--
  SYNC IMPACT REPORT
  ==================
  Version Change: 0.0.0 → 1.0.0 (MAJOR - initial constitution establishment)

  Modified Principles: N/A (first version)

  Added Sections:
  - I. Spec-Driven Development
  - II. Physical-First AI
  - III. Simulation-to-Real Mindset
  - Pedagogical Style Rules
  - Capstone Constitution
  - Quality Bar
  - Deployment & Maintenance
  - Final Authority Clause

  Removed Sections: N/A (template placeholders replaced)

  Templates Requiring Updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section aligned)
  - .specify/templates/spec-template.md: ✅ Compatible (user scenarios format preserved)
  - .specify/templates/tasks-template.md: ✅ Compatible (TDD workflow preserved)

  Follow-up TODOs: None
-->

# Physical AI & Humanoid Robotics Constitution

## Project Identity

**Project Name**: Physical AI & Humanoid Robotics
**Format**: AI-authored technical book
**Platform**: Docusaurus → GitHub Pages
**Authoring System**: Claude Code + Spec-Kit Plus
**Audience**: Advanced AI students, robotics engineers, and researchers transitioning from digital AI to embodied intelligence

**Core Thesis**:
> Intelligence reaches its full potential only when it is embodied. Physical AI bridges the digital brain and the physical body, enabling machines to perceive, reason, and act in the real world.

## Purpose & Goals

This project exists to:

1. Teach Physical AI and Embodied Intelligence from first principles to deployment
2. Enable students to design, simulate, and control humanoid robots
3. Integrate ROS 2, Gazebo, Unity, NVIDIA Isaac, and LLMs into a unified system
4. Demonstrate Vision-Language-Action (VLA) pipelines for autonomous humanoids
5. Serve as a capstone-quality reference usable in both academia and industry

**Success is defined by**:
- A complete, deployable book website
- Clear conceptual explanations + runnable examples
- A coherent capstone narrative culminating in an autonomous humanoid robot

## Core Principles

### I. Spec-Driven Development

Every chapter, module, and code example MUST follow an explicit specification.

**Non-Negotiable Rules**:
- Specs MUST define: learning objectives, inputs/outputs, assumptions, and constraints
- No undocumented features or unexplained abstractions
- Every feature starts with a spec before implementation
- Specs are the single source of truth for what gets built

**Rationale**: Spec-driven development ensures consistency, testability, and reproducibility across all book content. Without explicit specs, content drifts and students cannot reproduce results.

### II. Physical-First AI

Prefer embodied, sensor-driven intelligence over abstract AI.

**Non-Negotiable Rules**:
- Every AI concept MUST be grounded in: sensors, actuators, physics, and environment interaction
- Abstract algorithms MUST be connected to physical manifestation
- Digital-only AI concepts MUST include a bridge to physical implementation
- Simulation environments MUST model realistic physical constraints

**Rationale**: The book's thesis is that intelligence requires embodiment. Teaching disembodied AI concepts would contradict the core thesis and fail to prepare students for real robotics work.

### III. Simulation-to-Real Mindset

Assume all systems will eventually run on real hardware.

**Non-Negotiable Rules**:
- All code MUST emphasize determinism where required
- Latency MUST be explicitly considered and documented
- Noise, uncertainty, and failure modes MUST be addressed
- Sim-to-real transfer considerations MUST be documented for all simulated examples
- No "simulation-only" shortcuts that cannot transfer to hardware

**Rationale**: Students learn bad habits when simulation ignores real-world constraints. This principle ensures students are prepared for deployment.

## Pedagogical Style Rules

Claude Code and all content authors MUST:

1. Explain chain-of-thought intent for all reasoning
2. Generate explicit plans before complex implementations
3. Interface with deterministic ROS actions using documented patterns

The book MUST clearly separate:
- **Deterministic control**: ROS, Nav2, controllers (predictable, verifiable)
- **Probabilistic reasoning**: LLMs, perception models (uncertain, requires calibration)

**No anthropomorphism without technical grounding**. Phrases like "the robot thinks" MUST be accompanied by technical explanation of the underlying computation.

## Capstone Constitution

The final capstone MUST demonstrate an end-to-end autonomous humanoid system including:

| Component | Requirement |
|-----------|-------------|
| Voice Input | Speech recognition → text |
| Intent Understanding | NLU to parse user commands |
| LLM Task Decomposition | Break high-level goals into subtasks |
| ROS 2 Action Planning | Convert subtasks to ROS 2 action sequences |
| Navigation | Autonomous navigation with obstacle avoidance |
| Vision | Object recognition and scene understanding |
| Manipulation | Physical manipulation (simulated) |
| Integration | End-to-end autonomy in simulation |

**If any component is missing, the capstone is incomplete.**

The capstone serves as the ultimate validation that all book content integrates into a working system.

## Quality Bar

Claude Code and all authors MUST continuously ask:

> "Would this explanation allow a motivated engineer to reproduce the system without guessing?"

**Quality Gates**:
- If the answer is "no" → content MUST be revised
- Code examples MUST be complete and runnable
- Dependencies MUST be explicitly listed
- Expected outputs MUST be documented
- Error cases MUST be addressed

## Deployment & Maintenance

**Platform Requirements**:
- Content MUST be compatible with Docusaurus Markdown
- Navigation MUST mirror the learning progression
- No broken links or orphaned pages
- Designed for long-term evolution of Physical AI tooling

**Content Standards**:
- All code examples MUST specify versions and dependencies
- External links MUST be validated
- Images MUST have alt text and be optimized for web

## Governance

### Amendment Process

1. Proposed amendments MUST be documented with rationale
2. Amendments MUST be reviewed against the Final Authority Clause
3. Changes MUST include migration plan for affected content
4. Version MUST be incremented according to semantic versioning

### Versioning Policy

- **MAJOR**: Principle removal, redefinition, or backward-incompatible governance change
- **MINOR**: New principle added, section materially expanded
- **PATCH**: Clarifications, wording improvements, non-semantic changes

### Compliance Review

- All content MUST be verified against this constitution before publishing
- The Constitution Check in plan templates MUST reference these principles
- Violations MUST be justified in the Complexity Tracking section

## Final Authority Clause

If ambiguity arises in any content decision:

1. **Favor clarity over cleverness** - Simple explanations that work beat elegant ones that confuse
2. **Favor engineering truth over narrative elegance** - Accurate technical content beats smooth prose
3. **Favor physical reality over abstract optimization** - Real-world constraints beat theoretical optimums

**This constitution overrides stylistic preferences and governs all content generation for the project.**

**Version**: 1.0.0 | **Ratified**: 2025-12-15 | **Last Amended**: 2025-12-15
