---
id: 0001
title: Establish Physical AI Book Constitution
stage: constitution
date: 2025-12-15
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: 003-textbook-generation
user: Sheheryar
command: /sp.constitution
labels: ["constitution", "physical-ai", "robotics", "governance"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:
  - null
---

## Prompt

/sp.constitution
Unified Constitution for the Physical AI & Humanoid Robotics Book Project
1. Project Identity
Project Name: Physical AI & Humanoid Robotics
Format: AI-authored technical book
Platform: Docusaurus → GitHub Pages
Authoring System: Claude Code + Spec-Kit Plus
Audience: Advanced AI students, robotics engineers, and researchers transitioning from digital AI to embodied intelligence
Core Thesis:

Intelligence reaches its full potential only when it is embodied. Physical AI bridges the digital brain and the physical body, enabling machines to perceive, reason, and act in the real world.


2. Purpose & Goals
This project exists to:


Teach Physical AI and Embodied Intelligence from first principles to deployment.


Enable students to design, simulate, and control humanoid robots.


Integrate ROS 2, Gazebo, Unity, NVIDIA Isaac, and LLMs into a unified system.


Demonstrate Vision-Language-Action (VLA) pipelines for autonomous humanoids.


Serve as a capstone-quality reference usable in both academia and industry.


Success is defined by:


A complete, deployable book website


Clear conceptual explanations + runnable examples


A coherent capstone narrative culminating in an autonomous humanoid robot



3. Architectural Principles (Non-Negotiable)
3.1 Spec-Driven Development


Every chapter, module, and code example must follow an explicit spec


Specs define:


Learning objectives


Inputs / outputs


Assumptions


Constraints




No undocumented features or unexplained abstractions


3.2 Physical-First AI


Prefer embodied, sensor-driven intelligence over abstract AI


Every AI concept must be grounded in:


Sensors


Actuators


Physics


Environment interaction




3.3 Simulation-to-Real Mindset


Assume all systems will eventually run on real hardware


Emphasize:


Determinism


Latency awareness


Noise, uncertainty, and failure modes





4. Pedagogical Style Rules
Claude Code MUST:


Explain cot intent


Generate plans


Interface with deterministic ROS actions




The book MUST clearly separate:


Deterministic control (ROS, Nav2, controllers)


Probabilistic reasoning (LLMs, perception models)


No anthropomorphism without technical grounding.

8. Capstone Constitution
The final capstone MUST demonstrate:


Voice input → intent understanding


LLM-based task decomposition


ROS 2 action planning


Navigation with obstacle avoidance


Vision-based object recognition


Physical manipulation (simulated)


End-to-end autonomy in simulation


If any link is missing, the capstone is incomplete.

9. Quality Bar
Claude Code must continuously ask:

"Would this explanation allow a motivated engineer to reproduce the system without guessing?"

If the answer is no, the content must be revised.

10. Deployment & Maintenance


Content must be compatible with Docusaurus Markdown


Navigation must mirror the learning progression


No broken links or orphaned pages


Designed for long-term evolution of Physical AI tooling



11. Final Authority Clause
If ambiguity arises:


Favor clarity over cleverness


Favor engineering truth over narrative elegance


Favor physical reality over abstract optimization


This constitution overrides stylistic preferences and governs all content generation for the project.

## Response snapshot

Created constitution v1.0.0 for Physical AI & Humanoid Robotics Book Project with:

**Core Structure**:
- Project Identity: AI-authored technical book on Physical AI using Docusaurus + GitHub Pages
- Target Audience: Advanced AI students, robotics engineers, researchers

**3 Core Principles Established**:
1. Spec-Driven Development - Every chapter/module/example follows explicit specs
2. Physical-First AI - All AI concepts grounded in sensors, actuators, physics
3. Simulation-to-Real Mindset - Code emphasizes determinism, latency, noise handling

**Additional Governance Sections**:
- Pedagogical Style Rules (deterministic vs probabilistic separation)
- Capstone Constitution (8-component end-to-end humanoid system)
- Quality Bar (reproducibility test for all content)
- Deployment & Maintenance (Docusaurus compatibility requirements)
- Final Authority Clause (clarity > cleverness, truth > elegance, physical > abstract)

**Templates Validated**: plan-template.md, spec-template.md, tasks-template.md all compatible.

## Outcome

- Impact: Established foundational governance document for entire book project
- Tests: N/A (governance document)
- Files: .specify/memory/constitution.md created (v1.0.0)
- Next prompts: Generate chapter-level specs, create Docusaurus config, define book structure
- Reflection: Constitution successfully captures all user requirements with testable, declarative principles

## Evaluation notes (flywheel)

- Failure modes observed: None - user provided comprehensive input
- Graders run and results (PASS/FAIL): PASS - all placeholders filled, no ambiguity
- Prompt variant (if applicable): Standard /sp.constitution command
- Next experiment (smallest change to try): Monitor if constitution gates are effective during /sp.plan runs
