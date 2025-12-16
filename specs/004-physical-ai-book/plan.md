# Implementation Plan: Physical AI & Humanoid Robotics Book

**Branch**: `004-physical-ai-book` | **Date**: 2025-12-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-physical-ai-book/spec.md`

## Summary

Build a comprehensive, AI-authored technical book teaching Physical AI and Humanoid Robotics from first principles to deployment. The book will be deployed as a Docusaurus static site, integrating ROS 2, Gazebo Sim, Ollama/LLMs, and culminating in an 8-component autonomous humanoid capstone project.

**Key Technical Decisions** (from research):
- **Robot Model**: Custom 21-DOF URDF humanoid
- **Simulation**: Gazebo Sim (primary), Unity (secondary visual validation)
- **Control**: 70% Classical / 30% RL (conceptual + Isaac Gym hands-on)
- **LLM**: Stratified adapter pattern with Ollama as default provider

## Technical Context

**Language/Version**: Python 3.10+, C++17, TypeScript 5.x (Docusaurus)
**Primary Dependencies**: ROS 2 Humble, Gazebo Sim (Garden/Harmonic), Docusaurus 3.x, Ollama, stable-baselines3
**Storage**: Static Markdown files with YAML frontmatter, Git-based version control
**Testing**: pytest for Python, gtest for C++, Docusaurus build validation
**Target Platform**: Ubuntu 22.04 (Docker available for Windows/macOS)
**Project Type**: Documentation site with embedded runnable examples
**Performance Goals**: Site builds in <60s, code examples execute in <5min setup
**Constraints**: Offline-capable (Ollama default), no mandatory cloud API keys
**Scale/Scope**: 8 chapters, ~40 hours reader completion time, 8 capstone components

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. Spec-Driven Development** | PASS | Every chapter has explicit learning objectives, inputs/outputs defined in data-model.md |
| **II. Physical-First AI** | PASS | All AI concepts grounded in sensors/actuators via ROS 2 interfaces |
| **III. Simulation-to-Real Mindset** | PASS | Determinism via Gazebo Sim, latency documented in contracts, noise models specified |
| **Pedagogical Style Rules** | PASS | Deterministic control separated from probabilistic reasoning (70/30 split) |
| **Capstone Constitution** | PASS | All 8 components defined in ros2-interfaces.yaml with integration tests |
| **Quality Bar** | PASS | Reproducibility target: 5-min setup, verification script provided |
| **Deployment & Maintenance** | PASS | Docusaurus with GitHub Pages, version-pinned dependencies |

**Post-Design Re-Check**: PASS - All principles satisfied with documented evidence.

## Project Structure

### Documentation (this feature)

```text
specs/004-physical-ai-book/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research findings
├── data-model.md        # Entity definitions
├── quickstart.md        # Setup guide
├── contracts/
│   ├── action-schema.yaml    # LLM action contract
│   └── ros2-interfaces.yaml  # ROS 2 interface definitions
├── checklists/
│   └── requirements.md  # Validation checklist
└── tasks.md             # Phase 2 output (/sp.tasks command)
```

### Source Code (repository root)

```text
physical-ai-book/
├── docs/                    # Markdown lesson content
│   ├── intro.md
│   ├── chapter-01-ros2-fundamentals/
│   │   ├── _category_.json
│   │   ├── lesson-01-introduction.md
│   │   ├── lesson-02-nodes-topics.md
│   │   └── lesson-03-services-actions.md
│   ├── chapter-02-simulation/
│   ├── chapter-03-perception/
│   ├── chapter-04-navigation/
│   ├── chapter-05-manipulation/
│   ├── chapter-06-llm-integration/
│   ├── chapter-07-vla-pipelines/
│   └── chapter-08-capstone/
│
├── static/
│   ├── img/                 # Images and diagrams
│   └── models/
│       └── humanoid/        # URDF/SDF robot model
│
├── src/
│   ├── components/          # Custom Docusaurus components
│   ├── pages/               # Custom pages
│   └── examples/            # Runnable code examples
│       ├── ros2/            # ROS 2 Python/C++ examples
│       │   ├── ch01/
│       │   ├── ch02/
│       │   └── ...
│       ├── llm/             # LLM integration code
│       │   ├── core/
│       │   ├── providers/
│       │   ├── middleware/
│       │   └── robotics/
│       └── capstone/        # Complete capstone code
│
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── capstone/            # Capstone validation tests
│
├── scripts/
│   ├── verify-setup.sh      # Installation verification
│   └── build-examples.sh    # Build all code examples
│
├── docusaurus.config.js     # Site configuration
├── sidebars.js              # Navigation structure
├── package.json             # Node.js dependencies
├── docker-compose.yml       # Development environment
└── Dockerfile               # ROS 2 + Gazebo image
```

**Structure Decision**: Documentation site with embedded examples. Docusaurus handles content rendering, separate `src/examples/` directory for runnable code organized by chapter.

## Chapter Architecture

### Chapter Flow

```
Chapter 1: ROS 2 Fundamentals (Foundation)
    ↓
Chapter 2: Simulation with Gazebo Sim
    ↓
Chapter 3: Perception & Vision ←──────────┐
    ↓                                      │
Chapter 4: Navigation & Planning           │ Parallel tracks
    ↓                                      │ merge at capstone
Chapter 5: Manipulation & Control ←────────┘
    ↓
Chapter 6: LLM Integration
    ↓
Chapter 7: Vision-Language-Action Pipelines
    ↓
Chapter 8: Capstone Integration
```

### Chapter Details

| Chapter | Topics | Key Outputs | Components |
|---------|--------|-------------|------------|
| **1. ROS 2 Fundamentals** | Nodes, Topics, Services, Actions | Working pub/sub, action client | - |
| **2. Simulation** | Gazebo Sim, URDF, Physics | Humanoid in simulation | - |
| **3. Perception** | Camera, LIDAR, Object Detection | Vision pipeline | vision |
| **4. Navigation** | SLAM, Path Planning, Nav2 | Autonomous navigation | navigation |
| **5. Manipulation** | IK, Motion Planning, MoveIt 2 | Pick and place | manipulation |
| **6. LLM Integration** | Provider abstraction, Action schemas | LLM planner | voice_input, intent_understanding, llm_task_decomposition |
| **7. VLA Pipelines** | End-to-end learning, VLA models | Working VLA | ros_action_planning |
| **8. Capstone** | System integration, End-to-end | Autonomous humanoid | integration |

## Capstone Component Mapping

| Component | Chapter | ROS 2 Interfaces | Integration Test |
|-----------|---------|------------------|------------------|
| voice_input | Ch6 | /voice/audio_raw, /voice/transcript | test_voice_input.py |
| intent_understanding | Ch6 | /intent/parse | test_intent_parse.py |
| llm_task_decomposition | Ch6 | /llm/decompose_task, /llm/plan | test_llm_decompose.py |
| ros_action_planning | Ch7 | /execute_task_plan | test_action_planning.py |
| navigation | Ch4 | /navigate_to_pose, /cmd_vel | test_navigation.py |
| vision | Ch3 | /vision/detections, /vision/find_object | test_vision.py |
| manipulation | Ch5 | /grasp_object, /place_object | test_manipulation.py |
| integration | Ch8 | Full pipeline | test_capstone_e2e.py |

## Complexity Tracking

> No constitution violations requiring justification.

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Custom URDF vs Pre-built | Custom | Full control, optimized for education, no licensing |
| Gazebo vs Unity | Gazebo primary | ROS 2 native integration, deterministic |
| 70/30 Control split | Hybrid | Foundation first, RL exposure without depth |
| Ollama default | Local-first | Offline examples, no API key barriers |

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gazebo API changes | Low | High | Pin to LTS, Docker images |
| Ollama model quality | Medium | Medium | Test with multiple models |
| Isaac Gym deprecation | Low | Medium | Abstract RL interface |
| Reader hardware limits | Medium | Medium | Docker + cloud options |
| ROS 2 version fragmentation | Low | High | Document version matrix |

## Artifacts Generated

| Artifact | Path | Status |
|----------|------|--------|
| Research findings | [research.md](./research.md) | Complete |
| Data model | [data-model.md](./data-model.md) | Complete |
| Action schema contract | [contracts/action-schema.yaml](./contracts/action-schema.yaml) | Complete |
| ROS 2 interfaces | [contracts/ros2-interfaces.yaml](./contracts/ros2-interfaces.yaml) | Complete |
| Quickstart guide | [quickstart.md](./quickstart.md) | Complete |

## Next Steps

1. **Run `/sp.tasks`** to generate detailed implementation tasks
2. **Create humanoid URDF model** (Task T001)
3. **Initialize Docusaurus project** (Task T002)
4. **Write Chapter 1 content** (first content milestone)
5. **Implement LLM abstraction layer** (Task Txx for Ch6)

## Transition Clause

This plan authorizes execution. All open questions from specification have been resolved in research.md. Implementation SHALL proceed via `/sp.tasks` to generate actionable task list.
