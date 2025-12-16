# Tasks: Physical AI & Humanoid Robotics Book

**Input**: Design documents from `/specs/004-physical-ai-book/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included for critical integration points and capstone validation.

**Organization**: Tasks are grouped by user story (P1-P5) to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions

```text
physical-ai-book/
├── docs/                    # Markdown lesson content (chapters)
├── static/models/           # URDF/SDF robot models
├── src/examples/            # Runnable code examples
├── tests/                   # Test files
├── scripts/                 # Setup and build scripts
└── [config files]           # Docusaurus configuration
```

---

## Phase 1: Setup (Project Infrastructure)

**Purpose**: Repository initialization and Docusaurus setup

- [x] T001 Create project directory structure per plan.md in physical-ai-book/
- [x] T002 Initialize Git repository with README.md and LICENSE (MIT) in physical-ai-book/
- [x] T003 Initialize Docusaurus 3.x project with `npx create-docusaurus@latest` in physical-ai-book/
- [x] T004 [P] Configure docusaurus.config.js with project metadata and GitHub Pages settings
- [x] T005 [P] Create sidebars.js with 8-chapter navigation structure
- [x] T006 [P] Create docs/intro.md with book overview and learning path
- [x] T007 Create package.json with all dependencies (Docusaurus, Prism, MDX)
- [x] T008 [P] Create Dockerfile for ROS 2 Humble + Gazebo Sim development environment
- [x] T009 [P] Create docker-compose.yml for easy container management
- [x] T010 Create scripts/verify-setup.sh for installation verification
- [x] T011 [P] Create .github/workflows/deploy.yml for GitHub Pages CI/CD

**Checkpoint**: `npm start` runs locally, site renders without errors ✅

---

## Phase 2: Foundational (Shared Assets & Templates)

**Purpose**: Core assets that ALL user stories depend on

**⚠️ CRITICAL**: No chapter content work can begin until this phase is complete

- [x] T012 Create custom 21-DOF humanoid URDF model in static/models/humanoid/humanoid.urdf
- [x] T013 [P] Create humanoid SDF wrapper for Gazebo in static/models/humanoid/humanoid.sdf
- [x] T014 [P] Create humanoid mesh files (visual + collision) in static/models/humanoid/meshes/
- [x] T015 Create chapter template markdown in docs/_templates/lesson-template.md
- [x] T016 [P] Create code example template in docs/_templates/code-example-template.md
- [x] T017 [P] Create exercise template in docs/_templates/exercise-template.md
- [x] T018 Create src/examples/ directory structure per plan.md (ros2/, llm/, capstone/)
- [x] T019 [P] Create base Python package structure in src/examples/ros2/common/
- [x] T020 [P] Create LLM abstraction base classes in src/examples/llm/core/provider.py
- [x] T021 Create static/img/ directory with placeholder architecture diagrams

**Checkpoint**: Foundation ready - URDF loads in Gazebo, templates usable, base code exists

---

## Phase 3: User Story 1 - ROS 2 Fundamentals (Priority: P1) 🎯 MVP

**Goal**: Reader can create ROS 2 nodes, pub/sub, services, and actions

**Independent Test**: Run multi-node communication system in simulation with deterministic message passing

### Chapter Content for US1

- [x] T022 [US1] Create docs/chapter-01-ros2-fundamentals/_category_.json with chapter metadata
- [x] T023 [US1] Write docs/chapter-01-ros2-fundamentals/lesson-01-introduction.md (What is ROS 2, why humanoid robotics)
- [x] T024 [US1] Write docs/chapter-01-ros2-fundamentals/lesson-02-installation.md (Ubuntu/Docker setup)
- [x] T025 [US1] Write docs/chapter-01-ros2-fundamentals/lesson-03-nodes-topics.md (pub/sub patterns)
- [x] T026 [US1] Write docs/chapter-01-ros2-fundamentals/lesson-04-services-actions.md (request/response, long-running tasks)
- [x] T027 [US1] Write docs/chapter-01-ros2-fundamentals/lesson-05-urdf-basics.md (robot description)
- [x] T028 [US1] Write docs/chapter-01-ros2-fundamentals/exercises.md with 3-5 hands-on exercises

### Code Examples for US1

- [x] T029 [P] [US1] Create src/examples/ros2/ch01/simple_publisher.py (topic publishing)
- [x] T030 [P] [US1] Create src/examples/ros2/ch01/simple_subscriber.py (topic subscribing)
- [x] T031 [P] [US1] Create src/examples/ros2/ch01/service_server.py (service example)
- [x] T032 [P] [US1] Create src/examples/ros2/ch01/action_client.py (action example)
- [x] T033 [US1] Create src/examples/ros2/ch01/multi_node_system.py (integrated demo)

### Tests for US1

- [x] T034 [US1] Create tests/integration/test_ch01_ros2_basics.py validating node communication

**Checkpoint**: Reader completes Ch1, creates working ROS 2 nodes (MVP deliverable) ✅

---

## Phase 4: User Story 2 - Humanoid Simulation (Priority: P2)

**Goal**: Reader can load humanoid in Gazebo with realistic physics and sensors

**Independent Test**: Humanoid model loads, responds to commands, sensors publish data

### Chapter Content for US2

- [x] T035 [US2] Create docs/chapter-02-simulation/_category_.json with chapter metadata
- [x] T036 [US2] Write docs/chapter-02-simulation/lesson-01-gazebo-intro.md (Gazebo Sim overview)
- [x] T037 [US2] Write docs/chapter-02-simulation/lesson-02-urdf-deep-dive.md (joints, links, inertia)
- [x] T038 [US2] Write docs/chapter-02-simulation/lesson-03-physics-config.md (physics engines, parameters)
- [x] T039 [US2] Write docs/chapter-02-simulation/lesson-04-sensor-simulation.md (camera, LIDAR, IMU)
- [x] T040 [US2] Write docs/chapter-02-simulation/lesson-05-ros2-gazebo-bridge.md (gz_ros2_bridge)
- [x] T041 [US2] Write docs/chapter-02-simulation/exercises.md with simulation exercises

### Code Examples for US2

- [x] T042 [P] [US2] Create src/examples/ros2/ch02/launch_humanoid.py (Gazebo launch file)
- [x] T043 [P] [US2] Create src/examples/ros2/ch02/joint_controller.py (joint position control)
- [x] T044 [P] [US2] Create src/examples/ros2/ch02/sensor_reader.py (camera/LIDAR subscriber)
- [x] T045 [US2] Create src/examples/ros2/ch02/simple_locomotion.py (basic walking)

### Tests for US2

- [x] T046 [US2] Create tests/integration/test_ch02_simulation.py validating humanoid loads and sensors work

**Checkpoint**: Humanoid walks in Gazebo, sensors publish realistic data ✅

---

## Phase 5: User Story 3 - LLM Integration (Priority: P3)

**Goal**: Reader can interface LLM with ROS 2 via structured action schemas

**Independent Test**: Voice/text command → structured action plan → ROS 2 execution

**Dependencies**: Requires US1 (ROS 2) and US2 (simulation) for full demo

### Intermediate Chapters (Perception, Navigation, Manipulation)

**Note**: These chapters support US3+ and are grouped here for dependency order

#### Chapter 3: Perception & Vision

- [ ] T047 [US3] Create docs/chapter-03-perception/_category_.json
- [ ] T048 [US3] Write docs/chapter-03-perception/lesson-01-computer-vision.md (OpenCV, image processing)
- [ ] T049 [US3] Write docs/chapter-03-perception/lesson-02-object-detection.md (YOLO, detection models)
- [ ] T050 [US3] Write docs/chapter-03-perception/lesson-03-isaac-perception.md (Isaac ROS pipelines)
- [ ] T051 [P] [US3] Create src/examples/ros2/ch03/vision_node.py (camera processing)
- [ ] T052 [P] [US3] Create src/examples/ros2/ch03/object_detector.py (detection publisher)

#### Chapter 4: Navigation

- [ ] T053 [US3] Create docs/chapter-04-navigation/_category_.json
- [ ] T054 [US3] Write docs/chapter-04-navigation/lesson-01-slam.md (mapping fundamentals)
- [ ] T055 [US3] Write docs/chapter-04-navigation/lesson-02-nav2.md (Nav2 stack)
- [ ] T056 [US3] Write docs/chapter-04-navigation/lesson-03-path-planning.md (A*, RRT)
- [ ] T057 [P] [US3] Create src/examples/ros2/ch04/nav2_launcher.py
- [ ] T058 [P] [US3] Create src/examples/ros2/ch04/waypoint_follower.py

#### Chapter 5: Manipulation

- [ ] T059 [US3] Create docs/chapter-05-manipulation/_category_.json
- [ ] T060 [US3] Write docs/chapter-05-manipulation/lesson-01-kinematics.md (FK, IK)
- [ ] T061 [US3] Write docs/chapter-05-manipulation/lesson-02-moveit2.md (motion planning)
- [ ] T062 [US3] Write docs/chapter-05-manipulation/lesson-03-grasping.md (pick and place)
- [ ] T063 [P] [US3] Create src/examples/ros2/ch05/ik_solver.py
- [ ] T064 [P] [US3] Create src/examples/ros2/ch05/grasp_action_server.py

#### Chapter 6: LLM Integration (Core US3)

- [ ] T065 [US3] Create docs/chapter-06-llm-integration/_category_.json
- [ ] T066 [US3] Write docs/chapter-06-llm-integration/lesson-01-llm-overview.md (LLMs for robotics)
- [ ] T067 [US3] Write docs/chapter-06-llm-integration/lesson-02-provider-abstraction.md (Ollama, OpenAI)
- [ ] T068 [US3] Write docs/chapter-06-llm-integration/lesson-03-action-schemas.md (structured output)
- [ ] T069 [US3] Write docs/chapter-06-llm-integration/lesson-04-voice-input.md (Whisper STT)
- [ ] T070 [US3] Write docs/chapter-06-llm-integration/lesson-05-safety-constraints.md (validation layer)
- [ ] T071 [US3] Write docs/chapter-06-llm-integration/exercises.md

### LLM Code Implementation for US3

- [ ] T072 [US3] Create src/examples/llm/core/schemas.py (ActionSchema, ActionResponse types)
- [ ] T073 [US3] Create src/examples/llm/providers/ollama_adapter.py (Ollama provider)
- [ ] T074 [P] [US3] Create src/examples/llm/providers/openai_adapter.py (OpenAI provider)
- [ ] T075 [US3] Create src/examples/llm/middleware/validator.py (JSON schema validation)
- [ ] T076 [P] [US3] Create src/examples/llm/middleware/rate_limiter.py (token bucket)
- [ ] T077 [US3] Create src/examples/llm/robotics/action_planner.py (LLM → ROS 2 actions)
- [ ] T078 [US3] Create src/examples/llm/robotics/safety_validator.py (bounds checking)

### Tests for US3

- [ ] T079 [US3] Create tests/integration/test_ch06_llm_integration.py validating LLM → action flow

**Checkpoint**: Voice command produces valid ROS 2 action sequence

---

## Phase 6: User Story 4 - VLA Pipelines (Priority: P4)

**Goal**: Reader implements Vision-Language-Action pipeline for image+language → motor commands

**Independent Test**: Image + instruction → robot manipulation in simulation

**Dependencies**: Requires US1-US3

### Chapter Content for US4

- [ ] T080 [US4] Create docs/chapter-07-vla-pipelines/_category_.json
- [ ] T081 [US4] Write docs/chapter-07-vla-pipelines/lesson-01-vla-overview.md (what is VLA)
- [ ] T082 [US4] Write docs/chapter-07-vla-pipelines/lesson-02-vision-encoding.md (image features)
- [ ] T083 [US4] Write docs/chapter-07-vla-pipelines/lesson-03-language-grounding.md (CLIP, embedding)
- [ ] T084 [US4] Write docs/chapter-07-vla-pipelines/lesson-04-action-generation.md (motor output)
- [ ] T085 [US4] Write docs/chapter-07-vla-pipelines/lesson-05-end-to-end.md (full pipeline)
- [ ] T086 [US4] Write docs/chapter-07-vla-pipelines/exercises.md

### Code Examples for US4

- [ ] T087 [P] [US4] Create src/examples/ros2/ch07/vla_node.py (VLA ROS 2 node)
- [ ] T088 [P] [US4] Create src/examples/ros2/ch07/vision_encoder.py (image encoding)
- [ ] T089 [US4] Create src/examples/ros2/ch07/action_decoder.py (motor command generation)
- [ ] T090 [US4] Create src/examples/ros2/ch07/vla_demo.py (complete pipeline)

### Tests for US4

- [ ] T091 [US4] Create tests/integration/test_ch07_vla.py validating image+text → action

**Checkpoint**: VLA pipeline produces motor commands from visual+language input

---

## Phase 7: User Story 5 - Capstone Integration (Priority: P5)

**Goal**: Reader builds autonomous humanoid integrating ALL 8 capstone components

**Independent Test**: "Fetch the red cup" → autonomous navigation, vision, grasp, return

**Dependencies**: Requires US1-US4 completion

### Chapter Content for US5

- [ ] T092 [US5] Create docs/chapter-08-capstone/_category_.json
- [ ] T093 [US5] Write docs/chapter-08-capstone/lesson-01-system-architecture.md (integration overview)
- [ ] T094 [US5] Write docs/chapter-08-capstone/lesson-02-component-integration.md (wiring 8 components)
- [ ] T095 [US5] Write docs/chapter-08-capstone/lesson-03-state-machine.md (task execution flow)
- [ ] T096 [US5] Write docs/chapter-08-capstone/lesson-04-error-handling.md (failure recovery)
- [ ] T097 [US5] Write docs/chapter-08-capstone/lesson-05-demo-walkthrough.md (step-by-step demo)
- [ ] T098 [US5] Write docs/chapter-08-capstone/exercises.md (capstone challenges)

### Capstone Code for US5

- [ ] T099 [US5] Create src/examples/capstone/voice_input.py (Whisper integration)
- [ ] T100 [US5] Create src/examples/capstone/intent_parser.py (NLU component)
- [ ] T101 [US5] Create src/examples/capstone/task_planner.py (LLM decomposition)
- [ ] T102 [US5] Create src/examples/capstone/action_executor.py (ROS 2 action client)
- [ ] T103 [US5] Create src/examples/capstone/navigation_controller.py (Nav2 interface)
- [ ] T104 [US5] Create src/examples/capstone/vision_controller.py (object detection)
- [ ] T105 [US5] Create src/examples/capstone/manipulation_controller.py (grasp execution)
- [ ] T106 [US5] Create src/examples/capstone/capstone_main.py (orchestrator node)
- [ ] T107 [US5] Create src/examples/capstone/launch/capstone.launch.py (full system launch)

### Capstone Tests for US5

- [ ] T108 [US5] Create tests/capstone/test_voice_input.py
- [ ] T109 [P] [US5] Create tests/capstone/test_intent_parse.py
- [ ] T110 [P] [US5] Create tests/capstone/test_llm_decompose.py
- [ ] T111 [P] [US5] Create tests/capstone/test_navigation.py
- [ ] T112 [P] [US5] Create tests/capstone/test_vision.py
- [ ] T113 [P] [US5] Create tests/capstone/test_manipulation.py
- [ ] T114 [US5] Create tests/capstone/test_capstone_e2e.py (full integration test)

**Checkpoint**: Complete autonomous humanoid demonstrates "fetch object" task

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Quality assurance, documentation, and deployment

- [ ] T115 [P] Create static/img/architecture-overview.png (system diagram)
- [ ] T116 [P] Create static/img/chapter-flow.png (learning path diagram)
- [ ] T117 [P] Create static/img/capstone-components.png (8-component diagram)
- [ ] T118 Update README.md with comprehensive project documentation
- [ ] T119 [P] Create CONTRIBUTING.md with contribution guidelines
- [ ] T120 Run markdownlint on all docs/**/*.md and fix issues
- [ ] T121 Validate all code examples execute successfully via scripts/build-examples.sh
- [ ] T122 Run full Docusaurus build and verify no broken links
- [ ] T123 Create compliance checklist verifying all FR-001 through FR-012 satisfied
- [ ] T124 Create learning outcome traceability matrix (chapter → outcomes)
- [ ] T125 Final deployment to GitHub Pages via GitHub Actions

**Checkpoint**: Production site deployed, all quality gates passed

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup ──────────────────┐
                                 ↓
Phase 2: Foundational ───────────┤
                                 ↓
┌────────────────────────────────┴───────────────────────────────┐
│                                                                │
│  Phase 3: US1 (P1) ─────→ Phase 4: US2 (P2) ─────→ ...        │
│       ↓ (MVP)                   ↓                              │
│  Can deploy                Can deploy                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                                 ↓
Phase 8: Polish ─────────────────┘
```

### User Story Dependencies

| Story | Depends On | Can Start After | Independently Testable |
|-------|------------|-----------------|------------------------|
| US1 (P1) | Phase 2 | Foundational complete | Yes - ROS 2 nodes work alone |
| US2 (P2) | US1 | US1 complete | Yes - Simulation works alone |
| US3 (P3) | US1, US2 | US2 complete | Yes - LLM actions testable in sim |
| US4 (P4) | US1-US3 | US3 complete | Yes - VLA pipeline testable |
| US5 (P5) | US1-US4 | US4 complete | Yes - Full capstone demo |

### Within Each User Story

1. Chapter metadata (_category_.json) first
2. Lesson content (lesson-*.md) in order
3. Code examples ([P] tasks can parallel)
4. Integration test
5. Exercises last

---

## Parallel Execution Examples

### Phase 1 Parallel Tasks

```bash
# These can run simultaneously:
T004 Configure docusaurus.config.js
T005 Create sidebars.js
T006 Create docs/intro.md
T008 Create Dockerfile
T009 Create docker-compose.yml
T011 Create .github/workflows/deploy.yml
```

### Phase 2 Parallel Tasks

```bash
# These can run simultaneously:
T013 Create humanoid SDF
T014 Create humanoid meshes
T016 Create code example template
T017 Create exercise template
T019 Create Python package structure
T020 Create LLM base classes
```

### User Story Code Examples (Parallel)

```bash
# US1 code examples (T029-T032 are parallel):
T029 simple_publisher.py
T030 simple_subscriber.py
T031 service_server.py
T032 action_client.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (~2 hours)
2. Complete Phase 2: Foundational (~4 hours)
3. Complete Phase 3: User Story 1 - ROS 2 Fundamentals (~8 hours)
4. **STOP and VALIDATE**: Test that reader can complete Ch1 exercises
5. Deploy to GitHub Pages (MVP!)

### Incremental Delivery

| Milestone | Stories | Value Delivered |
|-----------|---------|-----------------|
| MVP | US1 | Reader learns ROS 2 basics |
| Alpha | US1 + US2 | Reader can simulate humanoid |
| Beta | US1-US3 | Reader can use LLMs with robots |
| RC | US1-US4 | Reader can build VLA pipelines |
| Release | US1-US5 | Complete capstone experience |

### Estimated Effort

| Phase | Tasks | Estimated Hours |
|-------|-------|-----------------|
| Setup | T001-T011 | 4 |
| Foundational | T012-T021 | 8 |
| US1 (P1) | T022-T034 | 12 |
| US2 (P2) | T035-T046 | 10 |
| US3 (P3) | T047-T079 | 20 |
| US4 (P4) | T080-T091 | 12 |
| US5 (P5) | T092-T114 | 16 |
| Polish | T115-T125 | 8 |
| **Total** | **125 tasks** | **~90 hours** |

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [USx] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All code examples MUST include expected output in comments
- All exercises MUST have verifiable success criteria
