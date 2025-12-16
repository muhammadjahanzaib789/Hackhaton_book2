# Data Model: Physical AI & Humanoid Robotics Book

**Branch**: `004-physical-ai-book` | **Date**: 2025-12-15

## Overview

This document defines the key entities, their attributes, relationships, and validation rules for the Physical AI book content management system.

---

## Entity Definitions

### 1. Chapter

A self-contained learning unit within the book.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (e.g., "ch01", "ch02") |
| `title` | string | Yes | Chapter title |
| `slug` | string | Yes | URL-friendly identifier |
| `order` | integer | Yes | Position in book sequence |
| `learning_objectives` | string[] | Yes | What reader will learn |
| `prerequisites` | string[] | No | Required prior chapters |
| `estimated_time` | integer | Yes | Minutes to complete |
| `difficulty` | enum | Yes | "beginner", "intermediate", "advanced" |
| `content_path` | string | Yes | Path to markdown content |
| `exercises` | Exercise[] | Yes | Associated exercises |
| `code_examples` | CodeExample[] | Yes | Associated code examples |

**Validation Rules**:
- `learning_objectives` must have 3-7 items
- `estimated_time` must be between 30-180 minutes
- `prerequisites` must reference valid chapter IDs

### 2. CodeExample

Runnable code demonstrating a specific concept.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `chapter_id` | string | Yes | Parent chapter reference |
| `title` | string | Yes | Example title |
| `description` | string | Yes | What this example demonstrates |
| `language` | enum | Yes | "python", "cpp", "yaml", "bash" |
| `source_path` | string | Yes | Path to source file |
| `expected_output` | string | No | Expected console output |
| `dependencies` | Dependency[] | Yes | Required packages/versions |
| `ros_packages` | string[] | No | Required ROS 2 packages |
| `estimated_runtime` | integer | Yes | Seconds to execute |

**Validation Rules**:
- `source_path` must exist and be readable
- `estimated_runtime` must be under 300 seconds (5 min setup requirement)
- All `dependencies` must have pinned versions

### 3. Exercise

A task for readers to complete independently.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `chapter_id` | string | Yes | Parent chapter reference |
| `title` | string | Yes | Exercise title |
| `description` | string | Yes | What reader should accomplish |
| `difficulty` | enum | Yes | "easy", "medium", "hard" |
| `verification_type` | enum | Yes | "output_match", "behavior_check", "manual" |
| `verification_script` | string | No | Path to verification script |
| `hints` | string[] | No | Progressive hints |
| `solution_path` | string | No | Path to solution (hidden) |
| `estimated_time` | integer | Yes | Minutes to complete |

**Validation Rules**:
- `verification_type` "output_match" requires `verification_script`
- `hints` should have 1-3 items if provided
- `estimated_time` must be between 10-60 minutes

### 4. CapstoneComponent

One of the 8 required subsystems for the final capstone project.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Component identifier |
| `name` | string | Yes | Component name |
| `type` | enum | Yes | See component types below |
| `chapter_refs` | string[] | Yes | Chapters that teach this |
| `ros_interfaces` | ROSInterface[] | Yes | ROS 2 topics/services/actions |
| `integration_test` | string | Yes | Path to integration test |
| `dependencies` | string[] | No | Other components required |

**Component Types** (from Constitution):
- `voice_input` - Speech recognition → text
- `intent_understanding` - NLU to parse commands
- `llm_task_decomposition` - High-level goals → subtasks
- `ros_action_planning` - Subtasks → ROS 2 action sequences
- `navigation` - Autonomous navigation with obstacle avoidance
- `vision` - Object recognition and scene understanding
- `manipulation` - Physical manipulation (simulated)
- `integration` - End-to-end autonomy

### 5. Dependency

External package requirement.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Package name |
| `version` | string | Yes | Version constraint |
| `source` | enum | Yes | "pip", "apt", "ros2", "npm" |
| `optional` | boolean | No | Required for core functionality |
| `documentation_url` | string | No | Link to package docs |

### 6. ROSInterface

ROS 2 communication interface definition.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Interface name |
| `type` | enum | Yes | "topic", "service", "action" |
| `message_type` | string | Yes | ROS 2 message type |
| `direction` | enum | Yes | "publish", "subscribe", "server", "client" |
| `qos_profile` | string | No | Quality of service settings |
| `description` | string | Yes | Purpose of this interface |

### 7. ActionSchema

Structured output schema for LLM → robot action translation.

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `action_type` | enum | Yes | Action category |
| `parameters` | object | Yes | Action-specific parameters |
| `confidence` | float | Yes | LLM certainty (0.0-1.0) |
| `timeout_ms` | integer | Yes | Maximum execution time |
| `fallback` | ActionSchema | No | Recovery action if this fails |

**Action Types**:
- `navigate` - Move to location
- `grasp` - Pick up object
- `place` - Put down object
- `speak` - Verbal output
- `wait` - Pause execution
- `look` - Orient sensors
- `sequence` - Ordered action list

---

## Entity Relationships

```
Book
├── Chapter (1:N)
│   ├── CodeExample (1:N)
│   ├── Exercise (1:N)
│   └── prerequisites → Chapter (N:M)
│
├── CapstoneComponent (1:8, fixed)
│   ├── chapter_refs → Chapter (N:M)
│   ├── ROSInterface (1:N)
│   └── dependencies → CapstoneComponent (N:M)
│
└── Dependency (1:N, shared across chapters)
```

---

## State Transitions

### Chapter Status
```
draft → review → published → deprecated
```

### Exercise Completion (Reader State)
```
not_started → in_progress → completed
                         → stuck (hint requested)
```

### Capstone Integration
```
component_incomplete → component_tested → integrated → capstone_complete
```

---

## Validation Constraints

### Cross-Entity Rules

1. **Chapter Ordering**: `chapter.order` must be unique and sequential
2. **Prerequisite Validity**: `chapter.prerequisites` must reference chapters with lower `order`
3. **Capstone Coverage**: All 8 `CapstoneComponent.type` values must be covered by at least one chapter
4. **Dependency Consistency**: Same package cannot have conflicting versions across chapters
5. **Exercise Progression**: Each chapter must have at least 2 exercises
6. **Code Example Completeness**: Each `CodeExample` must have runnable source code

### Content Quality Rules (from Constitution)

1. **Reproducibility**: Every `CodeExample` must execute successfully on documented platforms
2. **No Undocumented Dependencies**: Every import/include must map to a `Dependency` entry
3. **Latency Documentation**: Examples involving real-time control must document timing constraints
4. **Error Case Coverage**: Each `CodeExample` must document expected failure modes

---

## Index Definitions

| Index Name | Entity | Fields | Purpose |
|------------|--------|--------|---------|
| `idx_chapter_order` | Chapter | order | Sequential navigation |
| `idx_chapter_slug` | Chapter | slug | URL routing |
| `idx_example_chapter` | CodeExample | chapter_id | Chapter content lookup |
| `idx_exercise_chapter` | Exercise | chapter_id | Chapter exercises lookup |
| `idx_component_type` | CapstoneComponent | type | Capstone validation |

---

## Data Integrity Checks

Run these checks before publishing:

```yaml
checks:
  - name: "All chapters have learning objectives"
    query: "chapters WHERE learning_objectives IS EMPTY"
    expected: 0

  - name: "All code examples have dependencies"
    query: "code_examples WHERE dependencies IS EMPTY"
    expected: 0

  - name: "All capstone components have integration tests"
    query: "capstone_components WHERE integration_test IS NULL"
    expected: 0

  - name: "No circular chapter prerequisites"
    query: "chapters WHERE prerequisites CONTAINS self OR creates_cycle"
    expected: 0

  - name: "All 8 capstone component types covered"
    query: "DISTINCT capstone_components.type"
    expected: 8
```
