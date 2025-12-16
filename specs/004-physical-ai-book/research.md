# Research Document: Physical AI & Humanoid Robotics Book

**Branch**: `004-physical-ai-book` | **Date**: 2025-12-15
**Status**: Complete - All open questions resolved

## Overview

This document consolidates research findings for the four open questions identified in the specification. Each decision is documented with rationale and alternatives considered.

---

## Decision 1: Humanoid Robot Model

### Decision
**Primary Model**: Custom URDF based on open-source humanoid designs (Simple Humanoid + TALOS patterns)
**Reference Models**: ATLAS (Boston Dynamics DRC-Sim) for advanced chapters

### Rationale

1. **Availability**: Custom URDF provides full control and no licensing concerns for open textbook
2. **ROS 2 Support**: Native integration via standard URDF tools and gazebo_ros packages
3. **Educational Value**: Students learn URDF structure while having a working model
4. **Simulation Performance**: Optimized for educational hardware (laptops with modest specs)
5. **Documentation**: Can be fully documented within the book itself

### Alternatives Considered

| Model | Pros | Cons | Decision |
|-------|------|------|----------|
| **NAO** | Well-documented, educational focus | Proprietary focus, aging platform | Rejected - licensing concerns |
| **Digit (Agility)** | Modern design, industry backing | Limited public URDF, smaller community | Rejected - less accessible |
| **ATLAS** | Excellent documentation, proven | Complex, high compute needs | Secondary reference only |
| **Custom URDF** | Full control, optimized for book | Requires initial design work | **Selected** |

### Implementation Notes

- Provide a simplified 21-DOF humanoid model (7 per arm, 6 per leg, 1 torso)
- Include both visual and collision meshes optimized for Gazebo
- Document all joint limits, inertias, and sensor placements
- Offer "light" variant for low-spec machines

---

## Decision 2: Simulation Platform Balance

### Decision
**Primary Platform**: Gazebo Sim (Ignition Garden/Harmonic) - 90% of content
**Secondary Platform**: Unity with ROS 2 - 10% for visual validation

### Rationale

1. **ROS 2 Native Integration**: Gazebo Sim has mature `gz_ros2_bridge` and `gz_ros2_control`
2. **Pedagogical Alignment**: Students write ROS 2 code that transfers to real robots
3. **Reproducibility**: Gazebo simulations are deterministic (critical for textbook)
4. **Lower Barrier**: AI/ML students focus on robotics concepts, not game engine learning
5. **Open Source**: No licensing complexity for educational use

### Platform Usage

| Content Type | Platform | Rationale |
|--------------|----------|-----------|
| Core control tutorials | Gazebo Sim | ROS 2 native, deterministic |
| Sensor simulation | Gazebo Sim | Configurable noise models |
| Visual demonstrations | Unity | Superior rendering for marketing |
| Capstone project | Gazebo Sim | Integration with full ROS 2 stack |

### Physics Engine Selection

- **Primary**: Bullet-3 (default in Gazebo Sim) - good balance of accuracy and speed
- **Alternative**: DART for advanced dynamics chapters - best humanoid contact stability

### Alternatives Considered

| Platform | Pros | Cons | Decision |
|----------|------|------|----------|
| **Gazebo Classic** | Familiar | EOL January 2025 | Rejected |
| **Gazebo Sim** | Modern, ROS 2 native | Learning curve | **Selected** |
| **Unity** | Visual quality, PhysX | C# required, less ROS 2 mature | Secondary only |
| **Isaac Sim** | NVIDIA backed, RL focus | Heavy compute, proprietary | Out of scope |

---

## Decision 3: RL vs Classical Control Balance

### Decision
**Split**: 70% Classical Control / 30% Reinforcement Learning
**Classical**: Full hands-on implementation (students write controllers)
**RL**: Guided implementation with pre-built environment (students train + analyze)

### Rationale

1. **Pedagogical Order**: Classical control is prerequisite - students need deterministic foundation
2. **Sim-to-Real Transfer**: Classical controllers transfer directly; RL requires domain randomization
3. **Industry Practice**: Production robots use hybrid approaches, weighted toward classical
4. **Content Stability**: PID/impedance fundamentals stable for 30+ years; RL tooling changes rapidly
5. **Time Budget**: Realistic for 40-hour capstone completion target

### Coverage Matrix

| Topic | Classical | RL | Depth |
|-------|-----------|-----|-------|
| PID Control | Primary | - | Hands-on implementation |
| Impedance Control | Primary | - | Conceptual + simplified implementation |
| Trajectory Planning | Primary | - | Computational exercises |
| Policy Gradient Concepts | - | Primary | Conceptual (visual intuition) |
| Locomotion Learning | - | Primary | Guided (Isaac Gym, PPO) |
| Sim-to-Real Transfer | - | Primary | Reality check discussion |

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **100% Classical** | Stable, transfers | Misses modern AI | Rejected |
| **50/50 Split** | Balanced | RL content too deep for scope | Rejected |
| **70/30 Split** | Foundation + exposure | Limited RL depth | **Selected** |
| **RL-First** | Modern approach | Students can't debug | Rejected |

---

## Decision 4: LLM Provider Abstraction

### Decision
**Pattern**: Stratified Adapter Model with three abstraction layers
**Default Provider**: Ollama (local) - enables offline book examples
**Cloud Providers**: OpenAI/Anthropic abstracted for advanced chapters

### Architecture

```
Layer 1: Provider Interface (ILLMProvider)
├── invoke(request) -> ActionResponse
├── invoke_streaming(request) -> AsyncIterator
└── get_metrics() -> ProviderMetrics

Layer 2: Provider Implementations
├── OllamaAdapter (local, primary for book)
├── OpenAIAdapter (cloud, advanced chapters)
├── AnthropicAdapter (cloud, alternative)
└── FallbackChainAdapter (graceful degradation)

Layer 3: Robotics Enhancement
├── StructuredOutputEnforcer (JSON schema validation)
├── RateLimiter (token bucket + backoff)
├── LatencyMonitor (real-time tracking)
└── SafetyValidator (bounds checking)
```

### Key Design Decisions

1. **Structured Output**: All LLM responses validated against JSON action schemas
2. **Latency Budget**: 50ms target for real-time control (Ollama achieves this)
3. **Error Handling**: Transient → retry, Semantic → validate, Deterministic → fallback
4. **Cost Management**: Local-first, cache heavily, cloud for complex reasoning

### Rationale

1. **Offline Examples**: Students can follow book without API keys
2. **Determinism**: Local inference with fixed seeds enables reproducible results
3. **Flexibility**: Abstraction allows provider swapping without code changes
4. **Safety**: Schema validation prevents LLM hallucinations from reaching actuators
5. **Real-time**: Ollama meets latency requirements for 10Hz control loops

### Alternatives Considered

| Pattern | Pros | Cons | Decision |
|---------|------|------|----------|
| **Direct API calls** | Simple | Tight coupling, no fallback | Rejected |
| **Single provider** | Easy | No flexibility | Rejected |
| **Stratified Adapter** | Flexible, testable, safe | Initial complexity | **Selected** |
| **LangChain/LlamaIndex** | Pre-built | Heavy dependencies, less control | Rejected |

---

## Technology Stack Summary

Based on research findings, the complete technology stack is:

| Category | Technology | Version | Notes |
|----------|------------|---------|-------|
| **Robot Framework** | ROS 2 | Humble/Iron | LTS releases only |
| **Simulation** | Gazebo Sim | Garden/Harmonic | Primary platform |
| **Physics Engine** | Bullet-3 | (bundled) | Default; DART for advanced |
| **Robot Model** | Custom URDF | - | 21-DOF humanoid |
| **Languages** | Python 3.10+, C++17 | - | Python primary for examples |
| **LLM Local** | Ollama | Latest | Primary for book examples |
| **LLM Cloud** | OpenAI/Anthropic | - | Abstracted, optional |
| **RL Framework** | Isaac Gym | Latest | For locomotion chapter |
| **ML Framework** | stable-baselines3 | Latest | PPO implementation |
| **Documentation** | Docusaurus | 3.x | GitHub Pages deployment |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gazebo API changes | Low | High | Pin to LTS releases |
| Ollama model quality | Medium | Medium | Test with multiple models |
| Isaac Gym deprecation | Low | Medium | Abstract RL environment interface |
| Custom URDF complexity | Medium | Medium | Start simple, iterate |
| ROS 2 version fragmentation | Low | High | Document version matrix |

---

## Next Steps

1. **Phase 1**: Create custom humanoid URDF model
2. **Phase 1**: Set up Gazebo Sim + ROS 2 development environment
3. **Phase 1**: Implement LLM abstraction layer with Ollama adapter
4. **Phase 1**: Design chapter structure aligned with these decisions
5. **Phase 2**: Generate tasks.md for implementation

---

## References

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- Gazebo Sim Documentation: https://gazebosim.org/docs/latest/
- Ollama: https://ollama.ai/
- Isaac Gym: https://developer.nvidia.com/isaac-gym
- stable-baselines3: https://stable-baselines3.readthedocs.io/
