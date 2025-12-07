---
id: backend-091
title: Implement unified persistence infrastructure for all cognitive modules
status: todo
priority: high
tags:
- backend
- persistence
- architecture
- infrastructure
dependencies:
- backend-090
assignee: developer
created: 2025-12-07T14:00:00Z
estimate: ~
complexity: 5
area: backend
---

# Implement unified persistence infrastructure for all cognitive modules

## Context

Backend-090 implemented persistence for `GraphTransformNet` in grapheme-core. However, GRAPHEME has **17 cognitive modules** that may need persistence:

| Module | Lines | Learnable State? | Audited |
|--------|-------|------------------|---------|
| grapheme-core | 7,071 | ✅ Yes (GraphTransformNet, Embedding, etc.) | ✅ |
| grapheme-train | 4,369 | ✅ Yes (TrainingState, SGD, Adam) | ✅ |
| grapheme-engine | 2,797 | ❌ No (pure rule engine) | ✅ |
| grapheme-memory | 1,848 | ✅ Yes (LearnableMemoryRetrieval - 6 params) | ✅ |
| grapheme-math | 1,686 | ❌ No (rule-based MathEngine) | ✅ |
| grapheme-reason | 1,658 | ✅ Yes (LearnableReasoning - 6 params) | ✅ |
| grapheme-polish | 1,209 | ❌ No (stateless optimizer) | ✅ |
| grapheme-code | 1,130 | ❌ No (stateless parser) | ✅ |
| grapheme-meta | 1,058 | ✅ Yes (LearnableMetaCognition - 5 params) | ✅ |
| grapheme-agent | 1,045 | ✅ Yes (LearnableAgency - 6 params) | ✅ |
| grapheme-world | 977 | ✅ Yes (LearnableWorldModel - 6 params) | ✅ |
| grapheme-ground | 971 | ✅ Yes (LearnableGrounding - 5 params) | ✅ |
| grapheme-multimodal | 939 | ✅ Yes (LearnableMultimodal - 6 params) | ✅ |
| grapheme-parallel | 835 | ❌ No (infrastructure only) | ✅ |
| grapheme-chem | 662 | ❌ No (stateless ChemBrain) | ✅ |
| grapheme-music | 622 | ❌ No (stateless MusicBrain) | ✅ |
| grapheme-law | 619 | ❌ No (stateless LawBrain) | ✅ |

**Problem:** Each module implementing its own persistence leads to:
- Code duplication
- Inconsistent file formats
- No unified checkpoint system
- Harder to save/load entire GRAPHEME state

## Objectives

1. **Audit** each module for learnable/persistent state
2. **Design** unified persistence traits and infrastructure
3. **Implement** shared persistence in a common location
4. **Migrate** GraphTransformNet to use shared infrastructure
5. **Implement** persistence for each module that needs it

## Tasks

### Phase 1: Audit & Design

- [x] Audit grapheme-memory for persistent state (4 memory types)
- [x] Audit grapheme-reason for persistent state (5 reasoning modes)
- [x] Audit grapheme-math for persistent state (math brain)
- [x] Audit grapheme-engine for persistent state (rule engine)
- [x] Audit grapheme-world for persistent state (world model)
- [x] Audit grapheme-meta for persistent state (meta-cognition)
- [x] Audit grapheme-agent for persistent state (goals/policies)
- [x] Audit grapheme-ground for persistent state (symbol bindings)
- [x] Audit grapheme-multimodal for persistent state (cross-modal)
- [x] Audit grapheme-code for persistent state (code brain)
- [x] Audit grapheme-polish for persistent state (IR state)
- [x] Audit domain brains: grapheme-law, grapheme-music, grapheme-chem
- [x] Document findings in this task

#### Audit Results (2025-12-07)

**Modules WITH Learnable State (need persistence):**

| Module | Learnable Struct | Parameters | Location |
|--------|-----------------|------------|----------|
| grapheme-core | `GraphTransformNet` | Neural network weights (Array2) | lib.rs:4050 |
| grapheme-core | `Embedding`, `MessagePassingLayer` | Neural network weights | lib.rs:2703, 3709 |
| grapheme-memory | `LearnableMemoryRetrieval` | 6 params (node/edge/degree/type weights, importance_bias, temperature) | lib.rs:1104 |
| grapheme-reason | `LearnableReasoning` | 6 params (deduction/induction/abduction confidence, analogy/causal weights, temperature) | lib.rs:1100 |
| grapheme-world | `LearnableWorldModel` | 6 params (transition_bias, prediction_confidence, temporal_discount, entity/relation weights, uncertainty_scale) | lib.rs:598 |
| grapheme-meta | `LearnableMetaCognition` | 5 params (calibration_bias, uncertainty_scale, epistemic_weight, compute_bias, early_stop_threshold) | lib.rs:611 |
| grapheme-agent | `LearnableAgency` | 6 params (goal_importance_bias, curiosity/safety/efficiency weights, explore_temperature, discount_factor) | lib.rs:720 |
| grapheme-ground | `LearnableGrounding` | 5 params (grounding_threshold, perception/action weights, exploration_bonus, cooccurrence_rate) | lib.rs:624 |
| grapheme-multimodal | `LearnableMultimodal` | 6 params (visual/auditory/linguistic/tactile weights, binding_strength, fusion_temperature) | lib.rs:582 |
| grapheme-train | `TrainingState`, `TrainingMetrics` | Training checkpoints (epoch, step, losses, LR) | lib.rs:2413, 2449 |
| grapheme-train | `SGD`, `Adam` | Optimizer state (momentum, velocity) | lib.rs:2160, 2235 |

**Modules WITHOUT Learnable State (no persistence needed):**

| Module | Reason |
|--------|--------|
| grapheme-math | `MathBrain` has no learnable params - uses `MathEngine` (rule-based) |
| grapheme-engine | Pure rule engine - no learnable weights |
| grapheme-polish | `Optimizer` is stateless - rule-based transformations |
| grapheme-code | `CodeBrain` is stateless - parses code structure |
| grapheme-law | `LawBrain` is stateless - IRAC analysis |
| grapheme-music | `MusicBrain` is stateless - music theory analysis |
| grapheme-chem | `ChemBrain` is stateless - molecular analysis |
| grapheme-parallel | Infrastructure only - no persistent state |

**Key Finding:** All learnable modules use the `LearnableParam` struct from grapheme-core, implementing the `Learnable` trait. This is a consistent pattern that simplifies persistence design.

**Parameter Summary:**
- Neural networks (grapheme-core): ~100K+ parameters (Array2 matrices)
- Cognitive modules (memory, reason, world, meta, agent, ground, multimodal): 5-6 `LearnableParam` each (~40 total)
- Training state (grapheme-train): Checkpoints + optimizer momentum

### Phase 2: Shared Infrastructure

- [x] Create `grapheme-persist` crate OR add to grapheme-core → **Added to grapheme-core**
- [x] Define `Persistable` trait: `grapheme-core/src/lib.rs:2622`
- [x] Define unified `UnifiedCheckpoint` struct: `grapheme-core/src/lib.rs:2639`
- [x] Implement checkpoint save/load methods on `UnifiedCheckpoint`
- [ ] Add binary format option (bincode/messagepack) for efficiency
- [ ] Add compression option (zstd) for large models

### Phase 3: Implementation per Module

- [x] Implement Persistable for grapheme-core::GraphTransformNet (`lib.rs:2757`)
- [x] Implement Persistable for grapheme-core::LearnableParam (`lib.rs:2747`)
- [x] Implement Persistable for grapheme-memory::LearnableMemoryRetrieval (`lib.rs:1244`)
- [x] Implement Persistable for grapheme-reason::LearnableReasoning (`lib.rs:1245`)
- [x] Implement Persistable for grapheme-world::LearnableWorldModel (`lib.rs:710`)
- [x] Implement Persistable for grapheme-meta::LearnableMetaCognition (`lib.rs:713`)
- [x] Implement Persistable for grapheme-agent::LearnableAgency (`lib.rs:833`)
- [x] Implement Persistable for grapheme-ground::LearnableGrounding (`lib.rs:728`)
- [x] Implement Persistable for grapheme-multimodal::LearnableMultimodal (`lib.rs:706`)
- [N/A] Domain brains (code, law, music, chem) - No learnable state (see audit)

### Phase 4: Integration

- [ ] Update grapheme-train to use unified checkpointing
- [ ] Add checkpoint save/resume to training pipeline
- [ ] Add full GRAPHEME state save/load
- [ ] Add tests for cross-module checkpoint roundtrip

## Design Decisions

### Option A: New `grapheme-persist` crate
**Pros:** Clean separation, all persistence in one place
**Cons:** New dependency for all modules, circular dependency risk

### Option B: Add to `grapheme-core`
**Pros:** Already depended on by all modules, simpler
**Cons:** Core gets larger, may not be "core" functionality

### Option C: Traits in core, implementations in each module
**Pros:** Distributed responsibility, each module owns its persistence
**Cons:** More code duplication, harder to ensure consistency

**Recommended:** Option B - add to grapheme-core since it already has persistence infrastructure from backend-022 and backend-090.

## File Format Design

```json
{
  "checkpoint_version": 1,
  "created": "2025-12-07T14:00:00Z",
  "grapheme_version": "0.1.0",
  "modules": {
    "GraphTransformNet": {
      "version": 1,
      "data": { ... }
    },
    "EpisodicMemory": {
      "version": 1,
      "data": { ... }
    },
    "ReasoningEngine": {
      "version": 1,
      "data": { ... }
    }
  }
}
```

## Acceptance Criteria

- [ ] All modules with learnable state have persistence
- [ ] Single checkpoint file can save/load entire GRAPHEME state
- [ ] No code duplication across modules
- [ ] Binary format available for production use
- [ ] All existing tests still pass
- [ ] New tests for unified checkpoint system

## Dependencies

- backend-090 (GraphTransformNet persistence) - DONE

## Dependents

- Training binaries (backend-087, 088, 089) will benefit from unified checkpointing

## Updates

- 2025-12-07: Task created after completing backend-090
- 2025-12-07: **Phase 1 Complete** - Audited all 17 modules:
  - 9 modules have learnable state (need persistence)
  - 8 modules are stateless (no persistence needed)
  - Key insight: All learnable modules use `LearnableParam` struct from grapheme-core
- 2025-12-07: **Phase 2 Complete** - Implemented unified persistence infrastructure:
  - Added `Persistable` trait to grapheme-core
  - Added `UnifiedCheckpoint` struct with add_module/load_module methods
  - Added serde derives to `LearnableParam` and all `Learnable*` structs
  - Added 9 new tests for unified checkpoint system
- 2025-12-07: **Phase 3 Complete** - Implemented Persistable for all cognitive modules:
  - GraphTransformNet, LearnableParam (grapheme-core)
  - LearnableMemoryRetrieval (grapheme-memory)
  - LearnableReasoning (grapheme-reason)
  - LearnableWorldModel (grapheme-world)
  - LearnableMetaCognition (grapheme-meta)
  - LearnableAgency (grapheme-agent)
  - LearnableGrounding (grapheme-ground)
  - LearnableMultimodal (grapheme-multimodal)
  - All with validation methods for parameter constraints

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [To be filled on completion]

### Audit Results
- [Document which modules need persistence and what state they hold]

### Architecture Decisions
- [Document final architecture choice and rationale]
