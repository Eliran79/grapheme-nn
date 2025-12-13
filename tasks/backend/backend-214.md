---
id: backend-214
title: Unified Semantic Node Vocabulary
status: done
priority: critical
tags:
- backend
dependencies:
- backend-216
assignee: developer
created: 2025-12-11T23:02:36.534281450Z
estimate: ~
complexity: 3
area: backend
---

# Unified Semantic Node Vocabulary

## Context
The semantic code training showed 0% accuracy because the model could only output the same node types it received (Input chars). It couldn't generate NEW semantic node types like Keyword, Variable, Op, etc. This task provides the unified vocabulary infrastructure.

## Objectives
- [x] Collect all node types from all domain brains
- [x] Create unified vocabulary accessible throughout the codebase
- [x] Enable models to generate ANY semantic node type
- [x] Support auto-discovery from registered brains

## Tasks
- [x] Add `node_types()` method to DomainBrain trait (api-021)
- [x] Implement `node_types()` for all domain brains (backend-215)
- [x] Build SemanticDecoder with unified vocab (backend-216)
- [x] Add `collect_all_node_types()` helper in cortex_mesh.rs
- [x] Export unified vocabulary through lib.rs

## Acceptance Criteria
✅ **Vocabulary Collection:**
- `collect_all_node_types()` returns 4301 unique node types
- Covers all brains: Math, Code, Vision, Chem, Law, Music, Time, Classification

✅ **SemanticDecoder:**
- Can decode hidden states to any NodeType in unified vocab
- Supports training with `backward()` method
- Supports serialization with `save()/load()`

## Technical Notes
- Unified vocab: 4301 types (4096 Pixel, 97 Input, 35 Keyword, 25 Op, etc.)
- Xavier initialization for embeddings and output projection
- Cross-entropy loss with label smoothing (default 0.1)
- Softmax temperature for output diversity control

## Testing
- [x] `test_build_vocab_from_brains` - 8 tests pass
- [x] `test_decode`, `test_topk_decode` - Decoding works correctly
- [x] `test_backward_reduces_loss` - Training converges
- [x] `test_save_load` - Persistence works

## Version Control
- [x] Build passes with zero errors
- [x] All tests pass (8/8)
- [x] Committed: `8183d0d feat(semantic): Add SemanticDecoder with unified vocabulary from all brains`

## Updates
- 2025-12-11: Task created
- 2025-12-12: All dependencies completed, task complete

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- `grapheme-train/src/semantic_decoder.rs`: NEW - SemanticDecoder module (~600 lines)
  - `SemanticDecoder` struct with learnable embeddings for all 4301 node types
  - `decode()` returns predicted NodeType with confidence
  - `decode_topk()` returns top-k predictions
  - `backward()` for training with cross-entropy loss
  - `encode()` returns type embedding for a NodeType
  - `save()/load()` for persistence
- `grapheme-train/src/cortex_mesh.rs`: Added `collect_all_node_types()` function
- `grapheme-train/src/lib.rs`: Exports `SemanticDecoder`, `SemanticDecoderConfig`, `VocabStats`, `collect_all_node_types`

### Causality Impact
- Models can now output ANY semantic node type (Keyword, Variable, Op, etc.)
- Breaking change: Training must now use unified vocab to avoid model collapse
- The semantic training pipeline should use SemanticDecoder for output generation

### Dependencies & Integration
- Depends on: `DomainBrain::node_types()` (api-021) implemented in all brains (backend-215)
- Exports unified vocabulary through `collect_all_node_types()`
- SemanticDecoder integrates with GraphTransformNet by decoding hidden states

### Verification & Testing
```bash
# Verify unified vocab collection
cargo test -p grapheme-train --lib semantic_decoder::tests -- --nocapture

# Check vocab stats
# Output shows: Total size: 4301, 35 Keywords, 25 Ops, etc.
```

### Context for Next Task
- The SemanticDecoder provides the OUTPUT layer for semantic code generation
- Next step: Integrate SemanticDecoder into `train_semantic_code.rs` to replace the broken output path
- Key insight: The model collapse happened because outputs were copied from inputs - SemanticDecoder allows generating NEW node types not in the input
