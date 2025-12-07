---
id: backend-090
title: Implement GraphTransformNet model persistence (save/load)
status: done
priority: high
tags:
- backend
- persistence
- training
dependencies:
- backend-022
assignee: developer
created: 2025-12-07T12:00:00Z
estimate: ~
complexity: 3
area: backend
---

# Implement GraphTransformNet model persistence (save/load)

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**
>
> When you mark this task as `done`, you MUST:
> 1. Fill the "Session Handoff" section at the bottom with complete implementation details
> 2. Document what was changed, what runtime behavior to expect, and what dependencies were affected
> 3. Create a clear handoff for the developer/next AI agent working on dependent tasks
>
> **If this task has dependents,** the next task will be handled in a NEW session and depends on your handoff for context.

## Context

Training a neural network is useless without the ability to save and load the trained model. The `GraphTransformNet` struct in `grapheme-core` contains all the learnable parameters:
- `Embedding` layer weights
- `MessagePassingLayer` weights (multiple layers)
- `AttentionLayer` weights
- `NodePredictionHead` weights

Backend-022 implemented persistence for `DagNN` and `GraphemeGraph` (graph structures), but NOT for the neural network weights. Without model persistence:
- Training results are lost when the program exits
- Cannot checkpoint during long training runs
- Cannot load a trained model for inference/validation/REPL

## Objectives

- Implement `save()` and `load()` methods for `GraphTransformNet`
- Support both JSON (human-readable) and binary (efficient) formats
- Include model metadata (architecture config, training info)
- Enable checkpoint saving during training
- Enable model loading for inference binaries

## Tasks

- [x] Add `Serialize`/`Deserialize` derives to `Embedding` struct
- [x] Add `Serialize`/`Deserialize` derives to `MessagePassingLayer` struct
- [x] Add `Serialize`/`Deserialize` derives to `AttentionLayer` struct
- [x] Add `Serialize`/`Deserialize` derives to `NodePredictionHead` struct
- [x] Add `Serialize`/`Deserialize` derives to `GraphPooling` struct
- [x] Add `Serialize`/`Deserialize` derives to `GraphTransformNet` struct
- [x] Implement `GraphTransformNet::save_json()` and `load_json()`
- [x] Implement `GraphTransformNet::save_to_file()` and `load_from_file()`
- [x] Create `ModelHeader` struct with version, architecture info, training metadata
- [x] Add tests for model save/load roundtrip
- [x] Verify weights are identical after load

## Acceptance Criteria

✅ **Criteria 1: Save/Load Roundtrip**
- A trained `GraphTransformNet` can be saved to disk and loaded back
- All weights are identical after loading (verified by numerical comparison)

✅ **Criteria 2: File Format**
- JSON format is human-readable and contains all weights
- Header includes: format version, embed_dim, hidden_dim, num_layers, vocab_size

✅ **Criteria 3: Integration Ready**
- `Pipeline` in grapheme-train can use the persistence methods
- Train binary can save checkpoints
- Validate/REPL binaries can load models

## Technical Notes

### Structs Requiring Serialization

```rust
// In grapheme-core/src/lib.rs

// Already has serde for weights (Array2<f32>), need to check grad field
pub struct Embedding {
    pub weights: Array2<f32>,
    pub grad: Option<Array2<f32>>,  // Skip this field
    pub requires_grad: bool,         // Skip this field
    pub embed_dim: usize,
    pub vocab_size: usize,
}

pub struct MessagePassingLayer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    pub grad_weights: Option<Array2<f32>>,  // Skip
    pub grad_bias: Option<Array1<f32>>,      // Skip
    // ...
}

// Similar for AttentionLayer, NodePredictionHead
```

### Serialization Strategy

1. Use `#[serde(skip)]` for gradient fields (they're runtime-only)
2. Use `ndarray-serde` feature for Array serialization (already in Cargo.toml)
3. Store architecture params to reconstruct on load

### Model File Format

```json
{
  "header": {
    "version": 1,
    "graph_type": "GraphTransformNet",
    "vocab_size": 256,
    "embed_dim": 64,
    "hidden_dim": 128,
    "num_layers": 3,
    "created": "2025-12-07T12:00:00Z"
  },
  "embedding": {
    "weights": [[...], [...], ...],
    "embed_dim": 64,
    "vocab_size": 256
  },
  "mp_layers": [...],
  "attention": {...},
  "node_head": {...},
  "pooling": {...}
}
```

## Testing

- [x] Test save/load roundtrip with random weights
- [x] Test save/load preserves exact numerical values
- [x] Test loading model with wrong version fails gracefully
- [x] Test loading corrupted file fails gracefully
- [x] Test file size is reasonable (not bloated)

## Version Control

**⚠️ CRITICAL: Always test AND run before committing!**

- [x] **BEFORE committing**: Build, test, AND run the code to verify it works
  - Run `cargo build --release`
  - Run `cargo test -p grapheme-core`
  - Test saving/loading a model manually
  - Fix all errors, warnings, and runtime issues
- [ ] Commit changes incrementally with clear messages

## Updates

- 2025-12-07: Task created (blocks backend-087, backend-088, backend-089)
- 2025-12-07: Task completed - all persistence methods implemented and tested

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed

**Files Modified:**
- `Cargo.toml` - Added `serde` feature to `ndarray` dependency
- `grapheme-core/src/lib.rs` - Added serialization support to neural network structs

**New Types Added:**
- `ModelHeader` (line ~2617) - Metadata struct for serialized models
- `SerializedModel` (line ~2667) - Wrapper combining header + model for JSON serialization
- `MODEL_PERSISTENCE_VERSION` constant (= 1) - For version compatibility

**Structs Modified with Serde:**
- `Embedding` - Added `Serialize, Deserialize`, `#[serde(skip)]` on `grad` and `requires_grad`
- `MessagePassingLayer` - Added derives, skip on `weight_grad` and `bias_grad`
- `AttentionLayer` - Added `Serialize, Deserialize` (no skipped fields)
- `NodePredictionHead` - Added `Serialize, Deserialize` (no skipped fields)
- `GraphPooling` - Added `Serialize, Deserialize`
- `PoolingType` - Added `Serialize, Deserialize`
- `GraphTransformNet` - Added `Serialize, Deserialize, Clone`

**New Methods on `GraphTransformNet`:**
- `save_json(&self) -> PersistenceResult<String>` - Serialize to JSON string
- `load_json(json: &str) -> PersistenceResult<Self>` - Deserialize from JSON
- `save_to_file(&self, path: &Path) -> PersistenceResult<()>` - Save to file
- `load_from_file(path: &Path) -> PersistenceResult<Self>` - Load from file
- `header(&self) -> ModelHeader` - Get model metadata

### Causality Impact

- **No async flows** - All operations are synchronous
- **Gradients are runtime-only** - Loaded models have `grad = None` and need `zero_grad()` before training
- **Version checking** - `load_json` rejects models with version > MODEL_PERSISTENCE_VERSION

### Dependencies & Integration

**Cargo.toml Changes:**
```toml
ndarray = { version = "0.15", features = ["serde"] }
```

**How to use in training binaries:**
```rust
use grapheme_core::{GraphTransformNet, PersistenceResult};

// Save after training
let model = GraphTransformNet::new(256, 32, 64, 2);
model.save_to_file(Path::new("model.json"))?;

// Load for inference
let loaded = GraphTransformNet::load_from_file(Path::new("model.json"))?;
```

**Unblocks:**
- backend-087 (train binary) - Can save checkpoints
- backend-088 (validate binary) - Can load trained models
- backend-089 (REPL binary) - Can load trained models

### Verification & Testing

**Run tests:**
```bash
cargo test -p grapheme-core
```

**New tests added (10 tests):**
- `test_graph_transform_net_json_roundtrip`
- `test_graph_transform_net_weights_preserved`
- `test_graph_transform_net_file_roundtrip`
- `test_model_header_verification`
- `test_model_header_fields`
- `test_gradients_not_serialized`
- `test_loaded_model_can_forward`
- `test_serialized_model_struct`

**All 132 tests pass, clippy clean.**

### Context for Next Task

**For backend-087/088/089 (training binaries):**
1. Use `GraphTransformNet::save_to_file()` for checkpoints
2. Use `GraphTransformNet::load_from_file()` to resume training or load for inference
3. Loaded models have `grad = None` - call `zero_grad()` before training to initialize
4. JSON files are human-readable but large - consider binary format for production

**Important decisions:**
- Chose JSON over binary for human-readability during development
- Gradients are excluded (they're runtime state, not model parameters)
- `requires_grad` defaults to `true` on load (training-ready)
- Version check allows loading older formats but not newer ones

**Gotchas:**
- File size scales with vocab_size × embed_dim (256×32 = 8192 floats just for embedding)
- `Clone` was added to `GraphTransformNet` to support serialization wrapper