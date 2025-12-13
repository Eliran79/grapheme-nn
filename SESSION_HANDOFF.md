# GRAPHEME Session Handoff - 2025-12-12

## Goal
Beat HumanEval SOTA (96.2% pass@1) using GRAPHEME's graph-to-graph transformation architecture.

## What Was Done This Session

### 1. Removed Character-Level Similarity (Vision Drift Fix)
- **Removed** `similarity()` Jaccard function from `train_cortex_mesh.rs`
- **Removed** `val_similarity` metric from validation loop
- **Kept** `cosine_similarity` in `StructuralClassifier` (needed for classification)
- **Kept** `char_weight` parameter (can be set to 0.0)

### 2. Committed and Tagged
```
Commit: 61d5eb9 feat(semantic): Add semantic code graph training and HumanEval eval
Tag: v0.8.0-semantic
```

### 3. Created TaskGuard Tasks
```
api-021     → Add DomainBrain::node_types() trait method (available)
    ↓
backend-215 → Implement node_types for all domain brains (blocked)
    ↓
backend-216 → Build semantic node decoder with unified vocab (blocked)
    ↓
backend-214 → Unified Semantic Node Vocabulary (blocked)
```

### 4. Created Documentation
- `docs/VISION_DRIFT_PREVENTION.md` - Explains the char-to-char vs graph-to-graph vision drift problem

### 5. Uncommitted Files
```
docs/VISION_DRIFT_PREVENTION.md
tasks/api/api-021.md
tasks/backend/backend-214.md
tasks/backend/backend-215.md
tasks/backend/backend-216.md
```

## Critical Finding: Model Collapse

The semantic code training completed 100 epochs but shows **0% semantic accuracy**:

```
Loss: 952 → 639 (converged)
semantic_acc: 0% (never improved)
Output: All Input('x') nodes → "xxxx..."
```

**Root Cause**: The model CANNOT generate NEW semantic node types. It only transforms existing text nodes. The forward pass outputs the same node type it received.

## The Architecture Problem

```
Current (broken):
  Input: [Input('W'), Input('r'), Input('i'), ...]
  Output: [Input('x'), Input('x'), Input('x'), ...] ← Can't change node types!

Required:
  Input: [Input('W'), Input('r'), Input('i'), ...]
  Output: [Keyword(def), Variable(f), Punct('('), Variable(x), ...] ← NEW types!
```

The model needs:
1. **Unified semantic vocabulary** from all domain brains
2. **Node type decoder** that can output ANY semantic node type
3. **Auto-discovery** to collect all domain-specific node types

## Next Steps (Priority Order)

### Immediate (api-021)
Add `node_types()` method to `DomainBrain` trait:
```rust
pub trait DomainBrain {
    fn node_types(&self) -> Vec<NodeType>;  // Add this
    fn can_process(&self, input: &str) -> bool;
    fn domain_name(&self) -> &'static str;
    // ...
}
```

### Then (backend-215)
Implement for each brain:
- **TextBrain**: `Input(char)` for all ASCII chars
- **CodeBrain**: `Keyword(*)`, `Variable(*)`, `Int`, `Float`, `Str`, `Op(*)`, `Call(*)`, `Punct(*)`
- **MathBrain**: `MathInt`, `MathFloat`, `MathOp(*)`, `Function(*)`, `Subscript`, `Superscript`
- **ChemBrain**: `Atom(*)`, `Bond(*)`, `Coefficient`, `Formula`
- **MusicBrain**: `Note(*)`, `Rest(*)`, `Chord(*)`, `Tempo`

### Then (backend-216)
Build semantic node decoder:
```rust
struct SemanticDecoder {
    unified_vocab: Vec<NodeType>,  // All types from all brains
    type_embeddings: Vec<Vec<f32>>,  // Learnable embeddings per type
    output_projection: Matrix,  // hidden_dim → vocab_size
}

impl SemanticDecoder {
    fn decode(&self, hidden: &[f32]) -> NodeType {
        let logits = matmul(&self.output_projection, hidden);
        let idx = argmax(&softmax(&logits));
        self.unified_vocab[idx].clone()
    }
}
```

### Finally (backend-214)
Integrate into CortexMesh auto-discovery.

## Key Files

| File | Purpose |
|------|---------|
| `grapheme-core/src/lib.rs` | `NodeType` enum, `DomainBrain` trait |
| `grapheme-train/src/cortex_mesh.rs` | CortexMesh, BRAIN_FACTORIES |
| `grapheme-train/src/bin/train_semantic_code.rs` | Semantic training (broken) |
| `docs/VISION_DRIFT_PREVENTION.md` | Vision drift documentation |

## Background Processes
Multiple training runs may still be running. Check with `ps aux | grep train`.

## Commands to Continue
```bash
# Check task status
taskguard validate

# Start working on api-021
taskguard update status api-021 doing

# After implementing, mark done
taskguard update status api-021 done
```
