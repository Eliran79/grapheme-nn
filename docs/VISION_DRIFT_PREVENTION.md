# GRAPHEME Vision Drift Prevention

## The Problem: Character-Level Similarity Creep

During development, character-level similarity metrics repeatedly crept into the codebase, causing **vision drift** away from GRAPHEME's core graph-to-graph transformation paradigm.

### What Happened

1. **Initial Implementation**: Training code used Jaccard character similarity to measure output quality
2. **Encoder/Decoder**: Cosine similarity was added for character-level loss
3. **Metrics Confusion**: Character accuracy metrics dominated training feedback
4. **Result**: Model optimized for character reproduction instead of semantic graph generation

### Why This Is Wrong

GRAPHEME's vision is fundamentally different from traditional language models:

```
Traditional LLM:     Text → Tokens → Attention → Tokens → Text
GRAPHEME Vision:     Graph → Message Passing → Graph Transformation → Graph
```

**Characters are NOT the unit of computation in GRAPHEME.**

The correct semantic units are:
- `Keyword(def)`, `Keyword(if)`, `Keyword(return)` - Language constructs
- `Variable(x)`, `Variable(result)` - Named identifiers
- `Int(42)`, `Float(3.14)` - Numeric literals
- `Op(+)`, `Op(>)`, `Op(==)` - Operators
- `Str("hello")` - String literals
- `Call(print)` - Function invocations
- `Atom(H)`, `Atom(O)` - Chemistry elements
- `Note(C4)`, `Rest(quarter)` - Music notation

## The Correct Approach

### What We Removed

| Location | Removed | Reason |
|----------|---------|--------|
| `train_cortex_mesh.rs` | `similarity()` function | Jaccard char similarity misleads training |
| `train_cortex_mesh.rs` | `val_similarity` metric | Char-level validation is meaningless |
| Training output | `sim=X.XX%` display | Encourages wrong optimization target |

### What We Kept

| Location | Kept | Reason |
|----------|------|--------|
| `StructuralClassifier` | `cosine_similarity()` | Classification requires embedding comparison |
| `cortex_mesh.rs` | `char_weight` parameter | Can be set to 0.0 to disable; useful for debugging |

### The Correct Metrics

1. **Structural Loss** (WL Kernel)
   - Compares graph topology
   - Node type distribution matching
   - Edge connectivity patterns

2. **Semantic Node Accuracy**
   - Percentage of correct node TYPE predictions
   - `Keyword` vs `Variable` vs `Int` etc.
   - NOT character matching

3. **Exact Code Match** (for evaluation only)
   - Final decoded output comparison
   - Only meaningful after semantic graph is correct

## Code Examples

### WRONG: Character-Level Training
```rust
// DO NOT DO THIS - causes vision drift
fn similarity(a: &str, b: &str) -> f32 {
    let a_chars: HashSet<char> = a.chars().collect();
    let b_chars: HashSet<char> = b.chars().collect();
    let intersection = a_chars.intersection(&b_chars).count();
    intersection as f32 / a_chars.union(&b_chars).count() as f32
}

// Training with char similarity - WRONG
let sim = similarity(&output, &target);
println!("sim={:.2}%", sim * 100.0);  // Misleading metric
```

### CORRECT: Semantic Graph Training
```rust
// CORRECT - compare semantic graph structure
fn semantic_accuracy(pred: &GraphemeGraph, target: &GraphemeGraph) -> f32 {
    let pred_types: Vec<_> = pred.nodes().map(|n| n.node_type.name()).collect();
    let target_types: Vec<_> = target.nodes().map(|n| n.node_type.name()).collect();

    let matches = pred_types.iter().zip(&target_types)
        .filter(|(p, t)| p == t)
        .count();

    matches as f32 / target_types.len() as f32
}

// Training with structural loss - CORRECT
let loss = compute_structural_loss(&output_graph, &target_graph, &config);
println!("struct_loss={:.4}", loss.total_loss);
```

## Architecture Decision Record

### ADR-001: No Character-Level Similarity in Training

**Status**: Accepted

**Context**: Training code evolved to include character-level metrics that conflict with GRAPHEME's graph-to-graph vision.

**Decision**: Remove all character-level similarity from training loops. Keep only in classification tasks where embedding comparison is genuinely needed.

**Consequences**:
- Training optimizes for correct graph topology
- Model learns semantic node type prediction
- Evaluation uses `semantic_accuracy` not `char_similarity`
- Debugging may be harder (can re-enable with `char_weight=0.0`)

## Future Prevention Checklist

When adding new training code, verify:

- [ ] No `similarity()` or `jaccard()` functions comparing strings
- [ ] No character-level accuracy metrics in training output
- [ ] Loss functions operate on `GraphemeGraph` not `String`
- [ ] Metrics measure node TYPE accuracy, not character reproduction
- [ ] `char_weight` defaults to `0.0` or is explicitly documented

## The Vision: Graph-to-Graph

```
Input:  "write a function that prints Hi if x>2"
        ↓
        GraphemeGraph { nodes: [Input('w'), Input('r'), ...] }
        ↓
        Message Passing Layers (topology transformation)
        ↓
Output: GraphemeGraph {
          nodes: [
            Keyword(def), Variable(f), Punct('('), Variable(x), Punct(')'),
            Punct(':'), Space(Newline), Space(Indent), Keyword(if),
            Variable(x), Op(>), Int(2), Punct(':'), Space(Newline),
            Space(Indent), Call(print), Punct('('), Str("Hi"), Punct(')')
          ]
        }
        ↓
        to_code() → "def f(x):\n    if x > 2:\n        print('Hi')"
```

The model learns to TRANSFORM graph topology, not reproduce characters.

## Related Tasks

- `api-021`: Add `DomainBrain::node_types()` trait method
- `backend-215`: Implement node_types for all domain brains
- `backend-216`: Build semantic node decoder with unified vocab
- `backend-214`: Unified Semantic Node Vocabulary

## References

- `GRAPHEME_Vision.md` - Core architecture specification
- `grapheme-train/src/bin/train_semantic_code.rs` - Correct training approach
- `grapheme-core/src/lib.rs` - NodeType enum definitions
