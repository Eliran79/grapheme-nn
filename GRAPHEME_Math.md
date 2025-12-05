# GRAPHEME-Math

## Solving Real-World Math Problems from Natural Language

*A layered architecture extending GRAPHEME for mathematical reasoning*

---

## Architecture Overview

```
Layer 4: Natural Language     "What's the integral of x² from 0 to 5?"
              │
              ▼
Layer 3: Math Brain           Intent: Integrate | Expr: x² | Bounds: [0,5]
              │
              ▼
Layer 2: Polish IR            (integrate (^ x 2) x 0 5)
              │
              ▼
Layer 1: Math Engine          → 125/3 (symbolic) or 41.667 (numeric)
```

---

## Layer Responsibilities

### Layer 1: `grapheme-engine`
**The ground truth.**

- Formal algebraic rules
- Symbolic manipulation
- Numerical evaluation
- Compiles to executable code
- Generates training data
- Validates all outputs

### Layer 2: `grapheme-polish`
**The intermediate representation.**

- Unambiguous Polish notation
- Direct graph mapping
- Optimization passes
- Bidirectional: text ↔ graph

### Layer 3: `grapheme-math`
**The learned brain.**

- Typed nodes: `Int`, `Float`, `Operator`, `Function`, `Symbol`
- Extracts math intent from semantic graphs
- Learns graph transformations
- Trained by Layer 1, validated by Layer 1

### Layer 4: `grapheme-core`
**The universal interface.**

- Pure GRAPHEME (character-level, vocabulary-free)
- Natural language understanding
- Any human language input
- Extracts intent + parameters

---

## Why This Design

| Decision | Rationale |
|----------|-----------|
| Character-level NL is wrong for math | `1` in `123` ≠ `1` in `1+2`; wastes learning on solved problems |
| Typed nodes in Layer 3 | Semantic types, not embeddings; still vocabulary-free |
| Formal engine as foundation | Ground truth for training; infinite correct examples |
| Polish notation IR | Natural graph structure; unambiguous; proven |

---

## Node Types (Layer 3)

```rust
pub enum MathNode {
    Integer(i64),
    Float(f64),
    Symbol(String),        // x, y, θ
    Operator(MathOp),      // +, -, *, /, ^, %
    Function(MathFn),      // sin, cos, log, integrate, derive
}
```

---

## Training Loop

```
Engine generates:  (+ (* 2 3) 4) → 10
Brain predicts:    (+ (* 2 3) 4) → ?
Loss:              graph_distance(predicted, expected)
```

The engine provides infinite verified training pairs. The brain learns to approximate transformations. All outputs are validated against the engine.

---

## Real-World Applications

| Domain | Input | Polish IR |
|--------|-------|-----------|
| Finance | "Mortgage payment: $400k, 6.5%, 30yr" | `(pmt (/ 0.065 12) 360 400000)` |
| Physics | "Ball falls for 3 seconds" | `(* 0.5 9.8 (^ 3 2))` |
| Calculus | "Derivative of x³ at x=2" | `(eval (derive (^ x 3) x) x 2)` |
| Statistics | "Std dev of 4,8,6,5,3" | `(std [4 8 6 5 3])` |

---

## Crate Structure

```
grapheme/
├── grapheme-core/       # Layer 4: Character-level NL
├── grapheme-math/       # Layer 3: Math brain (typed nodes)
├── grapheme-polish/     # Layer 2: Polish notation IR
├── grapheme-engine/     # Layer 1: Formal rules + execution
└── grapheme-train/      # Training infrastructure
```

---

## Implementation Order

1. **`grapheme-engine`** — Foundation; must be correct
2. **`grapheme-polish`** — IR layer; parsing + graph conversion
3. **`grapheme-train`** — Generate training data from engine
4. **`grapheme-math`** — Train the brain
5. **`grapheme-core`** — NL interface last

---

## Dataset

See [GRAPHEME_Math_Dataset.md](./GRAPHEME_Math_Dataset.md) for:
- Self-generating training data from engine
- 7-level curriculum (basic arithmetic → equation solving)
- External datasets for NL layer (GSM8K, MATH, MathQA)
- Data format and file structure

---

## Success Criteria

- [ ] Engine passes formal verification
- [ ] Polish IR ↔ Graph is lossless
- [ ] Brain achieves >99% accuracy on engine-validated test set
- [ ] NL layer handles ambiguity gracefully
- [ ] End-to-end: English → verified numeric/symbolic result

---

*"Formal foundations. Learned intelligence. Universal interface."*