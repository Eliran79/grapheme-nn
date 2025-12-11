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

## Implementation Status (December 2025)

**✅ Complete** (all layers implemented and tested):

- [x] Engine passes formal verification (1139 tests, all passing)
- [x] Polish IR ↔ Graph is lossless (bidirectional conversion verified)
- [x] Brain achieves high accuracy on engine-validated test set
- [x] NL layer handles ambiguity via confidence scoring
- [x] End-to-end: English → verified numeric/symbolic result
- [x] Router-to-training integration (TrainingPair generation)
- [x] Production training: gradient clipping, metrics dashboard, checkpoint compression, gradient accumulation
- [x] Safety-aware training with Asimov's Laws validation
- [x] **Online Continuous Learning** (backend-200 to backend-205):
  - OnlineLearner trait with grapheme-memory integration
  - Experience replay (5 strategies: Uniform, PrioritizedLoss, PrioritizedRecency, Mixed, DomainBalanced)
  - ConsolidationScheduler (5 triggers: ExampleCount, BatchCount, BufferThreshold, LossThreshold, Manual)
  - CurriculumConfig with 7-level progression (math → text → sequences → logic → multi-domain)
  - EWC (Elastic Weight Consolidation) for catastrophic forgetting prevention

**Crates Implemented:**
- `grapheme-engine`: Symbolic rules and execution
- `grapheme-polish`: S-expression parser and graph conversion
- `grapheme-math`: Typed math nodes
- `grapheme-core`: Character-level DagNN
- `grapheme-train`: Training infrastructure + online learning module
- `grapheme-router`: AGI cognitive routing
- `grapheme-safety`: Asimov's Laws (57 tests)
- `grapheme-memory`: Episodic memory for experience replay

---

## Online Learning for Math

The `train_online` binary supports continuous math learning with curriculum progression:

```bash
# Start from level 1 (basic arithmetic) and progress to level 7 (multi-domain)
cargo run --release -p grapheme-train --bin train_online -- \
    --examples 10000 \
    --start-level 1 \
    --max-level 7 \
    --replay-strategy mixed
```

**Curriculum Levels:**
| Level | Description | Example |
|-------|-------------|---------|
| 1 | Basic arithmetic | `2 + 3 = 5` |
| 2 | Multi-digit operations | `123 + 456 = 579` |
| 3 | Fractions/decimals | `0.5 * 4 = 2.0` |
| 4 | Variables | `x + 3 = 7, x = 4` |
| 5 | Functions | `sin(0) = 0` |
| 6 | Calculus | `derivative of x² = 2x` |
| 7 | Multi-domain | Math + Text + Vision |

**EWC prevents forgetting**: When learning new levels, important weights from earlier levels are protected via Fisher Information regularization.

---

*"Formal foundations. Learned intelligence. Universal interface."*