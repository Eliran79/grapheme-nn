# GRAPHEME-Math Dataset

## Training Data Specification

---

## Generation Strategy

Layer 1 (Engine) generates its own verified training data. Infinite, correct, complexity-controlled.

```rust
fn generate_pair(level: Level) -> (Expression, Result) {
    let expr = random_valid_expression(level);
    let result = engine.evaluate(expr);  // Always correct
    (expr, result)
}
```

---

## Curriculum Levels

| Level | Example | Output | Skills |
|-------|---------|--------|--------|
| 1 | `(+ 2 3)` | `5` | Single binary op |
| 2 | `(+ (* 2 3) 4)` | `10` | Nested ops |
| 3 | `(+ x 3)` [x=2] | `5` | Symbol substitution |
| 4 | `(sin 0)` | `0` | Basic functions |
| 5 | `(derive (^ x 2) x)` | `(* 2 x)` | Symbolic output |
| 6 | `(integrate (^ x 2) x 0 1)` | `0.333` | Calculus |
| 7 | `(solve (= (+ (* 2 x) 5) 13) x)` | `4` | Equation solving |

---

## Level Specifications

```rust
struct LevelSpec {
    ops: Vec<MathOp>,
    functions: Vec<MathFn>,
    max_depth: usize,
    allow_symbols: bool,
    output: OutputType,  // Numeric | Symbolic | Both
    samples: usize,
}

const CURRICULUM: &[LevelSpec] = &[
    // Level 1: Basic arithmetic
    LevelSpec {
        ops: vec![Add, Sub],
        functions: vec![],
        max_depth: 1,
        allow_symbols: false,
        output: Numeric,
        samples: 10_000,
    },
    // Level 2: Nested arithmetic
    LevelSpec {
        ops: vec![Add, Sub, Mul, Div],
        functions: vec![],
        max_depth: 3,
        allow_symbols: false,
        output: Numeric,
        samples: 50_000,
    },
    // Level 3: Symbolic substitution
    LevelSpec {
        ops: vec![Add, Sub, Mul, Div, Pow],
        functions: vec![],
        max_depth: 3,
        allow_symbols: true,
        output: Numeric,
        samples: 50_000,
    },
    // Level 4: Functions
    LevelSpec {
        ops: vec![Add, Sub, Mul, Div, Pow],
        functions: vec![Sin, Cos, Tan, Log, Exp, Sqrt],
        max_depth: 3,
        allow_symbols: true,
        output: Numeric,
        samples: 100_000,
    },
    // Level 5: Symbolic differentiation
    LevelSpec {
        ops: vec![Add, Sub, Mul, Div, Pow],
        functions: vec![Derive],
        max_depth: 4,
        allow_symbols: true,
        output: Symbolic,
        samples: 100_000,
    },
    // Level 6: Integration
    LevelSpec {
        ops: vec![Add, Sub, Mul, Div, Pow],
        functions: vec![Integrate],
        max_depth: 4,
        allow_symbols: true,
        output: Both,
        samples: 100_000,
    },
    // Level 7: Equation solving
    LevelSpec {
        ops: vec![Add, Sub, Mul, Div, Pow],
        functions: vec![Solve],
        max_depth: 4,
        allow_symbols: true,
        output: Numeric,
        samples: 100_000,
    },
];
```

---

## Data Format

```json
{
  "id": "L2-00001",
  "level": 2,
  "polish": "(+ (* 2 3) 4)",
  "graph": {
    "nodes": [
      {"id": 0, "type": "op", "value": "+"},
      {"id": 1, "type": "op", "value": "*"},
      {"id": 2, "type": "int", "value": 2},
      {"id": 3, "type": "int", "value": 3},
      {"id": 4, "type": "int", "value": 4}
    ],
    "edges": [
      {"from": 0, "to": 1},
      {"from": 0, "to": 4},
      {"from": 1, "to": 2},
      {"from": 1, "to": 3}
    ]
  },
  "result_numeric": 10,
  "result_symbolic": null,
  "result_graph": {
    "nodes": [{"id": 0, "type": "int", "value": 10}],
    "edges": []
  }
}
```

---

## Mixed AGI Dataset

Multi-modal training data combining math, text, timeseries, and vision for unified AGI training.

| Domain | Examples | Description |
|--------|----------|-------------|
| Math | 400 | Polish notation arithmetic/algebra |
| Text | 400 | Character-level Q&A pairs |
| Timeseries | 400 | Sequence prediction patterns |
| Vision | 400 | Feature graph classification |

**Location:** `data/mixed_agi/` (train: 1600, val: 200, test: 200)

**Format:** Same JSON structure with `domain` field for routing.

---

## External Datasets (NL Layer)

For training Layer 4 (Natural Language → Math Intent):

| Dataset | Description | Size | Use |
|---------|-------------|------|-----|
| [GSM8K](https://github.com/openai/grade-school-math) | Grade school word problems | 8.5K | Basic NL extraction |
| [MATH](https://github.com/hendrycks/math) | Competition math | 12.5K | Complex reasoning |
| [MathQA](https://math-qa.github.io/) | Multi-step problems | 37K | Operation sequences |
| [MAWPS](https://github.com/sroy9/mawps) | Arithmetic word problems | 3.3K | Simple NL patterns |
| [DeepMind Math](https://github.com/google-deepmind/mathematics_dataset) | Synthetic algebra/calculus | 2M+ | Broad coverage |

### Vision Datasets

| Dataset | Description | Size | Use |
|---------|-------------|------|-----|
| [MNIST](http://yann.lecun.com/exdb/mnist/) | Handwritten digits | 70K | Basic vision training |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | Clothing items | 70K | Object classification |
| [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | Natural images | 60K | Complex features |

### Text/NLP Datasets

| Dataset | Description | Size | Use |
|---------|-------------|------|-----|
| [WikiText](https://huggingface.co/datasets/wikitext) | Wikipedia articles | 100M+ tokens | Language modeling |
| [OpenWebText](https://huggingface.co/datasets/openwebtext) | Web text corpus | 8M docs | General NL understanding |
| [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | Reading comprehension | 100K Q&A | Question answering |

---

## Dataset Loaders (Rust Implementation)

GRAPHEME includes production-ready dataset loaders in `grapheme-train/src/datasets/`:

### GSM8K Loader (`gsm8k.rs`)

```rust
use grapheme_train::datasets::{Gsm8kLoader, Gsm8kExample};

let loader = Gsm8kLoader::new("data/external/gsm8k");
let examples = loader.load("train")?;  // Load train split

// Convert to GRAPHEME training format
loader.to_grapheme_format(&examples, "data/gsm8k.jsonl")?;

// Get statistics
let stats = loader.stats(&examples);
println!("{}", stats);  // Total examples, avg lengths, etc.
```

### MATH Competition Loader (`math_comp.rs`)

```rust
use grapheme_train::datasets::{MathLoader, MathExample, MathCategory};

let loader = MathLoader::new("data/external/math");
let examples = loader.load("train")?;  // 7 categories: algebra, geometry, etc.

// Filter by category or difficulty
let algebra = loader.filter_by_category(&examples, "algebra");
let hard = loader.filter_by_level(&examples, 4, 5);  // Levels 4-5

// Automatic \boxed{} answer extraction
for ex in &examples {
    println!("Answer: {}", ex.final_answer);  // Extracted from solution
}
```

### SQuAD Loader (`squad.rs`)

```rust
use grapheme_train::datasets::{SquadLoader, SquadExample};

let loader = SquadLoader::new("data/external/squad");
let examples = loader.load("dev")?;  // Supports v1.1 and v2.0

// Filter by context length
let short = loader.filter_by_context_length(&examples, 0, 500);
let answerable = loader.filter_possible(&examples);  // Skip "impossible" questions

// Automatic difficulty calculation based on:
// - Context length
// - Answer position
// - Answer length
```

### Common Features

All loaders share:
- **JSONL export**: `to_grapheme_format()` for unified training
- **Statistics**: `stats()` returns detailed dataset metrics
- **Filtering**: By difficulty, category, length, etc.
- **Error handling**: Graceful handling of malformed entries

---

## NL Augmentation

Each generated expression gets NL variants:

```json
{
  "polish": "(+ (* 2 3) 4)",
  "nl_variants": [
    "What is 2 times 3 plus 4?",
    "Add 4 to the product of 2 and 3",
    "Calculate 2 × 3 + 4",
    "2 multiplied by 3, then add 4"
  ]
}
```

Generation methods:
- Template-based (reliable)
- Paraphrase models (diversity)
- Back-translation (robustness)

---

## Validation Split

| Set | Source | Purpose |
|-----|--------|---------|
| Train | Generated L1-L7 | Learn transformations |
| Val | Generated (held-out seeds) | Tune hyperparameters |
| Test-IID | Generated (held-out seeds) | In-distribution accuracy |
| Test-OOD | External datasets | Generalization |
| Test-Hard | Manual edge cases | Robustness |

---

## Edge Cases (Test-Hard)

```
// Numerical precision
(- 0.1 0.1 0.1)  // Floating point

// Large numbers  
(^ 2 64)

// Division edge cases
(/ 1 3)          // Repeating decimal
(/ 5 0)          // Error handling

// Symbolic edge cases
(derive (^ x x) x)           // x^x
(integrate (/ 1 x) x 1 e)    // ln

// Nested complexity
(+ (+ (+ (+ 1 1) 1) 1) 1)    // Deep nesting
```

---

## File Structure

```
data/
├── generated/
│   ├── level_1/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   ├── level_2/
│   └── ...
├── external/
│   ├── gsm8k/
│   ├── math/
│   └── ...
├── augmented/
│   └── nl_variants.jsonl
└── edge_cases/
    └── test_hard.jsonl
```

---

## Generation CLI

```bash
# Generate level 2 training data
cargo run --release -p grapheme-train --bin generate -- \
    --level 2 --samples 50000 --output data/generated/level_2/

# Generate all levels
cargo run --release -p grapheme-train --bin generate -- \
    --all-levels --output data/generated/

# Validate dataset integrity
cargo run --release -p grapheme-train --bin validate -- \
    --input data/generated/

# Train on generated data
cargo run --release -p grapheme-train --bin train -- \
    --config train_config.toml
```

---

*Self-generating, verified, curriculum-based. Core implementation complete (232/250 tasks done).*