# GRAPHEME
### Graph Representation through Adaptive Pattern Hierarchy and Emergent Modular Encoding

*No vocabulary. No limits. Just understanding.*

Revolutionary neural architecture that processes text without tokenization, growing dynamic graphs from characters (graphemes) for true language understanding.

**Status**: 🚧 Active Research & Development
**Paper**: Coming 2025
**Language**: Rust

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGI CAPABILITIES                        │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ grapheme-    │ grapheme-    │ grapheme-    │ grapheme-      │
│ agent        │ meta         │ ground       │ multimodal     │
│ Goals/Plans  │ Self-Monitor │ Sensors/Act  │ Cross-Modal    │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                     COGNITIVE LAYER                          │
├─────────────────────────────┬───────────────────────────────┤
│ grapheme-memory             │ grapheme-reason               │
│ Episodic/Semantic/Working   │ Deduction/Induction/Causal    │
├─────────────────────────────┴───────────────────────────────┤
│                     WORLD MODEL                              │
├─────────────────────────────┬───────────────────────────────┤
│ grapheme-world              │ grapheme-parallel             │
│ State/Prediction/Dynamics   │ Parallel Graph Processing     │
├─────────────────────────────┴───────────────────────────────┤
│                   DOMAIN BRAIN PLUGINS                       │
├────────────┬────────────┬────────────┬────────────┬─────────┤
│ grapheme-  │ grapheme-  │ grapheme-  │ grapheme-  │grapheme-│
│ math       │ code       │ law        │ music      │ chem    │
│ Algebra    │ AST/Types  │ Legal      │ Theory     │Molecular│
├────────────┴────────────┴────────────┴────────────┴─────────┤
│                     FOUNDATION                               │
├─────────────────────────────┬───────────────────────────────┤
│ grapheme-core               │ grapheme-train                │
│ DagNN/Cliques/DomainBrain   │ Dataset/GED Loss/WL Kernel    │
├─────────────────────────────┼───────────────────────────────┤
│ grapheme-engine             │ grapheme-polish               │
│ Symbolic Math Rules         │ Polish Notation IR            │
└─────────────────────────────┴───────────────────────────────┘
```

## Crates (17 total, 25K+ LOC)

### Core Foundation
| Crate | Purpose |
|-------|---------|
| `grapheme-core` | Character-to-graph processing, DagNN, clique detection, DomainBrain trait |
| `grapheme-engine` | Symbolic math: evaluate, differentiate, integrate, solve |
| `grapheme-polish` | Polish notation intermediate representation |
| `grapheme-train` | Training: datasets, GED loss, WL kernel, BP2 approximation |

### Domain Brain Plugins
| Crate | Domain | Features |
|-------|--------|----------|
| `grapheme-math` | Mathematics | Typed math nodes, expression graphs, simplification |
| `grapheme-code` | Source Code | AST nodes, language detection (Rust, Python, JS, C) |
| `grapheme-law` | Legal | Citations, statutes, IRAC analysis, stare decisis |
| `grapheme-music` | Music Theory | Notes, chords, scales, voice leading |
| `grapheme-chem` | Chemistry | Elements, molecules, bonds, reactions |

### Cognitive Modules
| Crate | Purpose |
|-------|---------|
| `grapheme-memory` | Episodic, semantic, working, procedural memory |
| `grapheme-reason` | Deduction, induction, abduction, analogy, causal reasoning |
| `grapheme-world` | World model: state, prediction, dynamics |
| `grapheme-parallel` | Parallel graph operations (rayon) |
| `grapheme-multimodal` | Cross-modal binding and fusion |
| `grapheme-meta` | Meta-cognition: uncertainty, resource allocation |
| `grapheme-agent` | Agency: goals, planning, value functions |
| `grapheme-ground` | Grounding: sensors, actuators, embodiment |

## Plugin Architecture

Domain brains are pluggable modules that extend GRAPHEME's capabilities to specific domains:

```rust
pub trait DomainBrain: Send + Sync + Debug {
    fn domain_id(&self) -> &str;
    fn can_process(&self, input: &str) -> bool;
    fn parse(&self, input: &str) -> DomainResult<DagNN>;
    fn execute(&self, graph: &DagNN) -> DomainResult<ExecutionResult>;
    fn get_rules(&self) -> Vec<DomainRule>;
    fn generate_examples(&self, count: usize) -> Vec<DomainExample>;
}
```

Register and use domain brains:
```rust
let mut registry = BrainRegistry::new();
registry.register(Box::new(MathBrain::new()));
registry.register(Box::new(CodeBrain::new()));
registry.register(Box::new(LawBrain::new()));

// Route to appropriate brain
if let Some(brain) = registry.get_for_input("solve x^2 = 4") {
    let result = brain.execute(&graph)?;
}
```

## Quick Start

```bash
cargo build --workspace
cargo test --workspace   # 500+ tests
cargo clippy --workspace # 0 warnings
```

## Key Features

- **No Tokenization**: Character-level processing, universal language support
- **Dynamic Graphs**: Network topology adapts to input complexity
- **Graph-to-Graph Loss**: Structural alignment via GED, not cross-entropy
- **Polynomial Complexity**: WL kernel O(nmk), BP2 O(n²), bounded cliques
- **AGI Architecture**: Memory, reasoning, world model, agency layers
- **Plugin System**: Extensible domain brains (math, code, law, music, chemistry)
- **Learnable Modules**: All cognitive components support gradient-based learning

## Task Management

This project uses [TaskGuard](https://github.com/anthropics/taskguard) for task tracking:

```bash
taskguard list          # View all tasks
taskguard validate      # Check ready tasks
taskguard update status <id> done  # Mark complete
```

Current status: **98 tasks** (88 done, 10 pending)

## License

MIT
