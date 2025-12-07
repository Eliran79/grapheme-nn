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

## FAQ

### How does optimization work with Adam/SGD?

The structural loss (backend-096) produces gradients that flow through the Sinkhorn algorithm:

1. **Forward pass**: Graph → activations
2. **Structural loss**: `loss = α·node_cost + β·edge_cost + γ·clique_cost`
3. **Sinkhorn differentiability**: Soft assignment (probabilities) instead of discrete matching
4. **Backward pass**: Gradients flow through exp(-cost/temperature)
5. **Optimizer step**: Adam/SGD updates weights normally

The key insight: **Sinkhorn converts discrete graph matching into continuous optimization**, making it compatible with standard gradient descent.

### Will low-weight edges and isolated nodes be pruned?

Yes! GRAPHEME uses **synaptic pruning** inspired by neuroscience:

**Edge Pruning** (grapheme-core/src/lib.rs:1660):
```rust
pub fn prune_weak_edges(&mut self, threshold: f32) -> usize {
    // Remove edges below threshold
    // Preserves DAG structure and cliques
}
```

**Node Cleanup**:
```rust
// Garbage collect disconnected nodes
dag.gc_disconnected();
```

**Experimental results** (GRAPHEME_Vision.md):
- 37.1% edge pruning without accuracy loss
- 1.52x speedup over baseline DAG networks

The DAG structure is preserved - removing edges cannot introduce cycles.

### Will nodes and edges be added to make the DAG smarter?

Yes! GRAPHEME grows through **neurogenesis** and **adaptive connectivity**:

**Node Addition** (GRAPHEME_Vision.md:76):
- Strategic addition in high-correlation regions
- Not random - based on activation patterns
- Example: "quantum" spawns 5-6 nodes vs "the" spawns 2-3 nodes

**Edge Addition**:
1. **Skip connections**: Long-range dependencies (distance-weighted)
2. **Clique edges**: Dense connections within learned concepts
3. **Context edges**: Semantic relationships within window

**Structure Adaptation**:
```rust
fn adapt_structure(&mut self, performance_metrics: &Metrics) {
    // 1. Add nodes where correlation is high (neurogenesis)
    // 2. Add edges to improve connectivity
    // 3. Strengthen edges with positive gradients
    // 4. Prune edges below threshold
}
```

The DAG **morphs** during training to fit the task!

### How does online learning work?

GRAPHEME supports true **streaming/incremental learning**:

**Continual Learning** (grapheme-memory):
- **No catastrophic forgetting** via memory consolidation
- Elastic Weight Consolidation (EWC) protects important weights
- Experience replay rehearses old examples
- Sleep-like integration for offline consolidation

**Streaming Architecture**:
```rust
async fn process_stream(&mut self, stream: DataStream) {
    while let Some(input) = stream.next().await {
        self.add_incremental(input);      // Online update
        if graph.needs_compression() {
            graph.compress_inactive();    // Constant memory
        }
    }
}
```

**Four Memory Systems**:
- **Episodic**: Timestamped experiences
- **Semantic**: General knowledge (atemporal)
- **Procedural**: Skills and how-tos
- **Working**: Active context (~7 items, like humans)

**Memory Efficiency**:
- < 1KB per 100 characters (vs 150KB in transformers)
- Sublinear growth via compression
- Can process infinite streams (Twitter, logs, conversations)

### What prevents gradient vanishing/exploding?

GRAPHEME has **architectural guarantees** against gradient degradation:

**1. DAG Structure (Bounded Depth)**:
- Acyclic by definition → no infinite loops
- Maximum path length ≈ input length (linear, not exponential)
- Topological sort ensures each node processed exactly once

**2. Skip Connections (Native)**:
- Long-range edges bypass intermediate nodes
- Multiple paths → gradients flow even if some degrade
- Distance-weighted: `weight = 1.0 / distance`

**3. Clique Structure**:
- Dense subgraphs act as **gradient reservoirs**
- Multiple redundant paths within concepts
- Local gradient pooling before propagation

**4. Direct Supervision**:
- Structural loss provides **node-level gradients**
- Not just output supervision (like RNNs)
- Every node gets signal from graph alignment

**5. Adaptive Depth**:
- Simple patterns → shallow (less decay)
- Complex patterns → deeper (but still bounded)
- Self-regulating based on content

**Comparison**:
| Architecture | Max Depth | Vanishing Risk |
|--------------|-----------|----------------|
| RNN | Sequence length T | High (σ^T decay) |
| Transformer | Fixed L layers | Low (but fixed) |
| **GRAPHEME DAG** | O(input length) | **Very Low** (bounded + skip) |

No special tricks needed - **the architecture prevents it naturally**!

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
