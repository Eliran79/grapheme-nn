# Backward Pass Analysis - Critical Architecture Gap

## Problem Statement

Backend-099 aims to implement backward pass through structural loss to model parameters. However, investigation reveals a **fundamental architecture disconnect**:

**The `GraphTransformNet` model is initialized but never used in the forward pass.**

## Current State (Broken Flow)

```rust
// Training loop (train.rs:323-350)
for (input, target) in inputs.iter().zip(targets.iter()) {
    // Convert text to graph (NO MODEL INVOLVEMENT)
    let predicted_graph = grapheme_core::GraphemeGraph::from_text(input);
    let target_graph = grapheme_core::GraphemeGraph::from_text(target);

    // Compute structural loss
    let loss_result = compute_structural_loss(&predicted_graph, &target_graph, &config);

    // TODO: Backpropagate gradients through graph structure
    // For now, the structural loss gradients are computed but not yet
    // connected to model parameter updates
}

// Update model weights (LINE 358)
model.step(lr);  // ← Updates weights that were NEVER USED!
```

**Result**: Loss is constant because the model doesn't influence predictions.

## Root Cause

`GraphemeGraph::from_text()` creates a fixed graph structure based purely on the input string:
- Nodes: One per character
- Edges: Sequential connections
- **No learned parameters involved**

The model (`GraphTransformNet`) has layers (embedding, message passing, attention) but they're never called.

## Required Solution

We need to connect the model to the graph transformation:

### Option 1: Feature-Based (Simpler)
Keep graph structure fixed, but learn **node features**:

```rust
// 1. Create initial graph structure
let mut graph = GraphemeGraph::from_text(input);

// 2. Model computes learned features for each node
let features = model.forward(&graph);  // Returns node features

// 3. Attach features to graph
graph.set_node_features(features);

// 4. Structural loss now compares BOTH structure AND features
let loss = compute_structural_loss(&graph, &target_graph, &config);

// 5. Backprop: loss → features → model parameters
let grads = loss_result.node_gradients;  // Gradients w.r.t. features
model.backward(grads);  // Backprop through layers
model.step(lr);  // Update weights
```

### Option 2: Structure-Based (Harder)
Model actually **modifies graph structure**:

```rust
// 1. Create initial graph
let input_graph = GraphemeGraph::from_text(input);

// 2. Model transforms the graph
let predicted_graph = model.transform_graph(&input_graph);  // Adds/removes edges/nodes

// 3. Structural loss on transformed graph
let loss = compute_structural_loss(&predicted_graph, &target_graph, &config);

// 4. Backprop through discrete graph operations (very hard!)
```

## Recommendation: Option 1 (Feature-Based)

**Why:**
1. **Structural loss already works** with node features (backend-096)
2. **Gradient flow is clear**: loss → features → model weights
3. **Matches GRAPHEME vision**: Model learns representations, structure emerges
4. **Incrementally deployable**: Can start with embedding layer only

**Implementation Plan:**

### Phase 1: Add Feature Storage to Graph
```rust
// grapheme-core/src/lib.rs
pub struct GraphemeGraph {
    // ... existing fields ...
    pub node_features: Option<HashMap<NodeId, Array1<f32>>>,  // NEW
}

impl GraphemeGraph {
    pub fn set_node_features(&mut self, features: HashMap<NodeId, Array1<f32>>) {
        self.node_features = Some(features);
    }
}
```

### Phase 2: Model Forward Pass
```rust
// grapheme-core/src/lib.rs - GraphTransformNet
impl GraphTransformNet {
    /// Compute learned features for graph nodes
    pub fn forward(&mut self, graph: &GraphemeGraph) -> HashMap<NodeId, Array1<f32>> {
        let mut features = HashMap::new();

        for &node_id in &graph.input_nodes {
            let node = &graph.graph[node_id];

            // Get character
            if let NodeType::Input(ch) = node.node_type {
                // Embed character
                let embedding = self.embedding.forward(ch);

                // TODO: Message passing, attention, etc.
                // For now, just use embedding
                features.insert(node_id, embedding);
            }
        }

        features
    }
}
```

### Phase 3: Update Structural Loss
Structural loss already uses node features in distance computation (Sinkhorn compares feature vectors). No changes needed if features are set.

### Phase 4: Backward Pass
```rust
// In training loop
let features = model.forward(&graph);
graph.set_node_features(features);

let loss_result = compute_structural_loss(&graph, &target_graph, &config);

// Backprop
for (node_id, feature) in features {
    if let Some(grad_slice) = loss_result.get_node_gradient(node_id) {
        let ch = graph.get_char(node_id);
        embedding.backward(ch as usize, &grad_slice);
    }
}

model.step(lr);
```

## Timeline Estimate

- **Phase 1** (Feature Storage): 1-2 hours
- **Phase 2** (Forward Pass): 2-3 hours
- **Phase 3** (Loss Integration): 1 hour
- **Phase 4** (Backward Pass): 3-4 hours
- **Testing & Validation**: 2-3 hours

**Total**: ~10-15 hours for full implementation

## Success Criteria

✅ Loss decreases monotonically in kindergarten training
✅ Model parameters visibly change (L2 norm delta > 0)
✅ Gradients pass finite difference check
✅ After 100 epochs, loss < 50% of initial

## Next Steps

1. Get user approval on Option 1 (feature-based) approach
2. Implement Phase 1 (feature storage)
3. Implement Phase 2 (forward pass)
4. Test forward pass produces reasonable features
5. Implement Phase 4 (backward pass)
6. Validate with kindergarten dataset
7. Mark backend-099 as DONE

## Open Questions

1. Should we implement all layers (message passing, attention) or start with embedding only?
2. How to handle edge features vs node features?
3. Should graph structure also be learned (add/remove edges)?
