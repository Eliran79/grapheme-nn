---
id: backend-238
title: Implement DynamicXavier weight recomputation on topology change
status: todo
priority: high
tags:
- backend
- weights
- dag-optimization
- protocol
dependencies:
- backend-232
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 3h
complexity: 4
area: backend
---

# Implement DynamicXavier weight recomputation on topology change

## Context
GRAPHEME Protocol requires DynamicXavier - weights must be recomputed when graph topology changes (nodes added/removed, edges modified). Currently edges are added but weights are not recalculated.

## Problem Locations
- `grapheme-core/src/lib.rs:531-532` - `add_edge()` doesn't update weights
- `grapheme-core/src/lib.rs:1116` - `strengthen_clique()` adds edges without weight update
- `grapheme-core/src/lib.rs:1185` - `gc_disconnected()` removes nodes but doesn't update neighbor weights

## Current Code (Missing Update)
```rust
pub fn add_edge(&mut self, source: NodeId, target: NodeId, weight: f32) {
    self.graph.add_edge(source, target, Edge::new(weight));
    // MISSING: Recompute weights for affected nodes!
}
```

## Fix
Add `update_weight()` method and call after topology changes:
```rust
/// Recompute edge weight based on current fan-in/fan-out (DynamicXavier)
fn update_edge_weight(&mut self, source: NodeId, target: NodeId) {
    if let Some(edge) = self.graph.find_edge(source, target) {
        let fan_in = self.graph.edges_directed(target, Incoming).count();
        let fan_out = self.graph.edges_directed(source, Outgoing).count();
        let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
        self.graph[edge].weight = scale;
    }
}

/// Update all weights for a node after topology change
fn update_node_weights(&mut self, node: NodeId) {
    // Update incoming edges
    let incoming: Vec<_> = self.graph.edges_directed(node, Incoming)
        .map(|e| e.source())
        .collect();
    for src in incoming {
        self.update_edge_weight(src, node);
    }
    // Update outgoing edges
    let outgoing: Vec<_> = self.graph.edges_directed(node, Outgoing)
        .map(|e| e.target())
        .collect();
    for tgt in outgoing {
        self.update_edge_weight(node, tgt);
    }
}

pub fn add_edge(&mut self, source: NodeId, target: NodeId, weight: f32) {
    self.graph.add_edge(source, target, Edge::new(weight));
    self.update_node_weights(source);
    self.update_node_weights(target);
}
```

## DAG Impact
- Weight initialization becomes stale after topology changes
- Dynamic graph morphogenesis breaks weight scaling
- Violates GRAPHEME Protocol

## Acceptance Criteria
- [ ] Add `update_edge_weight()` method
- [ ] Add `update_node_weights()` method
- [ ] Call after `add_edge()`
- [ ] Call after `remove_node()` for neighbors
- [ ] Call after `strengthen_clique()`
- [ ] Add tests verifying weight scales are correct after topology changes
- [ ] Document in CLAUDE.md
