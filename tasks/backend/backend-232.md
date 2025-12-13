---
id: backend-232
title: Implement Dynamic Xavier weight reinitialization
status: done
priority: high
tags:
- backend
- weights
- protocol
dependencies: []
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 2h
complexity: 3
area: backend
---

# Implement Dynamic Xavier weight reinitialization

## Context
GRAPHEME protocol requires **Dynamic Xavier** - weight scales are recomputed when
graph topology changes (nodes added/removed, edges modified).

Static Xavier initialization is unsuitable for dynamic DAGs where fan-in/fan-out
changes during graph morphogenesis.

## Objectives
- [x] Document Dynamic Xavier protocol in CLAUDE.md
- [x] Update semantic_decoder.rs comments
- [x] Update grapheme-vision InitStrategy documentation
- [x] grapheme-core uses dynamic √n normalization at activation time

## Tasks
- [x] Update `CLAUDE.md` with Dynamic Xavier formula
- [x] Update `grapheme-train/src/semantic_decoder.rs` comments
- [x] Update `grapheme-vision/src/lib.rs` InitStrategy docs
- [x] Verify dynamic √n normalization in grapheme-core

## Acceptance Criteria
✅ Documentation explains Dynamic Xavier
✅ All Xavier comments mention "Dynamic" or topology-aware
✅ Forward passes use dynamic √n normalization

## Technical Notes
```rust
// Dynamic Xavier: recompute when topology changes
fn update_weight(&mut self, edge: EdgeId) {
    let fan_in = self.incoming_edges(edge.target).count();
    let fan_out = self.outgoing_edges(edge.source).count();
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    self.edges[edge].weight *= scale;
}

// Dynamic √n normalization at activation time
let scale = 1.0 / (fan_in as f32).sqrt();
let activation = leaky_relu(scale * weighted_sum);
```

## Session Handoff
### What Changed
- `CLAUDE.md` - Added "GRAPHEME Protocol - Weight & Activation" section
- `grapheme-train/src/semantic_decoder.rs:101,114` - Updated comments
- `grapheme-vision/src/lib.rs:47-59` - Updated InitStrategy docs
- `grapheme-core/src/lib.rs:926-932` - Uses dynamic √n normalization

### Context for Next Task
- Future edge weight updates should call `update_weight()` when topology changes
- Consider adding `DagNN::reinitialize_weights()` method for full recomputation
