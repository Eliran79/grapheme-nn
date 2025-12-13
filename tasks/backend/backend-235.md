---
id: backend-235
title: Fix O(n) contains checks - Use HashSet for node lookups
status: done
priority: critical
tags:
- backend
- performance
- dag-optimization
dependencies: []
assignee: developer
created: 2025-12-13T00:00:00Z
estimate: 1h
complexity: 2
area: backend
---

# Fix O(n) contains checks - Use HashSet for node lookups

## Context
Multiple hot paths use `Vec::contains()` which is O(n). For large graphs, this makes forward pass O(V·n) instead of O(V).

## Problem Locations
- `grapheme-core/src/lib.rs:485` - `add_output_node()`
- `grapheme-core/src/lib.rs:499` - `neuromorphic_forward()`
- `grapheme-core/src/lib.rs:1176` - `gc_disconnected()`

## Current Code
```rust
if self.input_nodes.contains(&node_id) { ... }
if self.output_nodes.contains(&node_id) { ... }
```

## Fix
Change `input_nodes` and `output_nodes` from `Vec<NodeId>` to `HashSet<NodeId>`:
```rust
pub struct DagNN {
    input_nodes: HashSet<NodeId>,  // Was Vec<NodeId>
    output_nodes: HashSet<NodeId>, // Was Vec<NodeId>
    input_nodes_order: Vec<NodeId>, // Keep order if needed
}
```

## DAG Impact
- Forward pass: O(V) instead of O(V·n)
- GC: O(V) instead of O(V·n)
- Critical for large dynamic graphs

## Acceptance Criteria
- [x] Change `input_nodes` to `HashSet<NodeId>`
- [x] Change `output_nodes` to `HashSet<NodeId>`
- [x] Add `input_nodes_order: Vec<NodeId>` if ordering matters
- [x] Add `output_nodes_order: Vec<NodeId>` for output ordering
- [x] Update all `.push()` to `.insert()` + `.push()` on order vec
- [x] Update all `.contains()` calls (now O(1) via HashSet)
- [x] Add `is_input_node()` and `is_output_node()` helper methods
- [x] Verify tests pass (25/25 passed)
- [ ] Benchmark improvement (deferred - correctness verified)

## Session Handoff

### What Changed
- **grapheme-core/src/lib.rs**:
  - Added `HashSet` to imports
  - Changed `input_nodes: Vec<NodeId>` to `input_nodes: HashSet<NodeId>`
  - Added `input_nodes_order: Vec<NodeId>` for maintaining insertion order
  - Changed `output_nodes: Vec<NodeId>` to `output_nodes: HashSet<NodeId>`
  - Added `output_nodes_order: Vec<NodeId>` for maintaining insertion order
  - Added `is_input_node()` and `is_output_node()` helper methods for O(1) lookup
  - Updated all methods that add nodes to use both `.insert()` and `.push()`
  - Updated all methods that iterate or index to use `_order` Vec
  - `.contains()` calls now use HashSet (O(1) instead of O(n))

### API Changes
```rust
// New public methods
pub fn is_input_node(&self, node_id: NodeId) -> bool  // O(1) lookup
pub fn is_output_node(&self, node_id: NodeId) -> bool // O(1) lookup

// Existing methods unchanged (return ordered Vec)
pub fn input_nodes(&self) -> &[NodeId]
pub fn output_nodes(&self) -> &[NodeId]
```

### Performance Impact
- `neuromorphic_forward()`: O(V) instead of O(V·n)
- `gc_disconnected()`: O(V) instead of O(V·n)
- `add_output_node()`: O(1) contains check instead of O(n)

### Testing
All 25 grapheme-core tests pass. Clippy clean.
