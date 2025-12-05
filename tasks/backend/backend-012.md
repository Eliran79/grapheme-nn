---
id: backend-012
title: Optimize GC with HashSet for input_nodes
status: todo
priority: low
tags:
- backend
- performance
- optimization
dependencies: []
assignee: developer
created: 2025-12-05T21:52:49.847871450Z
estimate: ~
complexity: 1
area: backend
---

# Optimize GC with HashSet for input_nodes

## Context
The `gc_disconnected()` method in GraphemeGraph uses `Vec::contains()` to check if a node is an input node. This results in O(n) lookup per node, making the overall garbage collection O(n²) for n nodes.

**Current code** (`grapheme-core/src/lib.rs:1119`):
```rust
if !self.input_nodes.contains(&node_idx) {  // O(n) lookup!
    disconnected.push(node_idx);
}
```

## Objectives
- Replace `input_nodes: Vec<NodeIndex>` with `HashSet<NodeIndex>`
- Achieve O(1) lookup instead of O(n)
- Reduce GC complexity from O(n²) to O(n)

## Tasks
- [ ] Add `use std::collections::HashSet` if not present
- [ ] Change `input_nodes` field type from `Vec<NodeIndex>` to `HashSet<NodeIndex>`
- [ ] Update all methods that add to input_nodes (use `insert` instead of `push`)
- [ ] Update all methods that iterate over input_nodes
- [ ] Update methods that need ordered access (if any)
- [ ] Run tests to verify correctness
- [ ] Add benchmark to verify performance improvement

## Acceptance Criteria
✅ **Correctness:**
- All existing tests pass
- GC behavior unchanged (same nodes removed)

✅ **Performance:**
- O(1) lookup for input_nodes membership
- GC complexity reduced to O(n)

✅ **Compatibility:**
- No API changes for external callers
- Methods returning input_nodes may need to collect to Vec

## Technical Notes

### Before/After Code
```rust
// BEFORE
pub struct GraphemeGraph {
    input_nodes: Vec<NodeIndex>,
    // ...
}

// AFTER
pub struct GraphemeGraph {
    input_nodes: HashSet<NodeIndex>,
    // ...
}

// GC check becomes O(1)
if !self.input_nodes.contains(&node_idx) {
    disconnected.push(node_idx);
}
```

### Methods to Update
Search for `input_nodes` usage:
- `gc_disconnected()` - main beneficiary
- `from_text()` - adds nodes with `push` → use `insert`
- `form_cliques()` - iterates (still works with HashSet)
- `connect_relevant()` - iterates (still works with HashSet)
- Any method that expects ordered iteration

### Edge Cases
- If order of input_nodes matters anywhere, use `IndexSet` from `indexmap` crate
- Or maintain both Vec (for order) and HashSet (for lookup)

### Files to Modify
- `grapheme-core/src/lib.rs`: Change field type and update methods

## Testing
- [ ] Verify all 25 grapheme-core tests pass
- [ ] Verify GC removes same nodes as before
- [ ] Add benchmark comparing Vec vs HashSet lookup

## Updates
- 2025-12-05: Task created from NP-hard gap analysis

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]

### Causality Impact
- GC performance improved from O(n²) to O(n)
- No functional changes

### Dependencies & Integration
- May need `indexmap` crate if order matters
- Pure internal optimization, no API changes

### Verification & Testing
- Run `cargo test -p grapheme-core`
- Run GC benchmark before/after
