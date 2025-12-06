---
id: backend-014
title: Add clique size bounds validation
status: done
priority: medium
tags:
- backend
- validation
- safety
dependencies:
- backend-009
assignee: developer
created: 2025-12-05T21:52:58.768987970Z
estimate: ~
complexity: 2
area: backend
---

# Add clique size bounds validation

## Context
GRAPHEME's algorithms assume cliques are small (k=3-6). The `strengthen_clique()` method has O(k²) complexity, and clique enumeration has O(n^k) complexity. If cliques grow unexpectedly large, these operations become intractable.

**Risk**: Without bounds checking, a clique of size k=20 would require:
- `strengthen_clique()`: 190 edge operations (acceptable)
- `find_cliques()`: O(n^20) enumeration (catastrophic)

## Objectives
- Add runtime validation for clique sizes
- Prevent NP-hard complexity from manifesting
- Provide clear error messages when bounds violated
- Make assumptions explicit in code

## Tasks
- [ ] Define `MAX_CLIQUE_SIZE` constant (default: 10)
- [ ] Add validation in `form_clique()` method
- [ ] Add validation in `strengthen_clique()` method
- [ ] Add validation in `find_cliques()` (backend-009)
- [ ] Add `CliqueError` type for bound violations
- [ ] Add configuration option to override limit
- [ ] Add metrics/logging for clique sizes
- [ ] Write tests for boundary conditions

## Acceptance Criteria
✅ **Safety:**
- Clique operations reject k > MAX_CLIQUE_SIZE
- Clear error messages explain the limit
- No silent failures or hangs

✅ **Configurability:**
- Default limit of 10 (safe for all operations)
- Can be increased for specific use cases
- Debug logging shows clique size distribution

✅ **Performance:**
- Validation adds negligible overhead (O(1))
- No impact on normal operation (k ≤ 6)

## Technical Notes

### Constant Definition
```rust
/// Maximum allowed clique size to prevent NP-hard complexity.
/// GRAPHEME assumes k=3-6 for text concepts.
/// Higher values risk exponential blowup in enumeration.
pub const MAX_CLIQUE_SIZE: usize = 10;

#[derive(Debug, thiserror::Error)]
pub enum CliqueError {
    #[error("Clique size {0} exceeds maximum {1}")]
    SizeExceeded(usize, usize),
}
```

### Validation Points
```rust
impl GraphemeGraph {
    pub fn form_clique(&mut self, members: Vec<NodeId>, label: Option<String>)
        -> Result<usize, CliqueError>
    {
        if members.len() > MAX_CLIQUE_SIZE {
            return Err(CliqueError::SizeExceeded(members.len(), MAX_CLIQUE_SIZE));
        }
        // ... existing logic
    }

    pub fn strengthen_clique(&mut self, clique: &Clique, factor: f32)
        -> Result<(), CliqueError>
    {
        if clique.members.len() > MAX_CLIQUE_SIZE {
            return Err(CliqueError::SizeExceeded(clique.members.len(), MAX_CLIQUE_SIZE));
        }
        // ... existing logic
    }
}
```

### Metrics Collection
```rust
impl GraphemeGraph {
    /// Get statistics about clique sizes in the graph
    pub fn clique_stats(&self) -> CliqueStats {
        let sizes: Vec<usize> = self.cliques.iter()
            .map(|c| c.members.len())
            .collect();

        CliqueStats {
            count: sizes.len(),
            max_size: sizes.iter().max().copied().unwrap_or(0),
            avg_size: sizes.iter().sum::<usize>() as f32 / sizes.len().max(1) as f32,
            size_histogram: compute_histogram(&sizes),
        }
    }
}
```

### Files to Modify
- `grapheme-core/src/lib.rs`: Add constant, error type, validation
- `grapheme-core/src/lib.rs`: Add `clique_stats()` method

## Testing
- [ ] Test clique formation at boundary (k=10)
- [ ] Test rejection at k=11
- [ ] Test error message clarity
- [ ] Test metrics collection
- [ ] Verify existing tests still pass (k ≤ 6)

## Updates
- 2025-12-05: Task created from NP-hard gap analysis

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document code changes, new files, modified functions]

### Causality Impact
- Prevents NP-hard complexity from manifesting
- May cause errors in edge cases (cliques > 10)
- Provides observability into clique behavior

### Dependencies & Integration
- Should be done before backend-008/009 (clique algorithms)
- May need `thiserror` crate for error types
- Changes API to return Result types

### Verification & Testing
- Test with deliberately large cliques
- Verify normal operation unaffected