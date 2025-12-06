---
id: backend-026
title: Implement real reasoning algorithms (currently stubs)
status: todo
priority: high
tags:
- backend
- reasoning
- critical
dependencies: []
assignee: developer
created: 2025-12-06T00:00:00Z
estimate: ~
complexity: 4
area: backend
---

# Implement real reasoning algorithms (currently stubs)

## Context
Code review revealed that grapheme-reason implementations are mostly stubs/placeholders:
- Deduction: pattern_matches only compares node/edge counts
- Induction: returns empty TransformRule
- Abduction: returns observation as explanation
- Analogy: transfer() is no-op
- CausalReasoning: all methods return input unchanged

These need real implementations to be functional.

## Objectives
- Implement real pattern matching for deduction
- Implement actual rule induction from examples
- Implement meaningful abductive reasoning
- Complete analogical transfer
- Implement basic causal intervention/counterfactual

## Tasks

### Deduction (HIGH)
- [ ] Implement proper graph pattern matching (subgraph matching or WL-based)
- [ ] Implement real forward chaining with rule application
- [ ] Implement proper backward chaining proof search

### Induction (MEDIUM)
- [ ] Implement common subgraph extraction from examples
- [ ] Generate actual transformation rules from input-output pairs
- [ ] Add confidence scoring based on test examples

### Abduction (MEDIUM)
- [ ] Query background knowledge for potential causes
- [ ] Rank explanations by structural similarity
- [ ] Implement Occam's Razor (simpler = better)

### Analogy (HIGH)
- [ ] Implement actual knowledge transfer using mapping
- [ ] Apply mapped relations from source to target
- [ ] Improve mapping algorithm (Hungarian or at least feature-based)

### Causal Reasoning (LOW - complex)
- [ ] Implement intervention graph surgery (remove incoming edges)
- [ ] Implement basic counterfactual computation
- [ ] Add causal path detection for causes()

## Acceptance Criteria
✅ **Functionality:**
- Deduction can actually derive new facts from rules
- Induction produces usable transformation rules
- Abduction returns different explanations, not just observation
- Analogy transfer modifies the target graph
- Causal methods actually change graphs

✅ **Testing:**
- Each method has tests verifying non-trivial behavior
- Integration tests with realistic scenarios

## Technical Notes

### Pattern Matching Options
1. WL-based similarity (already in grapheme-train)
2. Subgraph isomorphism (expensive but accurate)
3. Feature vector comparison (fast but approximate)

### Induction Approach
Use anti-unification or version space learning to find common patterns.

### Causal Graph Surgery
```rust
fn intervene(&self, world: &Graph, do_action: &Graph) -> Graph {
    // 1. Find nodes in world that match do_action
    // 2. Remove incoming edges to those nodes (break causal chains)
    // 3. Set values according to do_action
    // 4. Propagate effects forward
}
```

## Files to Modify
- grapheme-reason/src/lib.rs: All Simple* implementations

## Updates
- 2025-12-06: Task created after code review revealed stubs

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- [Document actual implementations added]

### Verification & Testing
- [How to verify reasoning actually works]
