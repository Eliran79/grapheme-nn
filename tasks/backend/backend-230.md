---
id: backend-230
title: Structural loss improvements for code graphs
status: done
priority: medium
tags:
- backend
- stage2
- loss
- code
dependencies:
- backend-229
assignee: developer
created: 2025-12-12T17:29:43.642894555Z
estimate: 3h
complexity: 5
area: backend
---

# Structural loss improvements for code graphs

## Context
The generic structural loss in graph_trainer.rs works for any DagNN but doesn't leverage code-specific semantics. Code graphs have rich type information (CodeNode variants, CodeEdge types) that can improve loss computation for code generation tasks.

## Objectives
- [x] Create code-aware structural loss functions
- [x] Support node type distribution comparison
- [x] Support edge type distribution comparison
- [x] Add control flow pattern matching
- [x] Add function signature comparison
- [x] All polynomial-time algorithms (no NP-hard)

## Tasks
- [x] Create code_loss.rs module with NodeCategory enum
- [x] Add EdgeCategory enum for edge type grouping
- [x] Implement node_type_histogram() for distribution comparison
- [x] Implement edge_type_histogram() for edge distribution
- [x] Implement control_flow_distance() for control flow patterns
- [x] Implement function_signature_distance() for function comparison
- [x] Implement depth_histogram() for tree depth analysis
- [x] Add operation_histogram() for operator type distribution
- [x] Add CodeLossConfig for configurable weights
- [x] Implement code_structural_loss() combining all metrics
- [x] Add dagnn_code_loss() for DagNN compatibility
- [x] Write 15 comprehensive tests
- [x] Integrate with lib.rs re-exports

## Acceptance Criteria
✅ **Criteria 1: Code-Aware Loss**
- CodeStructuralLoss provides detailed breakdown of node_type_loss, edge_type_loss, control_flow_loss, function_loss, depth_loss

✅ **Criteria 2: All Tests Pass**
- 15 tests in code_loss module all pass
- 84 total tests in grapheme-train pass

✅ **Criteria 3: Zero Warnings**
- cargo clippy -p grapheme-train -- -D warnings passes

## Technical Notes
- NodeCategory groups CodeNode variants: Structure, Function, Variable, Literal, Operation, ControlFlow, Call, Type, Comment
- EdgeCategory groups CodeEdge variants: Structural, ControlFlow, DataFlow, Type
- ControlFlowType tracks: If, LoopFor, LoopWhile, LoopLoop, Return
- OperationType tracks: Arithmetic, Comparison, Logical, Bitwise, Assignment
- All algorithms are O(n) or O(e) - no NP-hard computations
- Histograms are L1-normalized for scale invariance

## Testing
- [x] test_node_category - NodeCategory::from_node() works correctly
- [x] test_edge_category - EdgeCategory::from_edge() works correctly
- [x] test_node_type_histogram - histograms are normalized
- [x] test_edge_type_histogram - edge histograms work
- [x] test_histogram_distance_identical - identical histograms have zero distance
- [x] test_histogram_distance_different - different histograms have positive distance
- [x] test_code_structural_loss_identical - identical graphs have near-zero loss
- [x] test_code_structural_loss_different - different graphs have significant loss
- [x] test_control_flow_distance - control flow patterns compared correctly
- [x] test_function_signature_distance_identical - function signatures compared
- [x] test_depth_histogram - depth distribution computed correctly
- [x] test_operation_histogram - operation types counted
- [x] test_operation_distance - operation distributions compared
- [x] test_dagnn_code_loss - DagNN compatibility works
- [x] test_code_loss_config_default - default config is sensible

## Version Control
- [x] cargo build passes
- [x] cargo clippy -- -D warnings passes
- [x] cargo test passes (84 tests)

## Updates
- 2025-12-12: Task created
- 2025-12-13: Task completed - created code_loss.rs module with 15 tests

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- Created `/home/user/grapheme-nn/grapheme-train/src/code_loss.rs` (new file, ~730 lines)
- Added module declaration and re-exports to lib.rs
- Key exports: `CodeStructuralLoss`, `CodeLossConfig`, `NodeCategory`, `EdgeCategory`, `OperationType`, `code_structural_loss`, `dagnn_code_loss`, `operation_distance`

### Causality Impact
- `code_structural_loss(predicted, target, config)` → `CodeStructuralLoss` with detailed breakdown
- `dagnn_code_loss(predicted, target)` → `CodeStructuralLoss` for DagNN graphs (when CodeGraph not available)
- No async flows; all computations are synchronous O(n) or O(e)

### Dependencies & Integration
- Depends on: grapheme-code (CodeNode, CodeEdge, CodeGraph, BinaryOperator, UnaryOperator, LoopKind)
- Depends on: grapheme-core (DagNN)
- Integrates with: graph_trainer.rs can use code_structural_loss for code-specific training
- Affects: testing-019 (HumanEval benchmark integration) is now unblocked

### Verification & Testing
- Run `cargo test -p grapheme-train code_loss` to verify 15 tests pass
- Use `code_structural_loss()` for CodeGraph pairs during code training
- Use `dagnn_code_loss()` when only DagNN is available (e.g., converted graphs)

### Context for Next Task
- The loss functions provide weighted combination of 5 metrics (node_type, edge_type, control_flow, function, depth)
- Default weights prioritize function signatures (1.5) and control flow (1.2)
- All algorithms are polynomial-time; no graph isomorphism or NP-hard computations
- testing-019 depends on this task and can now use code-aware loss for HumanEval evaluation