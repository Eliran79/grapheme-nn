---
id: backend-226
title: Autoencoder reconstruction loss and metrics
status: done
priority: high
tags:
- backend
- autoencoder
- stage1
- metrics
dependencies:
- backend-222
- backend-223
- backend-224
- backend-225
assignee: developer
created: 2025-12-12T17:29:31.901881168Z
estimate: 3h
complexity: 5
area: backend
---

# Autoencoder reconstruction loss and metrics

> **⚠️ SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**
>
> When you mark this task as `done`, you MUST:
> 1. Fill the "Session Handoff" section at the bottom with complete implementation details
> 2. Document what was changed, what runtime behavior to expect, and what dependencies were affected
> 3. Create a clear handoff for the developer/next AI agent working on dependent tasks
>
> **If this task has dependents,** the next task will be handled in a NEW session and depends on your handoff for context.

## Context
Unified metrics collection for autoencoder Stage 1 training evaluation.

## Objectives
- [x] Create AutoencoderMetrics struct for tracking reconstruction quality
- [x] Add evaluate_batch() method to GraphAutoencoder trait
- [x] Support metrics merging for parallel evaluation
- [x] Provide human-readable reporting (Display, summary())

## Tasks
- [x] Implement AutoencoderMetrics with loss, accuracy, perfect_rate, graph stats
- [x] Add evaluate_batch() to GraphAutoencoder trait
- [x] Add merge() for combining metrics from parallel evaluations
- [x] Add Display and summary() for reporting
- [x] Export AutoencoderMetrics from grapheme-brain-common
- [x] Write 8 unit tests for all metrics functionality

## Acceptance Criteria
✅ **Criteria 1:**
- AutoencoderMetrics tracks: samples, total_loss, perfect_count, total_nodes, total_edges, encoding_errors, decoding_errors

✅ **Criteria 2:**
- evaluate_batch() performs roundtrip on inputs and returns accumulated metrics

✅ **Criteria 3:**
- All 71 grapheme-brain-common tests pass (including 8 new metrics tests)

## Technical Notes
- Location: `grapheme-brain-common/src/autoencoder.rs` (lines 307-458)
- AutoencoderMetrics fields are public for direct access
- perfect_count increments when loss < 1e-6 OR original == reconstructed
- evaluate_batch() is a default trait method, can be overridden for parallel impl

## Testing
- [x] Write unit tests for new functionality (8 tests added)
- [x] Ensure all tests pass before marking task complete (71 passed)
- [x] Test edge cases (empty metrics, zero samples, error recording)

## Version Control
- [x] Commit: `50e73b2` - feat(autoencoder): add AutoencoderMetrics for Stage 1 evaluation
- [x] Pushed to claude/review-docs-switch-mKd25

## Updates
- 2025-12-12: Task created
- 2025-12-13: Task completed - AutoencoderMetrics implemented

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **File:** `grapheme-brain-common/src/autoencoder.rs` (+317 lines)
  - Added `AutoencoderMetrics` struct (lines 316-331)
  - Added impl with: `new()`, `record()`, `record_encoding_error()`, `record_decoding_error()`
  - Added metrics accessors: `avg_loss()`, `accuracy()`, `perfect_rate()`, `avg_nodes()`, `avg_edges()`, `error_count()`
  - Added `merge()` for combining metrics from parallel evaluation
  - Added `summary()` for full text report
  - Added `Display` impl for compact one-line format
  - Added `evaluate_batch()` to `GraphAutoencoder` trait (lines 306-339)
- **File:** `grapheme-brain-common/src/lib.rs` - Added `AutoencoderMetrics` to exports

### Causality Impact
- All methods are pure/functional - no side effects
- `evaluate_batch()` calls `encode()` → `decode()` → `reconstruction_loss()` for each input
- Errors are counted but don't stop batch processing
- Thread-safe for single-thread use; use `merge()` for parallel

### Dependencies & Integration
- Uses existing: `LatentGraph`, `AutoencoderError`, `DagNN`
- No new external dependencies
- Extends `GraphAutoencoder` trait with `evaluate_batch()` method
- Exported from `grapheme_brain_common`: `AutoencoderMetrics`

### Verification & Testing
```bash
# Build and test
cargo build --release -p grapheme-brain-common
cargo test -p grapheme-brain-common  # 71 tests pass

# Example usage
let brain = TextBrain::new();
let inputs: Vec<&str> = vec!["hello", "world", "test"];
let metrics = brain.evaluate_batch(&inputs);
println!("{}", metrics);  // Compact: loss=0.0000 acc=100.00% perfect=100.00% n=3 nodes=5.0 edges=4.0
println!("{}", metrics.summary());  // Full report
```

### Context for Next Task
- **For backend-227 (Graph-only training data format):**
  - Use `AutoencoderMetrics` to validate encoding quality before creating training pairs
  - `EncodedPair` already exists in autoencoder.rs for (input_graph, output_graph) pairs
  - Consider adding metrics to track graph transformation quality (not just reconstruction)

- **Key decisions:**
  - perfect_count uses epsilon (1e-6) OR exact string match
  - Metrics are additive via merge() for parallelism
  - Error counts separate from successful sample counts

- **Gotchas:**
  - `avg_loss()` returns 0.0 when samples=0 (not NaN)
  - `accuracy()` is 1.0 - avg_loss, so 100% when no samples
