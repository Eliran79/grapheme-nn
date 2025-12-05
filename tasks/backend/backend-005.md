---
id: backend-005
title: 'Review grapheme-train: Training Infrastructure'
status: done
priority: medium
tags:
- backend
dependencies:
- backend-001
- backend-002
- backend-003
- backend-004
assignee: developer
created: 2025-12-05T19:54:55.516564884Z
estimate: ~
complexity: 3
area: backend
---

# Review grapheme-train: Training Infrastructure

> **SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context
Review and expand grapheme-train to align with GRAPHEME_Math_Dataset.md specification.
Training infrastructure provides curriculum-based data generation, dataset management,
and loss computation for the GRAPHEME neural network.

## Objectives
- [x] Review GRAPHEME_Math_Dataset.md specifications
- [x] Review current grapheme-train implementation
- [x] Add LevelSpec struct and OutputType enum
- [x] Expand DataGenerator for all curriculum levels
- [x] Add Dataset management with JSONL save/load
- [x] Add BatchIterator for training loops
- [x] Enhance GraphEditDistance with better metrics
- [x] Add comprehensive tests
- [x] All tests pass (106 tests total)

## Tasks
- [x] Add OutputType enum (Numeric, Symbolic, Both)
- [x] Add LevelSpec struct with curriculum level specifications (1-7)
- [x] Expand TrainingExample with id, expected_symbolic, bindings
- [x] Implement DataGenerator.generate_symbol_substitution (Level 3)
- [x] Implement DataGenerator.generate_differentiation (Level 5)
- [x] Add Dataset struct with metadata, save/load JSONL
- [x] Add Dataset.split() for train/val/test splits
- [x] Add BatchIterator for training loops
- [x] Enhance GraphEditDistance with node/edge insertion/deletion
- [x] Add weighted_loss() using TrainingConfig weights
- [x] Add validate_dataset() utility function
- [x] Add ValidationReport struct
- [x] Expand TrainingConfig with val_frequency, patience
- [x] Add EpochMetrics struct and Trainer.record_epoch()
- [x] Add Clone derive to MathEngine
- [x] Add 16 tests for backend-005 functionality

## Acceptance Criteria
**Level Specifications:**
- LevelSpec for all 7 curriculum levels
- OutputType distinguishes Numeric vs Symbolic vs Both

**Data Generation:**
- Level 1-4: Numeric examples with verified results
- Level 3: Symbol substitution with bindings
- Level 5: Differentiation with symbolic results

**Dataset Management:**
- JSONL save/load with TrainingExample serialization
- Train/val/test split functionality
- Batch iteration for training loops

**Build & Test:**
- `cargo build` succeeds
- `cargo test` passes (106 tests)

## Technical Notes
- LevelSpec matches GRAPHEME_Math_Dataset.md curriculum table
- DataGenerator uses simple LCG for deterministic pseudo-randomness
- Dataset.split() provides 80/10/10 default splits
- GraphEditDistance computes node/edge insertion/deletion costs separately
- ValidationReport tracks valid/invalid/error counts

## Testing
- [x] 16 new tests added for backend-005 functionality
- [x] All 106 tests pass

## Updates
- 2025-12-05: Task created
- 2025-12-05: Implemented training infrastructure expansion - 106 tests pass

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-train/src/lib.rs** - Major expansion with training infrastructure:

  **OutputType enum:**
  - `Numeric` - Numeric result only
  - `Symbolic` - Symbolic result only
  - `Both` - Both numeric and symbolic

  **LevelSpec struct:**
  - `level_1()` through `level_7()` - Curriculum level specifications
  - `all_levels()` - Get all 7 level specs
  - `by_level(n)` - Get specific level spec
  - Fields: ops, functions, max_depth, allow_symbols, output, samples

  **TrainingExample:**
  - Now has `id` field (e.g., "L2-00001")
  - `expected_result: Option<f64>` for numeric
  - `expected_symbolic: Option<Expr>` for symbolic
  - `bindings: Vec<(String, f64)>` for symbol values
  - `numeric()` and `symbolic()` constructors
  - `with_bindings()` builder method

  **DataGenerator:**
  - `generate_basic_arithmetic()` - Level 1
  - `generate_nested_operations()` - Level 2
  - `generate_symbol_substitution()` - Level 3 (NEW)
  - `generate_basic_functions()` - Level 4
  - `generate_differentiation()` - Level 5 (NEW)
  - `generate_from_spec()` - Generate from LevelSpec
  - Uses simple LCG for deterministic randomness

  **Dataset struct:**
  - `from_examples()` - Create from example list
  - `add()` - Add single example
  - `filter_by_level()` - Filter examples by level
  - `split(train_ratio, val_ratio)` - Train/val/test split
  - `save_jsonl()` / `load_jsonl()` - JSONL persistence
  - `batches()` - Batch iterator for training

  **BatchIterator:**
  - `new()` - Create iterator
  - `num_batches()` - Get batch count
  - Implements `Iterator` trait

  **GraphEditDistance:**
  - Expanded with insertion/deletion costs for both nodes and edges
  - `node_mismatch_cost`, `edge_mismatch_cost` fields
  - `weighted_loss(config)` - Compute weighted loss

  **TrainingConfig:**
  - Added `val_frequency` - Validation every N epochs
  - Added `patience` - Early stopping patience

  **EpochMetrics:**
  - `avg_loss` - Average loss for epoch
  - `examples_processed` - Count of examples
  - `val_accuracy` - Validation accuracy (optional)

  **Trainer:**
  - `validate_numeric()` - Validate numeric prediction
  - `validate_symbolic()` - Validate symbolic prediction (structural)
  - `validate_example()` - Validate full training example
  - `compute_accuracy()` - Compute accuracy on dataset
  - `record_epoch()` - Add metrics to history
  - `history()` - Get training history

  **Validation utilities:**
  - `validate_dataset()` - Validate entire dataset
  - `ValidationReport` - Total/valid/invalid/errors counts

- **grapheme-engine/src/lib.rs:**
  - Added `Clone` derive to `MathEngine`

### Causality Impact
- DataGenerator generates verified data (engine always provides correct results)
- Symbol substitution in Level 3 creates bound expressions
- Differentiation in Level 5 uses SymbolicEngine from grapheme-engine
- Dataset validation recomputes results to verify correctness
- Batch iteration is stateless (can be restarted)

### Dependencies & Integration
- Imports `SymbolicEngine` from grapheme-engine for differentiation
- `MathEngine` now cloneable for validation with different bindings
- Dataset serialization uses serde_json for JSONL format
- File I/O uses std::fs and std::io

### Verification & Testing
```bash
cargo build        # Should succeed
cargo test         # 106 tests should pass
```

New tests (16 added):
- `test_data_generation_level1` - Level 1 basic arithmetic
- `test_data_generation_level2` - Level 2 nested operations
- `test_data_generation_level3_symbols` - Level 3 symbol substitution
- `test_data_generation_level5_differentiation` - Level 5 differentiation
- `test_curriculum_generation` - Multi-level curriculum
- `test_level_spec` - LevelSpec creation and properties
- `test_graph_edit_distance` - GED computation
- `test_dataset_creation` - Dataset from examples
- `test_dataset_split` - Train/val/test split
- `test_batch_iterator` - Batch iteration
- `test_validation_report` - Dataset validation
- `test_trainer_validation` - Numeric validation
- `test_trainer_example_validation` - Example validation
- `test_trainer_accuracy` - Accuracy computation
- `test_output_type` - OutputType enum values
- `test_ged_weighted_loss` - Weighted loss computation

### Context for Next Task
- **testing-001** and **testing-002** can now proceed
- Key new types: `LevelSpec`, `OutputType`, `Dataset`, `BatchIterator`, `ValidationReport`, `EpochMetrics`
- DataGenerator now supports levels 1-5 (6-7 are placeholders)
- Dataset provides complete persistence and batching infrastructure
- Validation utilities ensure data correctness
- Training loop infrastructure ready (config, metrics, history)
- Future work: Level 6 (integration), Level 7 (equation solving), full GED with node alignment
