---
id: testing-001
title: Review Test Strategy and Benchmarking Plan
status: done
priority: medium
tags:
- testing
dependencies:
- backend-001
- backend-002
- backend-003
- backend-004
- backend-005
assignee: developer
created: 2025-12-05T19:55:20.125474976Z
estimate: ~
complexity: 3
area: testing
---

# Review Test Strategy and Benchmarking Plan

> **SESSION WORKFLOW NOTICE (for AI Agents):**
>
> **This task should be completed in ONE dedicated session.**

## Context
Reviewed test coverage across all crates and enhanced the benchmark infrastructure to provide comprehensive performance measurements for the GRAPHEME project.

## Objectives
- [x] Review existing test coverage across all crates
- [x] Review existing benchmark infrastructure
- [x] Enhance benchmarks with comprehensive coverage
- [x] Verify all benchmarks compile and run correctly
- [x] All tests pass (106 tests)

## Tasks
- [x] Audit test counts per crate (106 total: core=25, polish=32, math=13, engine=20, train=16)
- [x] Fix train_bench.rs compilation errors (mutability, API usage)
- [x] Enhance engine_bench.rs with symbolic differentiation benchmarks
- [x] Enhance polish_bench.rs with parsing and conversion benchmarks
- [x] Enhance math_bench.rs with graph construction and conversion benchmarks
- [x] Enhance core_bench.rs with Unicode and scaling benchmarks
- [x] Enhance train_bench.rs with dataset and GED benchmarks
- [x] Verify all benchmarks run with `--test` flag

## Acceptance Criteria
**Test Coverage:**
- 106 tests total across 5 crates
- grapheme-core: 25 tests
- grapheme-polish: 32 tests
- grapheme-math: 13 tests
- grapheme-engine: 20 tests
- grapheme-train: 16 tests

**Benchmark Coverage:**
- engine_bench: 16 benchmarks (evaluation, symbolic, analysis, scaling)
- polish_bench: 16 benchmarks (parsing, conversion, round-trip, scaling)
- math_bench: 13 benchmarks (construction, conversion, operations, scaling)
- core_bench: 17 benchmarks (text, unicode, math, operations, scaling)
- train_bench: 13 benchmarks (generation, dataset, GED, scaling)

**Build & Test:**
- `cargo build` succeeds
- `cargo test` passes (106 tests)
- `cargo bench --bench <name> -- --test` succeeds for all 5 benches

## Technical Notes
- All benchmarks use criterion crate with BenchmarkId for parameterized tests
- Scaling benchmarks test performance at various depths/sizes
- Unicode benchmarks cover CJK, Arabic, emoji, and math symbols
- GED benchmarks use `compute_math()` for MathGraph comparison
- SymbolicEngine requires instance creation (`SymbolicEngine::new()`)
- Dataset::from_examples() takes name and examples parameters

## Testing
- [x] All 106 tests pass
- [x] All 5 benchmark suites compile and run

## Updates
- 2025-12-05: Task created
- 2025-12-05: Enhanced all benchmark files with comprehensive coverage

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### What Changed
- **grapheme-engine/benches/engine_bench.rs** - Enhanced from 2 to 16 benchmarks:
  - Basic evaluation: simple, nested, deeply nested (10 levels)
  - Function evaluation: sin, compound trig (sin^2 + cos^2)
  - Symbolic engine: differentiate (simple, polynomial, trig), substitute, evaluate_at
  - Expression analysis: builders, collect_symbols, depth, node_count
  - Scaling: evaluation at depths 5, 10, 20, 50

- **grapheme-polish/benches/polish_bench.rs** - Enhanced from 2 to 16 benchmarks:
  - Parsing: simple, nested, deeply nested, functions, compound, polynomial, symbols, floats, negative
  - Conversion: simple, nested, function, complex
  - Round-trip: parse and convert
  - Scaling: parse at depths 2, 4, 6, 8, 10

- **grapheme-math/benches/math_bench.rs** - Enhanced from 2 to 13 benchmarks:
  - Graph construction: simple, nested, deeply nested, functions, polynomial
  - Graph conversion: simple, nested, deeply nested
  - Round-trip: expr -> graph -> expr
  - Graph operations: node_count, edge_count
  - Scaling: construction and conversion at depths 5, 10, 20, 50

- **grapheme-core/benches/core_bench.rs** - Enhanced from 3 to 17 benchmarks:
  - Text to graph: short, medium, long
  - Unicode: basic, CJK, Arabic, emoji, math symbols, mixed
  - Mathematical text: expression, equation, calculus
  - Graph operations: node_count, edge_count
  - Scaling: text (1-50 repeats), unicode (1-50), word count (1-5)

- **grapheme-train/benches/train_bench.rs** - Fixed and enhanced from 2 to 13 benchmarks:
  - Fixed: mutability issues, API calls (from_examples, compute_math)
  - Data generation: level1, level2, level3 (symbols), level5 (diff), curriculum, from_spec
  - Dataset operations: creation, split, batch iterator, filter by level
  - Graph edit distance: simple, nested
  - Scaling: generation at sizes 10, 50, 100, 500

### Causality Impact
- Benchmarks are read-only and don't modify production code
- All benchmarks use `black_box` to prevent compiler optimizations
- Scaling benchmarks help identify performance bottlenecks

### Dependencies & Integration
- All benchmarks depend on criterion crate
- engine_bench uses SymbolicEngine (requires instance creation)
- train_bench uses Dataset, GraphEditDistance, CurriculumLevel from grapheme-train
- No new dependencies added

### Verification & Testing
```bash
cargo build        # Should succeed
cargo test         # 106 tests should pass
cargo bench --bench engine_bench -- --test   # Verify benchmarks run
cargo bench --bench polish_bench -- --test
cargo bench --bench math_bench -- --test
cargo bench --bench core_bench -- --test
cargo bench --bench train_bench -- --test
```

To run full benchmarks:
```bash
cargo bench        # Run all benchmarks
cargo bench --bench engine_bench   # Run specific benchmark suite
```

### Context for Next Task
- **testing-002** can proceed - benchmark infrastructure is complete
- Key benchmarks now exist for all major components
- Scaling benchmarks help track performance characteristics
- Criterion reports are generated in `target/criterion/`
- Future work: Add more edge case benchmarks, memory benchmarks
