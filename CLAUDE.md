# CLAUDE.md - Project Context for Claude Code

## Project Overview

**GRAPHEME** (Graph Representation through Adaptive Pattern Hierarchy and Emergent Modular Encoding) is a revolutionary neural architecture that eliminates tokenization and processes text directly at the character level using dynamic graphs.

## TaskGuard - Task Management

This project uses **TaskGuard v0.3.0** for task management.

### Quick Reference
```bash
# List all tasks
taskguard list

# See what's ready to work on
taskguard validate

# Update task status
taskguard update status <task-id> doing    # Start working
taskguard update status <task-id> done     # Complete task

# Create new task
taskguard create --title "Task name" --area backend --priority high

# Set dependencies
taskguard update dependencies <task-id> "dep1,dep2"
```

### Task Areas
- `setup` - Project initialization, Cargo workspace
- `api` - Core data structures, trait design
- `backend` - Layer implementations (core, math, polish, engine, train)
- `testing` - Test strategy, benchmarks, dataset generation

### Current Task Structure
```
setup-001 (Cargo workspace)
    └── api-001, api-002 (Data structures, Traits)
            └── backend-001 to 004 (Layer reviews)
                    └── backend-005 (Training)
                    └── testing-001, testing-002 (Tests, Datasets)
```

## Project Architecture

### Layered Structure (from GRAPHEME_Math.md)
- **Layer 4**: `grapheme-core` - Character-level NL processing
- **Layer 3**: `grapheme-math` - Math brain with typed nodes
- **Layer 2**: `grapheme-polish` - Polish notation intermediate representation
- **Layer 1**: `grapheme-engine` - Formal math rules and execution
- **Training**: `grapheme-train` - Training infrastructure

### Key Documentation
- `GRAPHEME_Vision.md` - Main specification (API signatures, pseudocode)
- `GRAPHEME_Math.md` - Mathematical reasoning extension
- `GRAPHEME_Math_Dataset.md` - Dataset generation strategy
- `GRAPHEME_Technical_Abstract.md` - Academic abstract

## Development Notes

### Build Commands (once Cargo.toml exists)
```bash
cargo build
cargo test
cargo run
```

### TaskGuard Installation
TaskGuard binary is installed at `~/.local/bin/taskguard` (added to PATH in ~/.bashrc).

If not in PATH, use:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Key Design Decisions

1. **No tokenization** - Character-to-node mapping, universal language support
2. **Dynamic graph morphogenesis** - Network topology adapts to input complexity
3. **Graph-to-graph transformations** - Structural alignment loss instead of cross-entropy
4. **Rust implementation** - Required for memory efficiency (17 bytes vs 150 bytes per node in Python)
