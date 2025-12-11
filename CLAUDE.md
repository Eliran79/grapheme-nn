# CLAUDE.md - Project Context for Claude Code

## ⚠️ CRITICAL REQUIREMENTS FOR AI AGENTS

Before implementing ANY new features or algorithms, you MUST:

### 1. Read Vision Documents First
**ALWAYS read these files before implementing:**
- `GRAPHEME_Vision.md` - Main specification with API signatures and pseudocode
- `VISION_ARCHITECTURE.md` - Component distinctions and data flow
- `GRAPHEME_Math.md` - Mathematical reasoning extension (if doing math-related work)

These documents define the architecture. Implementations that don't follow them will be rejected.

### 2. P-Time Complexity Only (No NP-Hard Algorithms)
**ALL algorithms MUST be polynomial time (P-time):**
- Use O(n), O(n log n), O(n²), O(n³) algorithms
- NEVER use exponential-time algorithms (O(2^n), O(n!))
- NEVER implement NP-hard solutions (TSP, SAT, clique finding, etc.)
- For graph algorithms: BFS (O(V+E)), not backtracking/exhaustive search
- For pattern matching: Use linear-time algorithms, not exponential regex

**Examples of ALLOWED algorithms:**
- BFS/DFS for graph traversal: O(V+E)
- Cosine similarity for pattern matching: O(n)
- Merge sort / quicksort: O(n log n)
- Dynamic programming (when polynomial): varies

**Examples of FORBIDDEN algorithms:**
- Backtracking without pruning bounds
- Exhaustive clique enumeration
- NP-complete problem solvers
- Exponential combinatorial searches

If unsure about complexity, ask or use a simpler algorithm.

---

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
- **Routing**: `grapheme-router` - AGI cognitive routing with training integration

### Key Documentation
- `GRAPHEME_Vision.md` - Main specification (API signatures, pseudocode)
- `GRAPHEME_Math.md` - Mathematical reasoning extension
- `GRAPHEME_Math_Dataset.md` - Dataset generation strategy
- `GRAPHEME_Technical_Abstract.md` - Academic abstract
- `TRAINING_STRATEGY.md` - Local training guide
- `VISION_ARCHITECTURE.md` - Vision component architecture

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
5. **Router-to-training integration** - CognitiveRouter generates TrainingPair (input_graph, output_graph) for AGI training

## Project Stats (December 2025)

- **Tests**: 1200 passing, zero warnings
- **Tasks**: 224 total (210 done, 14 planned)
- **LOC**: 70K+ Rust code
- **Crates**: 22 modules

### Recently Completed (AGI Roadmap)
- ✅ Text/Web Learning: File ingestion, web fetcher, HTML parser, preprocessing
- ✅ G2G Transformation: Graph-to-graph learning with structural loss
- ✅ LLM API Client: Claude, OpenAI, Gemini, Ollama unified interface
- ✅ MCP Server: 5 GRAPHEME tools (graph_from_text, query, transform, to_text, compare)
- ✅ A2A Protocol: Agent discovery, task lifecycle, 4 skills

### Remaining Tasks (14)
- A2A message format, multi-agent orchestration
- LLM bidirectional graph↔prompt translation
- Graph morphism detection, efficient serialization
- Integration tests for new capabilities

## grapheme-train Modules

The `grapheme-train` crate contains the following modules:

| Module | Description | Tasks |
|--------|-------------|-------|
| `text_ingestion` | File ingestion (TXT, MD, JSON) | backend-169 |
| `web_fetcher` | HTTP content fetching | backend-170 |
| `html_parser` | HTML parsing and extraction | data-002 |
| `text_preprocessor` | Text cleaning and chunking | data-001 |
| `g2g` | Graph-to-graph transformation | backend-175 |
| `llm_client` | LLM API client (Claude, OpenAI, Gemini, Ollama) | integration-001 |
| `mcp_server` | MCP protocol server with 5 tools | api-017 |
| `a2a_protocol` | A2A agent protocol with 4 skills | api-015 |

### Training Binaries
- `train` - Main curriculum training
- `train_from_text` - Train from text files (backend-171)
- `train_from_web` - Train from web content (backend-172)
- `generate` - Dataset generation
- `repl` - Interactive REPL
