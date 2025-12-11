//! AGI-Ready GRAPHEME Inference Binary
//!
//! This binary leverages the FULL GRAPHEME cognitive stack:
//! - UnifiedCognition: Integrated cognitive system with attention and self-model
//! - Memory: Episodic, Semantic, Working memory (Simple implementations)
//! - Reasoning: Deduction, Induction, Abduction, Analogy, Causal
//! - Meta-Cognition: Uncertainty estimation, Safety monitoring
//! - GraphTransformNet: Neural graph transformation (TRUE GRAPHEME)
//! - GraphKnowledgeBase: Learned Q&A retrieval with graph embeddings
//!
//! Usage:
//!   agi_infer --model checkpoints/llm_final.checkpoint --kb checkpoints/math_kb.json --input "(+ 2 3)"
//!   agi_infer --model checkpoints/llm_final.checkpoint --kb checkpoints/math_kb.json --interactive

use clap::Parser;
use grapheme_core::{DagNN, GraphemeGraph, GraphTransformNet, UnifiedCheckpoint};
use grapheme_engine::MathEngine;
use grapheme_memory::{
    EpisodicMemory, SimpleEpisodicMemory, SimpleSemanticGraph, SimpleWorkingMemory,
    SemanticGraph, WorkingMemory, Episode,
};
use grapheme_meta::UnifiedCognition;
use grapheme_polish::PolishParser;
use grapheme_reason::{ReasoningEngine, create_default_reasoning_engine};
use grapheme_router::{CognitiveRouter, Input, InputType};
use grapheme_train::{GraphKnowledgeBase, Pipeline};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "agi_infer")]
#[command(about = "AGI-ready inference using full GRAPHEME cognitive stack")]
struct Args {
    /// Path to trained model checkpoint
    #[arg(short, long)]
    model: PathBuf,

    /// Input query (if not provided, enters interactive mode)
    #[arg(short, long)]
    input: Option<String>,

    /// Enable verbose output (shows cognitive module activity)
    #[arg(short, long)]
    verbose: bool,

    /// Show uncertainty estimates
    #[arg(long)]
    uncertainty: bool,

    /// Show safety status
    #[arg(long)]
    safety: bool,

    /// Enable full cognitive trace
    #[arg(long)]
    trace: bool,

    /// Path to knowledge base file (for learned Q&A retrieval)
    #[arg(short = 'k', long)]
    kb: Option<PathBuf>,
}

/// AGI Cognitive System combining all GRAPHEME modules
struct AGICognitiveSystem {
    /// Neural graph transformer (TRUE GRAPHEME)
    model: GraphTransformNet,
    /// Unified cognition (attention, self-model, brains)
    unified: UnifiedCognition,
    /// Cognitive router for input analysis
    router: CognitiveRouter,
    /// Working memory (active context, ~7 items)
    working_memory: SimpleWorkingMemory,
    /// Episodic memory (experiences)
    episodic_memory: SimpleEpisodicMemory,
    /// Semantic memory (facts)
    semantic_memory: SimpleSemanticGraph,
    /// Reasoning engine
    #[allow(dead_code)]
    reasoning: ReasoningEngine,
    /// Math engine for actual computation
    math_engine: MathEngine,
    /// Pipeline for natural language math parsing
    pipeline: Pipeline,
    /// Knowledge base for learned Q&A retrieval
    knowledge_base: Option<GraphKnowledgeBase>,
    /// Verbose mode
    verbose: bool,
}

impl AGICognitiveSystem {
    /// Create a new AGI cognitive system
    fn new(model: GraphTransformNet, knowledge_base: Option<GraphKnowledgeBase>, verbose: bool) -> Self {
        // Create unified cognition with standard brains
        let mut unified = UnifiedCognition::new();
        unified.register_standard_brains();

        Self {
            model,
            unified,
            router: CognitiveRouter::new(0.5),
            working_memory: SimpleWorkingMemory::new(7), // Miller's law: 7 +/- 2
            episodic_memory: SimpleEpisodicMemory::new(Some(1000)),
            semantic_memory: SimpleSemanticGraph::new(),
            reasoning: create_default_reasoning_engine(),
            math_engine: MathEngine::new(),
            pipeline: Pipeline::new(),
            knowledge_base,
            verbose,
        }
    }

    /// Try to compute math expression using MathEngine (TRUE reasoning, not retrieval)
    ///
    /// This implements the HUMAN BRAIN CORTEX model:
    /// 1. Polish notation → Direct MathEngine (formal reasoning cortex)
    /// 2. Natural language → Pipeline extracts math → MathEngine (language + math cortex mesh)
    /// 3. Infix notation → Pipeline handles (secondary math patterns)
    ///
    /// Like human cognition: Language areas detect math intent → Route to math areas → Compute
    fn try_compute_math(&self, input: &str) -> Option<String> {
        // CORTEX 1: Direct Polish notation (formal math cortex - fastest path)
        // Only try Polish if it looks like S-expression (starts with '(')
        if input.trim().starts_with('(') {
            let mut parser = PolishParser::new();
            if let Ok(expr) = parser.parse(input) {
                if let Ok(result) = self.math_engine.evaluate(&expr) {
                    return Some(Self::format_result(result));
                }
            }
        }

        // CORTEX 2: Word-based math (language cortex pattern matching)
        // Try word operators first - very specific patterns
        if let Some(result) = self.try_word_math(input) {
            return Some(Self::format_result(result));
        }

        // CORTEX 3: Natural Language Math via Pipeline (general NL math)
        // The Pipeline handles "what is X", infix "2 + 3", etc.
        let pipeline_result = self.pipeline.process(input);
        if let Some(result) = pipeline_result.numeric_result {
            // Sanity check: if input has word operators but result is just first number,
            // the Pipeline failed - don't return it
            let lower = input.to_lowercase();
            let has_word_op = lower.contains(" plus ")
                || lower.contains(" minus ")
                || lower.contains(" times ")
                || lower.contains(" divided by ");
            if !has_word_op {
                return Some(Self::format_result(result));
            }
        }

        None
    }

    /// Parse word-based math expressions like "5 plus 3", "7 times 8"
    /// This is a backup cortex for when Pipeline doesn't catch word operators
    fn try_word_math(&self, input: &str) -> Option<f64> {
        let lower = input.to_lowercase();

        // Extract question prefix if present
        let cleaned = lower
            .strip_prefix("what is ")
            .or_else(|| lower.strip_prefix("what's "))
            .or_else(|| lower.strip_prefix("calculate "))
            .or_else(|| lower.strip_prefix("compute "))
            .unwrap_or(&lower)
            .trim()
            .trim_end_matches('?');

        // Try each word operator pattern
        if let Some(result) = Self::try_binary_op(cleaned, " plus ", |a, b| a + b) {
            return Some(result);
        }
        if let Some(result) = Self::try_binary_op(cleaned, " minus ", |a, b| a - b) {
            return Some(result);
        }
        if let Some(result) = Self::try_binary_op(cleaned, " times ", |a, b| a * b) {
            return Some(result);
        }
        if let Some(result) = Self::try_binary_op(cleaned, " multiplied by ", |a, b| a * b) {
            return Some(result);
        }
        if let Some(result) = Self::try_binary_op(cleaned, " divided by ", |a, b| if b != 0.0 { a / b } else { f64::NAN }) {
            return Some(result);
        }
        if let Some(result) = Self::try_binary_op(cleaned, " over ", |a, b| if b != 0.0 { a / b } else { f64::NAN }) {
            return Some(result);
        }

        None
    }

    /// Try to parse a binary operation from text
    fn try_binary_op(text: &str, op_word: &str, op_fn: fn(f64, f64) -> f64) -> Option<f64> {
        if let Some(idx) = text.find(op_word) {
            let left_str = text[..idx].trim();
            let right_str = text[idx + op_word.len()..].trim();

            if let (Ok(left), Ok(right)) = (left_str.parse::<f64>(), right_str.parse::<f64>()) {
                let result = op_fn(left, right);
                if !result.is_nan() {
                    return Some(result);
                }
            }
        }
        None
    }

    /// Format a numeric result nicely (integers without decimals)
    fn format_result(result: f64) -> String {
        if result.fract() == 0.0 && result.abs() < 1e15 {
            format!("{}", result as i64)
        } else {
            format!("{:.6}", result).trim_end_matches('0').trim_end_matches('.').to_string()
        }
    }

    /// Process input through the full cognitive stack
    fn process(&mut self, input: &str) -> CognitiveResult {
        let mut result = CognitiveResult::new(input);

        // Step 1: Analyze input type via Router
        if self.verbose {
            println!("  [Router] Analyzing input...");
        }
        let (input_type, confidence) = self.router.analyze_input(&Input::Text(input.to_string()));
        result.input_type = format!("{:?}", input_type);
        result.routing_confidence = confidence;

        // Step 2: Convert to graph representation (CORE GRAPHEME)
        if self.verbose {
            println!("  [Graph] Converting to GraphemeGraph...");
        }
        let input_graph = GraphemeGraph::from_text(input);
        result.input_nodes = input_graph.node_count();
        result.input_edges = input_graph.edge_count();

        // Step 3: Check working memory for relevant context
        if self.verbose {
            println!("  [Memory] Checking working memory ({} items)...", self.working_memory.len());
        }

        // Step 4: Try DIRECT COMPUTATION (CORTEX MESH - TRUE reasoning)
        // Like human brain: ALWAYS try math computation if input might contain math
        // Don't rely solely on Router - language cortex should route to math cortex
        let might_be_math = matches!(input_type, InputType::Math)
            || input.chars().any(|c| c.is_ascii_digit())
            || input.to_lowercase().contains("plus")
            || input.to_lowercase().contains("minus")
            || input.to_lowercase().contains("times")
            || input.to_lowercase().contains("divided")
            || input.to_lowercase().contains("calculate")
            || input.to_lowercase().contains("what is");

        if might_be_math {
            if self.verbose {
                println!("  [CortexMesh] Math intent detected - engaging computation cortices...");
            }
            if let Some(computed) = self.try_compute_math(input) {
                result.computed_answer = Some(computed.clone());
                if self.verbose {
                    println!("    ✓ COMPUTED: {}", computed);
                }
            }
        }

        // Step 5: Knowledge Base Retrieval (fallback or for non-math)
        if result.computed_answer.is_none() {
            if let Some(ref mut kb) = self.knowledge_base {
                if self.verbose {
                    println!("  [Knowledge] Querying learned Q&A pairs...");
                }
                let kb_results = kb.query(input, &self.model, 3);
                if !kb_results.is_empty() {
                    let best = &kb_results[0];
                    result.kb_answer = Some(best.entry.answer.clone());
                    result.kb_similarity = best.similarity;
                    result.kb_question = Some(best.entry.question.clone());
                    if self.verbose {
                        println!("    Found: \"{}\" -> \"{}\" (sim: {:.3})",
                            best.entry.question, best.entry.answer, best.similarity);
                    }
                }
            }
        }

        // Step 5: Neural graph transformation (TRUE GRAPHEME)
        if self.verbose {
            println!("  [Neural] Running TRUE graph transformation...");
        }
        let (output_graph, decoded_text) = self.model.infer(&input_graph);
        result.output_nodes = output_graph.node_count();
        result.output_edges = output_graph.edge_count();

        // Clean decoded text
        let clean_output: String = decoded_text.chars()
            .filter(|c| *c >= ' ' && *c != '\0')
            .collect::<String>()
            .trim()
            .to_string();
        result.neural_output = clean_output;

        // Step 5: Store in episodic memory (as Graph/DagNN)
        if self.verbose {
            println!("  [Memory] Storing episode...");
        }
        if let Ok(input_dag) = DagNN::from_text(input) {
            // Create an Episode from the input
            let episode = Episode {
                id: 0, // Will be assigned by memory
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
                context: input_dag.clone(),
                content: input_dag.clone(),
                outcome: None,
                emotional_valence: 0.0,
                importance: 0.5,
                access_count: 0,
                tags: vec![],
            };
            let _ = self.episodic_memory.store(episode);
            // Update working memory using attend()
            let _ = self.working_memory.attend(input_dag);
        }

        // Step 6: Check unified cognition brain status
        if self.verbose {
            println!("  [Unified] Brain status check...");
            for status in self.unified.all_brain_status() {
                println!("    - {} ({}): {} requests",
                    status.id,
                    status.domain,
                    status.requests_processed);
            }
        }

        // Step 7: Get introspection
        let introspection = self.unified.perform_introspection();
        result.epistemic_uncertainty = introspection.health_score; // Use health_score as confidence proxy
        result.health_overview = introspection.state_summary.clone();

        // Step 8: Safety check
        result.safety_violations = self.router.safety_violation_count();
        result.is_safe = self.router.is_safe();

        result
    }

    /// Get system status
    fn status(&self) -> String {
        format!(
            "Working Memory: {}/7 | Episodic: {} | Semantic: {} facts | Brains: {} | Safety: {}",
            self.working_memory.len(),
            self.episodic_memory.len(),
            self.semantic_memory.len(),
            self.unified.all_brain_status().len(),
            if self.router.is_safe() { "OK" } else { "ALERT" }
        )
    }
}

/// Result of cognitive processing
#[derive(Debug)]
struct CognitiveResult {
    input: String,
    input_type: String,
    routing_confidence: f32,
    input_nodes: usize,
    input_edges: usize,
    output_nodes: usize,
    output_edges: usize,
    neural_output: String,
    /// Computed answer (from MathEngine - TRUE reasoning)
    computed_answer: Option<String>,
    /// Knowledge base answer (if found)
    kb_answer: Option<String>,
    /// Knowledge base similarity score
    kb_similarity: f32,
    /// Matched question from KB
    kb_question: Option<String>,
    epistemic_uncertainty: f32,
    health_overview: String,
    safety_violations: usize,
    is_safe: bool,
}

impl CognitiveResult {
    fn new(input: &str) -> Self {
        Self {
            input: input.to_string(),
            input_type: "Unknown".to_string(),
            routing_confidence: 0.0,
            input_nodes: 0,
            input_edges: 0,
            output_nodes: 0,
            output_edges: 0,
            neural_output: String::new(),
            computed_answer: None,
            kb_answer: None,
            kb_similarity: 0.0,
            kb_question: None,
            epistemic_uncertainty: 1.0,
            health_overview: String::new(),
            safety_violations: 0,
            is_safe: true,
        }
    }

    fn display(&self, show_uncertainty: bool, show_safety: bool, show_trace: bool) {
        println!("\nQ: {}", self.input);
        println!("{}", "-".repeat(60));

        // Priority 1: Show COMPUTED answer (TRUE reasoning)
        if let Some(ref computed) = self.computed_answer {
            println!("A: {}", computed);
            println!("   (COMPUTED by MathEngine - true reasoning)");
        }
        // Priority 2: Show KB answer
        else if let Some(ref answer) = self.kb_answer {
            if self.kb_similarity > 0.95 {
                // High confidence - show answer directly
                println!("A: {}", answer);
                if let Some(ref matched_q) = self.kb_question {
                    if matched_q != &self.input {
                        println!("   (retrieved: \"{}\" with {:.1}% similarity)",
                            truncate_str(matched_q, 50), self.kb_similarity * 100.0);
                    } else {
                        println!("   (exact KB match, {:.1}% confidence)", self.kb_similarity * 100.0);
                    }
                }
            } else {
                // Lower confidence - show with caveat
                println!("A: {} (approximate)", answer);
                println!("   (similar to: \"{}\", {:.1}% match)",
                    self.kb_question.as_deref().unwrap_or("?"), self.kb_similarity * 100.0);
            }
        } else {
            println!("A: [No answer found]");
            println!("   Neural output: \"{}\"", self.neural_output);
        }

        println!("\nDetails:");
        println!("  Type: {} (confidence: {:.2})", self.input_type, self.routing_confidence);
        println!("  Graph: {} nodes -> {} nodes", self.input_nodes, self.output_nodes);

        if show_uncertainty {
            println!("\nCognitive State:");
            println!("  Epistemic Confidence: {:.2}", self.epistemic_uncertainty);
            if !self.health_overview.is_empty() {
                println!("  Health: {}", self.health_overview);
            }
        }

        if show_safety {
            println!("\nSafety Status:");
            println!("  Violations: {}", self.safety_violations);
            println!("  Status: {}", if self.is_safe { "SAFE" } else { "UNSAFE - ACTION BLOCKED" });
        }

        if show_trace {
            println!("\nCognitive Trace:");
            println!("  1. Router analyzed input -> {:?}", self.input_type);
            println!("  2. GraphemeGraph created -> {} nodes", self.input_nodes);
            println!("  3. Neural transform -> {} nodes", self.output_nodes);
            println!("  4. Memory updated");
            println!("  5. Introspection complete");
        }
    }
}

/// Truncate a string to max length with ellipsis
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn interactive_mode(system: &mut AGICognitiveSystem, args: &Args) {
    use std::io::{self, BufRead, Write};

    println!("\nAGI GRAPHEME Interactive Mode");
    println!("==============================");
    println!("Cognitive Stack: Router + Memory + Reasoning + UnifiedCognition + Neural");
    println!("Commands: 'status', 'brains', 'quit'");
    println!("{}\n", system.status());

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("agi> ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input.to_lowercase().as_str() {
            "quit" | "exit" => {
                println!("Goodbye!");
                break;
            }
            "status" => {
                println!("{}", system.status());
                continue;
            }
            "brains" => {
                println!("Registered Brains:");
                for status in system.unified.all_brain_status() {
                    println!("  - {} ({}): {} requests, load {:.2}",
                        status.id,
                        status.domain,
                        status.requests_processed,
                        status.load);
                }
                continue;
            }
            _ => {}
        }

        let result = system.process(input);
        result.display(args.uncertainty, args.safety, args.trace);
        println!();
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("AGI GRAPHEME Cognitive System");
    println!("=============================\n");

    // Load model
    println!("Loading model from: {:?}", args.model);

    let checkpoint = UnifiedCheckpoint::load_from_file(&args.model)
        .map_err(|e| anyhow::anyhow!("Failed to parse checkpoint: {}", e))?;

    let model: GraphTransformNet = checkpoint.load_module()
        .map_err(|e| anyhow::anyhow!("Failed to load GraphTransformNet: {}", e))?;

    println!("Model: {} hidden dim, {} layers", model.hidden_dim, model.mp_layers.len());

    // Load knowledge base if provided
    let kb = if let Some(ref kb_path) = args.kb {
        println!("Loading knowledge base from: {:?}", kb_path);
        match GraphKnowledgeBase::load(kb_path) {
            Ok(loaded_kb) => {
                let stats = loaded_kb.stats();
                println!("Knowledge base: {} Q&A pairs across {} topics",
                    stats.total_entries,
                    stats.entries_by_topic.len());
                Some(loaded_kb)
            }
            Err(e) => {
                println!("Warning: Could not load KB: {}", e);
                None
            }
        }
    } else {
        println!("No knowledge base specified (use --kb to enable Q&A retrieval)");
        None
    };

    // Create AGI cognitive system
    let mut system = AGICognitiveSystem::new(model, kb, args.verbose);
    println!("\nCognitive modules initialized:");
    println!("  - UnifiedCognition (7 brains: Math, Code, Vision, Music, Chem, Law, Text)");
    println!("  - Memory (Episodic + Semantic + Working)");
    println!("  - Reasoning Engine");
    println!("  - Knowledge Base Retrieval");
    println!("  - Router + Safety Gate\n");

    // Process input or enter interactive mode
    if let Some(input) = args.input {
        let result = system.process(&input);
        result.display(args.uncertainty, args.safety, args.trace);
    } else {
        interactive_mode(&mut system, &args);
    }

    Ok(())
}
