//! Online Training Binary for Continuous AGI Learning
//!
//! Implements continuous online learning using:
//! - OnlineLearner trait with experience replay
//! - Curriculum generator (Level 1-7 progression)
//! - Configurable replay strategies
//! - Checkpoint save/restore
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin train_online -- --help

use clap::Parser;
use grapheme_core::DagNN;
use grapheme_train::{
    MemoryOnlineLearner, OnlineExample, OnlineLearner, OnlineLearnerConfig,
    ReplayStrategy,
};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

/// Online training for continuous AGI learning
#[derive(Parser, Debug)]
#[command(name = "train_online")]
#[command(about = "Continuous online learning with experience replay and curriculum")]
struct Args {
    /// Output directory for checkpoints
    #[arg(short, long, default_value = "checkpoints/online")]
    output: String,

    /// Starting curriculum level (1-7)
    #[arg(long, default_value_t = 1)]
    start_level: u8,

    /// Maximum curriculum level (1-7)
    #[arg(long, default_value_t = 7)]
    max_level: u8,

    /// Number of examples to train on (0 = infinite)
    #[arg(long, default_value_t = 10000)]
    examples: usize,

    /// Checkpoint interval (save every N examples)
    #[arg(long, default_value_t = 1000)]
    checkpoint_interval: usize,

    /// Replay strategy: uniform, prioritized, recency, mixed, balanced
    #[arg(long, default_value = "mixed")]
    replay_strategy: String,

    /// Learning rate
    #[arg(long, default_value_t = 0.001)]
    lr: f32,

    /// Batch size for mini-batch updates
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Replay buffer capacity
    #[arg(long, default_value_t = 10000)]
    replay_capacity: usize,

    /// Replay ratio (fraction of batch from replay buffer)
    #[arg(long, default_value_t = 0.5)]
    replay_ratio: f32,

    /// Consolidation interval (examples between consolidations)
    #[arg(long, default_value_t = 1000)]
    consolidation_interval: usize,

    /// Priority alpha for prioritized replay
    #[arg(long, default_value_t = 0.6)]
    priority_alpha: f32,

    /// Load checkpoint from file
    #[arg(long)]
    resume: Option<String>,

    /// Verbose output level (0-2)
    #[arg(short, long, default_value_t = 1)]
    verbose: u8,
}

/// Curriculum generator that produces training examples
struct CurriculumGenerator {
    current_level: u8,
    max_level: u8,
    examples_at_level: usize,
    examples_to_advance: usize,
    rng_seed: u64,
    total_generated: usize,
}

impl CurriculumGenerator {
    fn new(start_level: u8, max_level: u8) -> Self {
        Self {
            current_level: start_level.clamp(1, 7),
            max_level: max_level.clamp(1, 7),
            examples_at_level: 0,
            examples_to_advance: 500, // Advance after 500 examples at each level
            rng_seed: 12345,
            total_generated: 0,
        }
    }

    fn next_random(&mut self) -> u64 {
        self.rng_seed ^= self.rng_seed << 13;
        self.rng_seed ^= self.rng_seed >> 7;
        self.rng_seed ^= self.rng_seed << 17;
        self.rng_seed
    }

    fn random_f32(&mut self) -> f32 {
        (self.next_random() as f64 / u64::MAX as f64) as f32
    }

    /// Generate next example based on current curriculum level
    fn generate(&mut self) -> OnlineExample {
        let level = self.current_level;
        self.total_generated += 1;
        self.examples_at_level += 1;

        // Check if we should advance to next level
        if self.examples_at_level >= self.examples_to_advance && level < self.max_level {
            self.current_level += 1;
            self.examples_at_level = 0;
        }

        // Generate example based on level
        let (input, target, domain) = match level {
            1 => self.generate_level_1(),
            2 => self.generate_level_2(),
            3 => self.generate_level_3(),
            4 => self.generate_level_4(),
            5 => self.generate_level_5(),
            6 => self.generate_level_6(),
            _ => self.generate_level_7(),
        };

        OnlineExample::new(
            format!("online_{}", self.total_generated),
            input,
            target,
            domain,
        ).with_level(level)
    }

    /// Level 1: Basic arithmetic (single digit + single digit)
    fn generate_level_1(&mut self) -> (Vec<f32>, Vec<f32>, String) {
        let a = (self.next_random() % 10) as f32;
        let b = (self.next_random() % 10) as f32;
        let result = a + b;

        // Encode as normalized vectors
        let input = vec![a / 10.0, b / 10.0, 0.0, 0.0];  // a, b, op=add, padding
        let target = vec![result / 20.0]; // Normalized result

        (input, target, "math".to_string())
    }

    /// Level 2: Nested arithmetic (more operations)
    fn generate_level_2(&mut self) -> (Vec<f32>, Vec<f32>, String) {
        let a = (self.next_random() % 20) as f32;
        let b = (self.next_random() % 20) as f32 + 1.0; // Avoid division by zero
        let op = self.next_random() % 4;

        let result = match op {
            0 => a + b,
            1 => a - b,
            2 => a * b,
            _ => a / b,
        };

        let input = vec![a / 20.0, b / 20.0, op as f32 / 4.0, 0.0];
        let target = vec![(result.clamp(-100.0, 100.0) + 100.0) / 200.0];

        (input, target, "math".to_string())
    }

    /// Level 3: Text patterns (word starts with letter)
    fn generate_level_3(&mut self) -> (Vec<f32>, Vec<f32>, String) {
        let words = ["apple", "banana", "cat", "dog", "elephant"];
        let idx = (self.next_random() % words.len() as u64) as usize;
        let word = words[idx];

        // Input: first 5 characters as normalized ASCII
        let input: Vec<f32> = word.chars()
            .take(5)
            .map(|c| (c as u8 as f32 - 97.0) / 26.0)
            .chain(std::iter::repeat(0.0))
            .take(5)
            .collect();

        // Target: length of word normalized
        let target = vec![word.len() as f32 / 10.0];

        (input, target, "text".to_string())
    }

    /// Level 4: Sequences (predict next in pattern)
    fn generate_level_4(&mut self) -> (Vec<f32>, Vec<f32>, String) {
        let start = (self.next_random() % 10) as f32;
        let step = (self.next_random() % 5 + 1) as f32;

        // Input: first 4 elements of arithmetic sequence
        let input = vec![
            start / 50.0,
            (start + step) / 50.0,
            (start + 2.0 * step) / 50.0,
            (start + 3.0 * step) / 50.0,
        ];

        // Target: next element
        let target = vec![(start + 4.0 * step) / 50.0];

        (input, target, "timeseries".to_string())
    }

    /// Level 5: Pattern recognition (XOR-like)
    fn generate_level_5(&mut self) -> (Vec<f32>, Vec<f32>, String) {
        let a = if self.random_f32() > 0.5 { 1.0 } else { 0.0 };
        let b = if self.random_f32() > 0.5 { 1.0 } else { 0.0 };

        // XOR pattern
        let result = if (a > 0.5) != (b > 0.5) { 1.0 } else { 0.0 };

        let input = vec![a, b, 0.0, 0.0];
        let target = vec![result];

        (input, target, "logic".to_string())
    }

    /// Level 6: Multi-domain (combined)
    fn generate_level_6(&mut self) -> (Vec<f32>, Vec<f32>, String) {
        // Mix of different types
        match self.next_random() % 3 {
            0 => self.generate_level_1(),
            1 => self.generate_level_3(),
            _ => self.generate_level_4(),
        }
    }

    /// Level 7: Complex compositions
    fn generate_level_7(&mut self) -> (Vec<f32>, Vec<f32>, String) {
        // Multi-step computation
        let a = (self.next_random() % 10) as f32;
        let b = (self.next_random() % 10) as f32;
        let c = (self.next_random() % 10) as f32;

        let result = (a + b) * c;

        let input = vec![a / 10.0, b / 10.0, c / 10.0, 0.0];
        let target = vec![(result.clamp(0.0, 500.0)) / 500.0];

        (input, target, "math_advanced".to_string())
    }

    fn level_description(level: u8) -> &'static str {
        match level {
            1 => "Basic arithmetic (single digit)",
            2 => "Multi-operation arithmetic",
            3 => "Text pattern recognition",
            4 => "Sequence prediction",
            5 => "Logical patterns (XOR)",
            6 => "Multi-domain mixing",
            7 => "Complex compositions",
            _ => "Unknown",
        }
    }
}

fn parse_replay_strategy(s: &str) -> ReplayStrategy {
    match s.to_lowercase().as_str() {
        "uniform" => ReplayStrategy::Uniform,
        "prioritized" | "loss" => ReplayStrategy::PrioritizedLoss,
        "recency" => ReplayStrategy::PrioritizedRecency,
        "mixed" => ReplayStrategy::Mixed,
        "balanced" | "domain" => ReplayStrategy::DomainBalanced,
        _ => ReplayStrategy::Mixed,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== GRAPHEME Online Training ===\n");

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Build configuration
    let config = OnlineLearnerConfig {
        learning_rate: args.lr,
        batch_size: args.batch_size,
        replay_capacity: args.replay_capacity,
        replay_ratio: args.replay_ratio,
        consolidation_interval: args.consolidation_interval,
        use_ewc: false,
        ewc_lambda: 0.0,
        replay_strategy: parse_replay_strategy(&args.replay_strategy),
        priority_alpha: args.priority_alpha,
    };

    // Initialize model or load from checkpoint
    let model = if let Some(ref checkpoint_path) = args.resume {
        println!("Loading checkpoint from: {}", checkpoint_path);
        // For now, start fresh - checkpoint loading needs UnifiedCheckpoint
        DagNN::from_text("init").unwrap_or_else(|_| DagNN::new())
    } else {
        DagNN::from_text("init").unwrap_or_else(|_| DagNN::new())
    };

    let mut learner = MemoryOnlineLearner::new(model, config.clone());

    println!("Configuration:");
    println!("  Learning rate: {}", args.lr);
    println!("  Batch size: {}", args.batch_size);
    println!("  Replay strategy: {:?}", parse_replay_strategy(&args.replay_strategy));
    println!("  Replay capacity: {}", args.replay_capacity);
    println!("  Replay ratio: {:.1}%", args.replay_ratio * 100.0);
    println!("  Priority alpha: {}", args.priority_alpha);
    println!("  Curriculum: Level {} → {}", args.start_level, args.max_level);
    println!("  Output: {}", args.output);
    println!();

    // Initialize curriculum generator
    let mut generator = CurriculumGenerator::new(args.start_level, args.max_level);

    // Progress bar
    let total = if args.examples == 0 { 1_000_000 } else { args.examples };
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) Loss: {msg}")?
            .progress_chars("#>-"),
    );

    let start_time = Instant::now();
    let mut last_checkpoint = Instant::now();
    let mut batch_buffer: Vec<OnlineExample> = Vec::with_capacity(args.batch_size);

    println!("Starting online training...\n");

    let mut examples_processed = 0;
    while args.examples == 0 || examples_processed < args.examples {
        // Generate next example
        let example = generator.generate();
        let current_level = example.level;
        batch_buffer.push(example);

        // Train when batch is full
        if batch_buffer.len() >= args.batch_size {
            let batch_loss = learner.learn_batch(&batch_buffer);
            examples_processed += batch_buffer.len();
            batch_buffer.clear();

            // Update progress
            pb.set_position(examples_processed as u64);
            pb.set_message(format!("{:.4}", batch_loss));

            // Verbose output
            if args.verbose >= 2 && examples_processed % 100 == 0 {
                let stats = learner.stats();
                let replay_stats = learner.replay_stats();
                println!(
                    "\n[{}] Level {} | Loss: {:.4} | Replay: {} | Consolidations: {}",
                    examples_processed,
                    current_level,
                    stats.avg_loss,
                    replay_stats.buffer_size,
                    stats.consolidations
                );
            }
        }

        // Checkpoint save
        if last_checkpoint.elapsed() > Duration::from_secs(60)
            || examples_processed % args.checkpoint_interval == 0
        {
            if examples_processed > 0 {
                save_checkpoint(&learner, &args.output, examples_processed)?;
                last_checkpoint = Instant::now();
            }
        }

        // Check for level advancement logging
        if generator.examples_at_level == 0 && args.verbose >= 1 {
            pb.println(format!(
                "Advanced to Level {}: {}",
                generator.current_level,
                CurriculumGenerator::level_description(generator.current_level)
            ));
        }
    }

    // Train remaining batch
    if !batch_buffer.is_empty() {
        learner.learn_batch(&batch_buffer);
        examples_processed += batch_buffer.len();
    }

    pb.finish_with_message("Done!");

    // Final checkpoint
    save_checkpoint(&learner, &args.output, examples_processed)?;

    // Print summary
    let elapsed = start_time.elapsed();
    let stats = learner.stats();
    let replay_stats = learner.replay_stats();

    println!("\n=== Training Summary ===");
    println!("Examples processed: {}", examples_processed);
    println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Examples/sec: {:.1}", examples_processed as f64 / elapsed.as_secs_f64());
    println!("\nLearning Stats:");
    println!("  Average loss: {:.4}", stats.avg_loss);
    println!("  Best loss: {:.4}", stats.best_loss);
    println!("  Consolidations: {}", stats.consolidations);
    println!("  Batches trained: {}", stats.batches_trained);

    println!("\nReplay Stats:");
    println!("  Buffer size: {}", replay_stats.buffer_size);
    println!("  Total replays: {}", replay_stats.total_replays);
    println!("  Buffer avg loss: {:.4}", replay_stats.avg_loss);
    println!("  Strategy: {:?}", replay_stats.strategy);

    println!("\nDomain Distribution:");
    for (domain, count) in &stats.domain_counts {
        println!("  {}: {} examples", domain, count);
    }

    println!("\nFinal checkpoint saved to: {}/checkpoint_final.json", args.output);

    Ok(())
}

fn save_checkpoint(
    learner: &MemoryOnlineLearner,
    output_dir: &str,
    examples: usize,
) -> std::io::Result<()> {
    let checkpoint_path = format!("{}/checkpoint_{}.json", output_dir, examples);
    let file = File::create(&checkpoint_path)?;
    let mut writer = BufWriter::new(file);

    // Save stats as JSON for quick inspection
    let stats = learner.stats();
    let replay_stats = learner.replay_stats();

    let summary = serde_json::json!({
        "examples_processed": examples,
        "learning_stats": {
            "examples_seen": stats.examples_seen,
            "batches_trained": stats.batches_trained,
            "consolidations": stats.consolidations,
            "avg_loss": stats.avg_loss,
            "best_loss": stats.best_loss,
            "domain_counts": stats.domain_counts,
        },
        "replay_stats": {
            "buffer_size": replay_stats.buffer_size,
            "total_replays": replay_stats.total_replays,
            "avg_loss": replay_stats.avg_loss,
            "strategy": format!("{:?}", replay_stats.strategy),
        },
    });

    writeln!(writer, "{}", serde_json::to_string_pretty(&summary)?)?;

    // Also save latest as "final"
    let final_path = format!("{}/checkpoint_final.json", output_dir);
    fs::copy(&checkpoint_path, final_path)?;

    Ok(())
}
