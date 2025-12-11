//! LLM-Augmented Training Binary for GRAPHEME
//!
//! Integration-005: Train GRAPHEME using LLM-generated examples.
//!
//! This binary uses an LLM to generate training pairs:
//! - Queries the LLM for Q&A pairs
//! - Converts both question and answer to GraphemeGraphs
//! - Trains the model to transform question graphs into answer graphs
//!
//! Usage:
//!   train_with_llm --provider ollama --model llama2 --topics math,science
//!   train_with_llm --provider claude --topics coding --examples 100
//!   OPENAI_API_KEY=xxx train_with_llm --provider openai

use clap::Parser;
use grapheme_core::{GraphTransformNet, GraphemeGraph, UnifiedCheckpoint};
use grapheme_train::{
    compute_structural_loss, Adam, LLMClient, LRScheduler,
    StructuralLossConfig, TrainingLoop, TrainingMetrics, TrainingState,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_with_llm")]
#[command(about = "Train GRAPHEME using LLM-generated examples")]
struct Args {
    /// LLM provider: ollama, claude, openai, gemini
    #[arg(short, long, default_value = "ollama")]
    provider: String,

    /// Model name (provider-specific)
    #[arg(short, long, default_value = "llama2")]
    model: String,

    /// Topics to generate examples for (comma-separated)
    #[arg(short, long, default_value = "general,math,science")]
    topics: String,

    /// Number of examples to generate per topic
    #[arg(short, long, default_value = "10")]
    examples: usize,

    /// Output directory for checkpoints
    #[arg(short, long, default_value = "checkpoints")]
    output: PathBuf,

    /// Number of training epochs
    #[arg(long, default_value = "50")]
    epochs: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f64,

    /// Enable verbose output
    #[arg(long)]
    verbose: bool,

    /// Dry run - generate examples but don't train
    #[arg(long)]
    dry_run: bool,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<PathBuf>,
}

/// A training pair generated from LLM
#[derive(Debug, Clone)]
struct LLMTrainingPair {
    topic: String,
    question: String,
    answer: String,
}

/// Generate training examples using LLM
fn generate_examples(
    client: &LLMClient,
    topics: &[&str],
    examples_per_topic: usize,
    verbose: bool,
) -> Vec<LLMTrainingPair> {
    let mut pairs = Vec::new();

    for topic in topics {
        if verbose {
            eprintln!("Generating {} examples for topic: {}", examples_per_topic, topic);
        }

        let prompt = format!(
            "Generate {} question-answer pairs about {}. \
             Format each pair as:\n\
             Q: [question]\n\
             A: [answer]\n\n\
             Keep answers concise (1-2 sentences). Start now:",
            examples_per_topic, topic
        );

        match client.generate(&prompt) {
            Ok(response) => {
                // Parse Q&A pairs from response
                let parsed = parse_qa_pairs(&response, topic);
                if verbose {
                    eprintln!("  Parsed {} pairs from response", parsed.len());
                }
                pairs.extend(parsed);
            }
            Err(e) => {
                eprintln!("  Error generating for {}: {}", topic, e);
            }
        }
    }

    pairs
}

/// Parse Q&A pairs from LLM response
fn parse_qa_pairs(text: &str, topic: &str) -> Vec<LLMTrainingPair> {
    let mut pairs = Vec::new();
    let mut current_question = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("Q:") || line.starts_with("Question:") {
            current_question = line
                .trim_start_matches("Q:")
                .trim_start_matches("Question:")
                .trim()
                .to_string();
        } else if (line.starts_with("A:") || line.starts_with("Answer:")) && !current_question.is_empty() {
            let answer = line
                .trim_start_matches("A:")
                .trim_start_matches("Answer:")
                .trim()
                .to_string();

            if !answer.is_empty() {
                pairs.push(LLMTrainingPair {
                    topic: topic.to_string(),
                    question: current_question.clone(),
                    answer,
                });
            }
            current_question.clear();
        }
    }

    pairs
}

/// Save checkpoint
fn save_checkpoint(
    path: &PathBuf,
    model: &GraphTransformNet,
    training_state: &TrainingState,
    metrics: &TrainingMetrics,
    optimizer: &Adam,
) -> anyhow::Result<()> {
    let mut checkpoint = UnifiedCheckpoint::new();
    checkpoint.add_module(model)?;
    checkpoint.add_module(training_state)?;
    checkpoint.add_module(metrics)?;
    checkpoint.add_module(optimizer)?;
    checkpoint.save_to_file(path)?;
    Ok(())
}

/// Load checkpoint
fn load_checkpoint(
    path: &PathBuf,
) -> anyhow::Result<(GraphTransformNet, TrainingState, TrainingMetrics, Adam)> {
    let checkpoint = UnifiedCheckpoint::load_from_file(path)?;
    let model: GraphTransformNet = checkpoint.load_module()?;
    let training_state: TrainingState = checkpoint.load_module()?;
    let metrics: TrainingMetrics = checkpoint.load_module()?;
    let optimizer: Adam = checkpoint.load_module()?;
    Ok((model, training_state, metrics, optimizer))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME LLM-Augmented Training");
    println!("===============================");
    println!("Provider: {}", args.provider);
    println!("Model: {}", args.model);
    println!("Topics: {}", args.topics);
    println!("Examples per topic: {}", args.examples);
    println!("Output: {:?}", args.output);

    // Create LLM client
    let client = match args.provider.as_str() {
        "ollama" => LLMClient::ollama(&args.model),
        "claude" => LLMClient::claude(),
        "openai" => LLMClient::openai(),
        "gemini" => LLMClient::gemini(),
        _ => {
            anyhow::bail!("Unknown provider: {}. Use ollama, claude, openai, or gemini", args.provider);
        }
    };

    // Parse topics
    let topics: Vec<&str> = args.topics.split(',').map(|s| s.trim()).collect();

    // Generate examples
    println!("\nGenerating training examples from LLM...");
    let examples = generate_examples(&client, &topics, args.examples, args.verbose);
    println!("Generated {} training pairs", examples.len());

    if examples.is_empty() {
        anyhow::bail!("No training examples generated. Check LLM connectivity.");
    }

    // Show samples
    if args.verbose {
        println!("\nSample pairs:");
        for (i, pair) in examples.iter().take(3).enumerate() {
            println!("  [{}] Topic: {}", i, pair.topic);
            println!("      Q: {}", &pair.question[..pair.question.len().min(60)]);
            println!("      A: {}", &pair.answer[..pair.answer.len().min(60)]);
        }
    }

    if args.dry_run {
        println!("\n[Dry run] Examples generated successfully");
        println!("  Total pairs: {}", examples.len());
        for topic in &topics {
            let count = examples.iter().filter(|e| e.topic == *topic).count();
            println!("  {}: {} pairs", topic, count);
        }
        return Ok(());
    }

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Model parameters
    const VOCAB_SIZE: usize = 256;
    const EMBED_DIM: usize = 64;
    const HIDDEN_DIM: usize = 128;
    const NUM_LAYERS: usize = 3;

    // Initialize or resume
    let config = grapheme_train::TrainingConfig {
        learning_rate: args.lr as f32,
        batch_size: 16,
        epochs: args.epochs,
        ..Default::default()
    };

    let (mut model, _optimizer, mut training_loop) = if let Some(ref resume_path) = args.resume {
        println!("\nResuming from checkpoint: {:?}", resume_path);
        let (model, state, metrics, opt) = load_checkpoint(resume_path)?;
        let mut loop_state = TrainingLoop::new(config.clone()).with_scheduler(
            LRScheduler::CosineAnnealingLR {
                t_max: args.epochs,
                eta_min: args.lr as f32 * 0.01,
            },
        );
        loop_state.state = state;
        loop_state.metrics = metrics;
        (model, opt, loop_state)
    } else {
        println!("\nInitializing new model...");
        let model = GraphTransformNet::new(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS);
        let optimizer = Adam::new(args.lr as f32);
        let training_loop = TrainingLoop::new(config.clone()).with_scheduler(
            LRScheduler::CosineAnnealingLR {
                t_max: args.epochs,
                eta_min: args.lr as f32 * 0.01,
            },
        );
        (model, optimizer, training_loop)
    };

    // Structural loss configuration
    let structural_config = StructuralLossConfig {
        alpha: 1.0,
        beta: 1.0,
        gamma: 0.1,
        sinkhorn: grapheme_train::SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-6,
        },
    };

    // Training loop
    println!("\nStarting training...");
    let total_start = Instant::now();

    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for pair in &examples {
            // Convert question and answer to graphs
            let input_graph = GraphemeGraph::from_text(&pair.question);
            let target_graph = GraphemeGraph::from_text(&pair.answer);

            model.zero_grad();

            // Forward pass
            let (predicted_graph, pooling_result) = model.forward(&input_graph);

            // Compute loss
            let loss_result = compute_structural_loss(
                &predicted_graph,
                &target_graph,
                &structural_config,
            );

            epoch_loss += loss_result.total_loss;
            batch_count += 1;

            // Backward pass
            model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);

            // Update weights
            let lr = training_loop.state.current_lr;
            model.step(lr);
            training_loop.record_batch(loss_result.total_loss);

            if args.verbose && batch_count % 10 == 0 {
                println!(
                    "  Batch {}/{}: loss = {:.4}",
                    batch_count,
                    examples.len(),
                    loss_result.total_loss
                );
            }
        }

        let avg_loss = if batch_count > 0 {
            epoch_loss / batch_count as f32
        } else {
            0.0
        };
        let elapsed = epoch_start.elapsed();

        println!(
            "Epoch {}/{}: loss = {:.4}, lr = {:.6}, time = {:.1}s",
            epoch + 1,
            args.epochs,
            avg_loss,
            training_loop.state.current_lr,
            elapsed.as_secs_f32()
        );

        training_loop.complete_epoch();

        // Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 {
            let checkpoint_path = args.output.join(format!("llm_epoch_{}.checkpoint", epoch + 1));
            let adam = Adam::new(training_loop.state.current_lr);
            save_checkpoint(
                &checkpoint_path,
                &model,
                &training_loop.state,
                &training_loop.metrics,
                &adam,
            )?;
            println!("  Saved checkpoint: {:?}", checkpoint_path);
        }
    }

    // Final save
    let final_path = args.output.join("llm_final.checkpoint");
    let adam = Adam::new(training_loop.state.current_lr);
    save_checkpoint(
        &final_path,
        &model,
        &training_loop.state,
        &training_loop.metrics,
        &adam,
    )?;

    let total_elapsed = total_start.elapsed();
    println!("\nTraining complete!");
    println!("Total time: {:.1}s", total_elapsed.as_secs_f32());
    println!("Final checkpoint: {:?}", final_path);
    println!("Training pairs used: {}", examples.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qa_pairs() {
        let text = "Q: What is 2+2?\nA: 4\n\nQ: What is the capital of France?\nA: Paris";
        let pairs = parse_qa_pairs(text, "test");
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].question, "What is 2+2?");
        assert_eq!(pairs[0].answer, "4");
    }

    #[test]
    fn test_parse_qa_pairs_alternate_format() {
        let text = "Question: How does photosynthesis work?\nAnswer: Plants convert sunlight to energy.";
        let pairs = parse_qa_pairs(text, "science");
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].question.contains("photosynthesis"));
    }
}
