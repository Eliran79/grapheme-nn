//! Training from text files binary for GRAPHEME.
//!
//! Backend-171: Learn from TXT/MD/JSON files directly.
//!
//! Usage:
//!   train_from_text --input path/to/files --output checkpoints/
//!   train_from_text --input documents/ --epochs 100

use clap::Parser;
use grapheme_core::{GraphTransformNet, GraphemeGraph, UnifiedCheckpoint};
use grapheme_train::{
    compute_structural_loss, Adam, LRScheduler, StructuralLossConfig, TextIngestion,
    TextPipeline, TrainingLoop, TrainingMetrics, TrainingState,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_from_text")]
#[command(about = "Train GRAPHEME from text files (TXT/MD/JSON)", long_about = None)]
struct Args {
    /// Input file or directory containing text files
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory for checkpoints
    #[arg(short, long, default_value = "checkpoints")]
    output: PathBuf,

    /// Number of training epochs
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// Batch size for training
    #[arg(short, long, default_value = "32")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    lr: f64,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Dry run - load data but don't train
    #[arg(long)]
    dry_run: bool,

    /// Maximum number of chunks to train on (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_chunks: usize,
}

/// Save a unified checkpoint
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

/// Load a unified checkpoint
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

    println!("GRAPHEME Text Training");
    println!("======================");
    println!("Input: {:?}", args.input);
    println!("Output: {:?}", args.output);
    println!("Epochs: {}", args.epochs);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.lr);

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Load text files
    println!("\nLoading text files...");
    let ingestion = TextIngestion::new();
    let mut all_text = String::new();
    let file_count: usize;

    if args.input.is_file() {
        // Single file
        match ingestion.load_file(&args.input) {
            Ok(doc) => {
                all_text.push_str(&doc.content);
                all_text.push_str("\n\n");
                file_count = 1;
            }
            Err(e) => anyhow::bail!("Failed to load file: {}", e),
        }
    } else if args.input.is_dir() {
        // Directory
        match ingestion.load_directory(&args.input) {
            Ok(files) => {
                for file in &files {
                    all_text.push_str(&file.content);
                    all_text.push_str("\n\n");
                }
                file_count = files.len();
            }
            Err(e) => anyhow::bail!("Failed to load directory: {}", e),
        }
    } else {
        anyhow::bail!("Input path does not exist: {:?}", args.input);
    }

    println!("Loaded {} files, {} total characters", file_count, all_text.len());

    // Preprocess text
    println!("\nPreprocessing text...");
    let pipeline = TextPipeline::new();
    let chunks = pipeline.process(&all_text);

    let mut chunks: Vec<_> = chunks
        .into_iter()
        .filter(|c| !c.text.is_empty() && c.word_count >= 3)
        .collect();

    // Limit chunks if requested
    if args.max_chunks > 0 && chunks.len() > args.max_chunks {
        chunks.truncate(args.max_chunks);
    }

    println!("Created {} text chunks for training", chunks.len());

    if chunks.is_empty() {
        anyhow::bail!("No valid text chunks found in input");
    }

    if args.verbose {
        println!("\nSample chunks:");
        for (i, chunk) in chunks.iter().take(3).enumerate() {
            let preview_len = chunk.text.len().min(60);
            println!(
                "  [{}] {} words: {}...",
                i,
                chunk.word_count,
                &chunk.text[..preview_len]
            );
        }
    }

    if args.dry_run {
        println!("\n[Dry run] Data loaded and preprocessed successfully");
        return Ok(());
    }

    // Model parameters
    const VOCAB_SIZE: usize = 256; // ASCII characters
    const EMBED_DIM: usize = 64;
    const HIDDEN_DIM: usize = 128;
    const NUM_LAYERS: usize = 3;

    // Initialize or resume
    let config = grapheme_train::TrainingConfig {
        learning_rate: args.lr as f32,
        batch_size: args.batch_size,
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

        // Create training pairs from consecutive chunks
        // Each chunk becomes input, next chunk becomes target (next-text prediction)
        let pairs: Vec<(&str, &str)> = chunks
            .windows(2)
            .map(|w| (w[0].text.as_str(), w[1].text.as_str()))
            .collect();

        // Process in batches
        for batch_start in (0..pairs.len()).step_by(args.batch_size) {
            let batch_end = (batch_start + args.batch_size).min(pairs.len());
            let batch = &pairs[batch_start..batch_end];

            if batch.is_empty() {
                continue;
            }

            // Zero gradients
            model.zero_grad();

            let mut batch_loss = 0.0;

            for (input, target) in batch {
                // Convert text to graph structures
                let input_graph = GraphemeGraph::from_text(input);
                let target_graph = GraphemeGraph::from_text(target);

                // Forward pass
                let (predicted_graph, pooling_result) = model.forward(&input_graph);

                // Compute structural loss
                let loss_result = compute_structural_loss(
                    &predicted_graph,
                    &target_graph,
                    &structural_config,
                );

                batch_loss += loss_result.total_loss;

                // Backward pass
                model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);
            }

            // Average loss for batch
            batch_loss /= batch.len() as f32;
            epoch_loss += batch_loss;
            batch_count += 1;

            // Record and step
            training_loop.record_batch(batch_loss);
            let lr = training_loop.state.current_lr;
            model.step(lr);

            if args.verbose && batch_count % 10 == 0 {
                println!(
                    "  Batch {}/{}: loss = {:.4}",
                    batch_count,
                    (pairs.len() + args.batch_size - 1) / args.batch_size,
                    batch_loss
                );
            }
        }

        // Epoch summary
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

        // Update training loop state
        training_loop.complete_epoch();

        // Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 {
            let checkpoint_path = args.output.join(format!("text_epoch_{}.checkpoint", epoch + 1));
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
    let final_path = args.output.join("text_final.checkpoint");
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

    Ok(())
}
