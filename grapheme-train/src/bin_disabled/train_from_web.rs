//! Training from web content binary for GRAPHEME.
//!
//! Backend-172: Learn from web pages (text + images).
//!
//! Usage:
//!   train_from_web --urls urls.txt --output checkpoints/
//!   train_from_web --url https://example.com --epochs 50
//!   train_from_web --sitemap https://example.com/sitemap.xml

use clap::Parser;
use grapheme_core::{GraphTransformNet, GraphemeGraph, UnifiedCheckpoint};
use grapheme_train::{
    compute_structural_loss, Adam, HtmlParser, LRScheduler, StructuralLossConfig, TextPipeline,
    TrainingLoop, TrainingMetrics, TrainingState, WebFetcher,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "train_from_web")]
#[command(about = "Train GRAPHEME from web content (text + images)", long_about = None)]
struct Args {
    /// Single URL to fetch and train from
    #[arg(short, long)]
    url: Option<String>,

    /// File containing list of URLs (one per line)
    #[arg(long)]
    urls: Option<PathBuf>,

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

    /// Include image training (multimodal)
    #[arg(long)]
    include_images: bool,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Dry run - fetch and parse but don't train
    #[arg(long)]
    dry_run: bool,

    /// Maximum number of chunks to train on (0 = unlimited)
    #[arg(long, default_value = "0")]
    max_chunks: usize,
}

/// Training example (text or image description)
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WebExample {
    /// Source URL
    url: String,
    /// Content type
    content_type: ExampleType,
    /// Text content
    text: String,
    /// Image data (if image)
    image_data: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
enum ExampleType {
    Text,
    ImageAlt, // Image with alt text description
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

    println!("GRAPHEME Web Training");
    println!("=====================");
    println!("Output: {:?}", args.output);
    println!("Epochs: {}", args.epochs);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.lr);
    println!("Include images: {}", args.include_images);

    // Collect URLs
    let mut urls: Vec<String> = Vec::new();

    if let Some(url) = &args.url {
        urls.push(url.clone());
    }

    if let Some(urls_file) = &args.urls {
        let content = fs::read_to_string(urls_file)?;
        for line in content.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                urls.push(line.to_string());
            }
        }
    }

    if urls.is_empty() {
        anyhow::bail!("No URLs provided. Use --url or --urls");
    }

    println!("\nURLs to process: {}", urls.len());

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Fetch and parse web content
    println!("\nFetching web content...");
    let fetcher = WebFetcher::new();
    let parser = HtmlParser::new();
    let pipeline = TextPipeline::new();

    let mut examples: Vec<WebExample> = Vec::new();
    let mut fetch_errors = 0;

    for (i, url) in urls.iter().enumerate() {
        if args.verbose {
            println!("  [{}/{}] Fetching: {}", i + 1, urls.len(), url);
        }

        match fetcher.fetch(url) {
            Ok(content) => {
                if let Some(text) = &content.text {
                    // Parse HTML if it's HTML content
                    let parsed = if content.is_html() {
                        parser.parse(text)
                    } else {
                        // Plain text - wrap in simple parsed structure
                        grapheme_train::ParsedHtml {
                            text: text.clone(),
                            metadata: Default::default(),
                            links: Vec::new(),
                            images: Vec::new(),
                            headings: Vec::new(),
                            paragraphs: vec![text.clone()],
                        }
                    };

                    // Process text content
                    let chunks = pipeline.process(&parsed.text);
                    let chunk_count = chunks.len();
                    for chunk in chunks {
                        if chunk.word_count >= 3 {
                            examples.push(WebExample {
                                url: url.clone(),
                                content_type: ExampleType::Text,
                                text: chunk.text,
                                image_data: None,
                            });
                        }
                    }

                    // Process images with alt text (multimodal learning)
                    if args.include_images {
                        for (img_src, alt_text) in &parsed.images {
                            if !alt_text.is_empty() && alt_text.split_whitespace().count() >= 2 {
                                examples.push(WebExample {
                                    url: img_src.clone(),
                                    content_type: ExampleType::ImageAlt,
                                    text: alt_text.clone(),
                                    image_data: None, // Could fetch image bytes here
                                });
                            }
                        }
                    }

                    if args.verbose {
                        println!(
                            "    Extracted {} text chunks, {} images",
                            chunk_count,
                            parsed.images.len()
                        );
                    }
                }
            }
            Err(e) => {
                if args.verbose {
                    println!("    Error: {}", e);
                }
                fetch_errors += 1;
            }
        }
    }

    println!(
        "\nFetched {} examples from {} URLs ({} errors)",
        examples.len(),
        urls.len(),
        fetch_errors
    );

    // Limit examples if requested
    if args.max_chunks > 0 && examples.len() > args.max_chunks {
        examples.truncate(args.max_chunks);
        println!("Limited to {} examples", args.max_chunks);
    }

    if examples.is_empty() {
        anyhow::bail!("No valid content extracted from URLs");
    }

    // Show sample
    if args.verbose {
        println!("\nSample examples:");
        for (i, ex) in examples.iter().take(3).enumerate() {
            let preview_len = ex.text.len().min(60);
            println!(
                "  [{}] {:?}: {}...",
                i,
                ex.content_type,
                &ex.text[..preview_len]
            );
        }
    }

    if args.dry_run {
        println!("\n[Dry run] Content fetched and parsed successfully");
        println!("  Text examples: {}", examples.iter().filter(|e| matches!(e.content_type, ExampleType::Text)).count());
        println!("  Image examples: {}", examples.iter().filter(|e| matches!(e.content_type, ExampleType::ImageAlt)).count());
        return Ok(());
    }

    // Model parameters
    const VOCAB_SIZE: usize = 256;
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

        // Create training pairs from consecutive examples
        let pairs: Vec<(&str, &str)> = examples
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

            model.zero_grad();
            let mut batch_loss = 0.0;

            for (input, target) in batch {
                let input_graph = GraphemeGraph::from_text(input);
                let target_graph = GraphemeGraph::from_text(target);

                let (predicted_graph, pooling_result) = model.forward(&input_graph);

                let loss_result = compute_structural_loss(
                    &predicted_graph,
                    &target_graph,
                    &structural_config,
                );

                batch_loss += loss_result.total_loss;
                model.backward(&input_graph, &pooling_result, &loss_result.activation_gradients, EMBED_DIM);
            }

            batch_loss /= batch.len() as f32;
            epoch_loss += batch_loss;
            batch_count += 1;

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
            let checkpoint_path = args.output.join(format!("web_epoch_{}.checkpoint", epoch + 1));
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
    let final_path = args.output.join("web_final.checkpoint");
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
