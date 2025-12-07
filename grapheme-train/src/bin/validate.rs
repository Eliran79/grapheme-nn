//! Model validation CLI for GRAPHEME.
//!
//! Validates trained models against test data.

use clap::Parser;
use grapheme_core::GraphTransformNet;
use grapheme_polish::expr_to_polish;
use grapheme_train::{compute_ged_loss, validate_dataset, Dataset, Pipeline};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "validate")]
#[command(about = "Validate GRAPHEME model", long_about = None)]
struct Args {
    /// Path to trained model file
    #[arg(short, long)]
    model: PathBuf,

    /// Path to validation data (file or directory)
    #[arg(short, long)]
    data: PathBuf,

    /// Output results to JSON file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Run dataset validation (check data integrity)
    #[arg(long)]
    validate_data: bool,

    /// Show per-example results
    #[arg(long)]
    show_examples: bool,

    /// Maximum examples to show (with --show-examples)
    #[arg(long, default_value = "10")]
    max_examples: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, serde::Serialize)]
struct ValidationResults {
    model_path: String,
    data_path: String,
    total_examples: usize,
    correct: usize,
    incorrect: usize,
    errors: usize,
    accuracy: f64,
    avg_loss: f64,
    elapsed_secs: f64,
    examples_per_sec: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Model Validation");
    println!("=========================");
    println!("Model: {:?}", args.model);
    println!("Data: {:?}", args.data);

    // Load model
    let start = Instant::now();
    let model = GraphTransformNet::load_from_file(&args.model)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());

    // Create pipeline with loaded model
    let pipeline = Pipeline::new().with_transform_net(model);

    // Load datasets
    let datasets = if args.data.is_dir() {
        // Load all .jsonl files from directory
        let mut datasets = Vec::new();
        for entry in fs::read_dir(&args.data)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "jsonl") {
                let name = path.file_stem().unwrap().to_string_lossy().to_string();
                let dataset = Dataset::load_jsonl(&path, &name)?;
                datasets.push((path, dataset));
            }
        }
        datasets
    } else {
        // Load single file
        let name = args
            .data
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let dataset = Dataset::load_jsonl(&args.data, &name)?;
        vec![(args.data.clone(), dataset)]
    };

    if datasets.is_empty() {
        anyhow::bail!("No .jsonl files found in {:?}", args.data);
    }

    println!("\nFound {} dataset(s)", datasets.len());

    // Optional: validate dataset integrity first
    if args.validate_data {
        println!("\n--- Dataset Validation ---");
        for (path, dataset) in &datasets {
            print!("Validating {:?}... ", path);
            let report = validate_dataset(dataset)?;
            println!(
                "{}/{} valid ({:.1}%)",
                report.valid,
                report.total,
                report.valid as f64 / report.total as f64 * 100.0
            );
            if !report.errors.is_empty() && args.verbose {
                for err in report.errors.iter().take(5) {
                    println!("  - {}", err);
                }
                if report.errors.len() > 5 {
                    println!("  ... and {} more errors", report.errors.len() - 5);
                }
            }
        }
    }

    // Validate model on each dataset
    println!("\n--- Model Validation ---");

    let mut all_results = Vec::new();

    for (path, dataset) in &datasets {
        let dataset_start = Instant::now();

        println!("\nDataset: {:?} ({} examples)", path, dataset.len());

        let mut correct = 0;
        let mut incorrect = 0;
        let mut errors = 0;
        let mut total_loss = 0.0;
        let mut example_results = Vec::new();

        for (i, example) in dataset.examples.iter().enumerate() {
            // Process through pipeline
            let result = pipeline.process(&example.input_polish);

            // Check correctness
            let is_correct = if let Some(expected) = example.expected_result {
                if let Some(actual) = result.numeric_result {
                    (actual - expected).abs() < 1e-6
                } else {
                    false
                }
            } else if let Some(ref expected) = &example.expected_symbolic {
                let expected_polish = expr_to_polish(expected);
                result.symbolic_result.as_ref() == Some(&expected_polish)
            } else {
                false
            };

            if result.success() {
                if is_correct {
                    correct += 1;
                } else {
                    incorrect += 1;
                }
            } else {
                errors += 1;
            }

            // Compute loss if we have graphs
            if let (Some(ref nl_graph), Some(expected)) = (&result.nl_graph, example.expected_result)
            {
                let target_graph = grapheme_core::GraphemeGraph::from_text(&expected.to_string());
                let loss = compute_ged_loss(nl_graph, &target_graph, 1.0, 0.5, 2.0);
                total_loss += loss as f64;
            }

            // Collect example results for display
            if args.show_examples && example_results.len() < args.max_examples {
                example_results.push((
                    example.id.clone(),
                    example.input_polish.clone(),
                    result.result_string(),
                    is_correct,
                ));
            }

            if args.verbose && (i + 1) % 1000 == 0 {
                print!(
                    "\r  Processed {}/{} ({:.1}%)",
                    i + 1,
                    dataset.len(),
                    (i + 1) as f64 / dataset.len() as f64 * 100.0
                );
            }
        }

        if args.verbose {
            println!();
        }

        let elapsed = dataset_start.elapsed();
        let accuracy = correct as f64 / dataset.len() as f64 * 100.0;
        let avg_loss = total_loss / dataset.len() as f64;
        let examples_per_sec = dataset.len() as f64 / elapsed.as_secs_f64();

        println!("  Results:");
        println!("    Correct:   {} ({:.1}%)", correct, accuracy);
        println!(
            "    Incorrect: {} ({:.1}%)",
            incorrect,
            incorrect as f64 / dataset.len() as f64 * 100.0
        );
        println!(
            "    Errors:    {} ({:.1}%)",
            errors,
            errors as f64 / dataset.len() as f64 * 100.0
        );
        println!("    Avg Loss:  {:.4}", avg_loss);
        println!(
            "    Time:      {:.2}s ({:.0} ex/s)",
            elapsed.as_secs_f64(),
            examples_per_sec
        );

        if args.show_examples && !example_results.is_empty() {
            println!("\n  Example results:");
            for (id, input, output, correct) in &example_results {
                let status = if *correct { "OK" } else { "WRONG" };
                println!("    [{}] {} -> {} ({})", status, id, input, output);
            }
        }

        all_results.push(ValidationResults {
            model_path: args.model.to_string_lossy().to_string(),
            data_path: path.to_string_lossy().to_string(),
            total_examples: dataset.len(),
            correct,
            incorrect,
            errors,
            accuracy,
            avg_loss,
            elapsed_secs: elapsed.as_secs_f64(),
            examples_per_sec,
        });
    }

    // Summary
    if datasets.len() > 1 {
        println!("\n--- Overall Summary ---");
        let total: usize = all_results.iter().map(|r| r.total_examples).sum();
        let total_correct: usize = all_results.iter().map(|r| r.correct).sum();
        let total_incorrect: usize = all_results.iter().map(|r| r.incorrect).sum();
        let total_errors: usize = all_results.iter().map(|r| r.errors).sum();
        let overall_accuracy = total_correct as f64 / total as f64 * 100.0;

        println!("  Total examples: {}", total);
        println!("  Correct:        {} ({:.1}%)", total_correct, overall_accuracy);
        println!(
            "  Incorrect:      {} ({:.1}%)",
            total_incorrect,
            total_incorrect as f64 / total as f64 * 100.0
        );
        println!(
            "  Errors:         {} ({:.1}%)",
            total_errors,
            total_errors as f64 / total as f64 * 100.0
        );
    }

    // Save results to file
    if let Some(output_path) = &args.output {
        let json = serde_json::to_string_pretty(&all_results)?;
        fs::write(output_path, json)?;
        println!("\nResults saved to: {:?}", output_path);
    }

    Ok(())
}
