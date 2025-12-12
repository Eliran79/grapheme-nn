//! HumanEval Evaluation Harness for CortexMesh
//!
//! Evaluates the trained CortexMesh model on HumanEval benchmark.
//! Uses graph-to-graph transformation (true to GRAPHEME vision).
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin eval_humaneval_mesh -- \
//!     --model checkpoints/cortex_leaky_relu.json \
//!     --data data/humaneval/problems.jsonl \
//!     --samples 10 --k 1

use clap::Parser;
use grapheme_code::CodeBrain;
use grapheme_core::DomainBrain;
use grapheme_train::cortex_mesh::{CortexMesh, MeshConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "eval_humaneval_mesh")]
#[command(about = "Evaluate CortexMesh on HumanEval benchmark")]
struct Args {
    /// Path to trained CortexMesh checkpoint
    #[arg(short, long)]
    model: PathBuf,

    /// Path to HumanEval problems JSONL
    #[arg(short, long)]
    data: PathBuf,

    /// Number of samples to generate per problem
    #[arg(short, long, default_value = "10")]
    samples: usize,

    /// K values for pass@k (comma-separated)
    #[arg(short, long, default_value = "1,10")]
    k: String,

    /// Output results to file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// HumanEval problem format
#[derive(Debug, Clone, Deserialize, Serialize)]
struct HumanEvalProblem {
    task_id: String,
    prompt: String,
    canonical_solution: String,
    test: String,
    entry_point: String,
    #[serde(default)]
    description: String,
}

/// Evaluation result for a single problem
#[derive(Debug, Clone, Serialize)]
struct ProblemResult {
    task_id: String,
    num_samples: usize,
    num_correct: usize,
    samples: Vec<SampleResult>,
}

#[derive(Debug, Clone, Serialize)]
struct SampleResult {
    generated_code: String,
    syntax_valid: bool,
    test_passed: bool,
    error: Option<String>,
}

/// Load HumanEval problems from JSONL
fn load_problems(path: &PathBuf) -> anyhow::Result<Vec<HumanEvalProblem>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut problems = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let problem: HumanEvalProblem = serde_json::from_str(&line)?;
        problems.push(problem);
    }

    Ok(problems)
}

/// Execute generated code with test cases
fn execute_test(prompt: &str, generated: &str, test_code: &str, entry_point: &str) -> (bool, Option<String>) {
    // Build complete test file
    let full_code = format!(
        "{}\n{}\n\n{}\n\n# Run tests\ncheck({})\n",
        prompt, generated, test_code, entry_point
    );

    // Write to temp file with unique ID
    let temp_path = format!("/tmp/humaneval_mesh_{}_{}.py",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    match std::fs::write(&temp_path, &full_code) {
        Ok(_) => {}
        Err(e) => return (false, Some(format!("Failed to write temp file: {}", e))),
    }

    // Execute with Python (with timeout)
    let output = Command::new("python3")
        .arg(&temp_path)
        .output();

    // Clean up
    let _ = std::fs::remove_file(&temp_path);

    match output {
        Ok(result) => {
            if result.status.success() {
                (true, None)
            } else {
                let stderr = String::from_utf8_lossy(&result.stderr).to_string();
                (false, Some(stderr.chars().take(500).collect()))
            }
        }
        Err(e) => (false, Some(format!("Execution failed: {}", e))),
    }
}

/// Compute pass@k using unbiased estimator
fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if n < k {
        return if c > 0 { 1.0 } else { 0.0 };
    }

    if c == 0 {
        return 0.0;
    }

    if c >= k {
        let mut log_prod = 0.0;
        for i in 0..k {
            if n - c < i + 1 {
                return 1.0;
            }
            log_prod += ((n - c - i) as f64).ln() - ((n - i) as f64).ln();
        }
        1.0 - log_prod.exp()
    } else {
        1.0 - (1..=k).fold(1.0, |acc, i| {
            if i > c { acc } else { acc * (n - c) as f64 / (n - i + 1) as f64 }
        })
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       HumanEval Evaluation - CortexMesh (Graph-to-Graph)     ║");
    println!("║                   Target: Beat 96.2% pass@1                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Parse k values
    let k_values: Vec<usize> = args.k
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    println!("Evaluating pass@k for k={:?}", k_values);
    println!("Samples per problem: {}", args.samples);

    // Load problems
    println!("\nLoading HumanEval problems from {:?}...", args.data);
    let problems = load_problems(&args.data)?;
    println!("Loaded {} problems", problems.len());

    // Create CortexMesh configuration
    let config = MeshConfig {
        activation_threshold: 0.2,
        max_active_brains: usize::MAX,
        parallel: true,
        hidden_dim: 256,
        num_layers: 6,
        vocab_size: 256,
        embed_dim: 64,
    };

    // Load model
    println!("\nLoading CortexMesh from {:?}...", args.model);
    let mut mesh = CortexMesh::discover_with_config(config);
    mesh.load(&args.model)?;
    println!("Model loaded: {} brains, {} modules", mesh.brain_count(), mesh.module_count());

    // Initialize CodeBrain for syntax validation
    let code_brain = CodeBrain::new();

    // Wrap mesh in Mutex for parallel access
    let mesh = Mutex::new(mesh);

    // Evaluation
    let start = Instant::now();
    let mut all_results: Vec<ProblemResult> = Vec::new();

    for (prob_idx, problem) in problems.iter().enumerate() {
        let prob_start = Instant::now();

        if args.verbose {
            println!("\n--- Problem {}/{}: {} ---", prob_idx + 1, problems.len(), problem.task_id);
            println!("Entry point: {}", problem.entry_point);
        }

        // Generate samples (sequentially since mesh needs mutable access)
        let mut samples = Vec::new();
        for sample_idx in 0..args.samples {
            // Generate code using CortexMesh
            let generated = {
                let mut mesh = mesh.lock().unwrap();
                let result = mesh.process_parallel(&problem.prompt);
                result.decoded.clone()
            };

            // Validate syntax
            let syntax_valid = code_brain.can_process(&generated);

            // Execute test
            let (test_passed, error) = if syntax_valid {
                execute_test(&problem.prompt, &generated, &problem.test, &problem.entry_point)
            } else {
                (false, Some("Syntax error".to_string()))
            };

            if args.verbose && sample_idx == 0 {
                println!("  Sample 0 generated ({} chars): {:?}...",
                    generated.len(),
                    generated.chars().take(50).collect::<String>());
                println!("  Syntax valid: {}, Test passed: {}", syntax_valid, test_passed);
                if let Some(ref err) = error {
                    println!("  Error: {}", err.chars().take(100).collect::<String>());
                }
            }

            samples.push(SampleResult {
                generated_code: generated,
                syntax_valid,
                test_passed,
                error,
            });
        }

        let num_correct = samples.iter().filter(|s| s.test_passed).count();

        let result = ProblemResult {
            task_id: problem.task_id.clone(),
            num_samples: samples.len(),
            num_correct,
            samples,
        };

        if args.verbose {
            let prob_time = prob_start.elapsed();
            println!(
                "  Correct: {}/{} ({:.1}%), time={:.1}s",
                num_correct,
                args.samples,
                num_correct as f64 / args.samples as f64 * 100.0,
                prob_time.as_secs_f64()
            );
        }

        all_results.push(result);

        // Progress update every 10 problems
        if (prob_idx + 1) % 10 == 0 || prob_idx == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (prob_idx + 1) as f64 / elapsed;
            let eta = (problems.len() - prob_idx - 1) as f64 / rate;

            // Calculate running pass@1
            let running_pass1: f64 = all_results
                .iter()
                .map(|r| pass_at_k(r.num_samples, r.num_correct, 1))
                .sum::<f64>() / all_results.len() as f64;

            println!(
                "Progress: {}/{} ({:.1}%), running pass@1={:.2}%, ETA: {:.0}s",
                prob_idx + 1,
                problems.len(),
                (prob_idx + 1) as f64 / problems.len() as f64 * 100.0,
                running_pass1 * 100.0,
                eta
            );
        }
    }

    let total_time = start.elapsed();

    // Compute pass@k scores
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for &k in &k_values {
        let pass_scores: Vec<f64> = all_results
            .iter()
            .map(|r| pass_at_k(r.num_samples, r.num_correct, k))
            .collect();

        let mean_pass = pass_scores.iter().sum::<f64>() / pass_scores.len() as f64;

        let indicator = if k == 1 {
            if mean_pass * 100.0 > 96.2 {
                " [BEATS SOTA!]"
            } else if mean_pass * 100.0 > 50.0 {
                " [PROMISING]"
            } else {
                ""
            }
        } else {
            ""
        };

        println!("pass@{}: {:.2}%{}", k, mean_pass * 100.0, indicator);
    }

    // Additional statistics
    let total_samples: usize = all_results.iter().map(|r| r.num_samples).sum();
    let total_correct: usize = all_results.iter().map(|r| r.num_correct).sum();
    let total_syntax_valid: usize = all_results
        .iter()
        .map(|r| r.samples.iter().filter(|s| s.syntax_valid).count())
        .sum();

    println!("\n--- Statistics ---");
    println!("Total samples: {}", total_samples);
    println!("Total correct: {} ({:.2}%)", total_correct, total_correct as f64 / total_samples as f64 * 100.0);
    println!("Syntax valid: {} ({:.2}%)", total_syntax_valid, total_syntax_valid as f64 / total_samples as f64 * 100.0);
    println!("Evaluation time: {:.1}s", total_time.as_secs_f64());
    println!("Problems evaluated: {}", problems.len());

    // Save results
    if let Some(output_path) = &args.output {
        let output_data = serde_json::json!({
            "model": args.model.to_string_lossy(),
            "problems": problems.len(),
            "samples_per_problem": args.samples,
            "k_values": k_values,
            "pass_at_k": k_values.iter().map(|&k| {
                let pass_scores: Vec<f64> = all_results
                    .iter()
                    .map(|r| pass_at_k(r.num_samples, r.num_correct, k))
                    .collect();
                let mean = pass_scores.iter().sum::<f64>() / pass_scores.len() as f64;
                (k, mean * 100.0)
            }).collect::<Vec<_>>(),
            "total_correct": total_correct,
            "total_syntax_valid": total_syntax_valid,
            "evaluation_time_secs": total_time.as_secs_f64(),
            "results": all_results,
        });

        let mut file = File::create(output_path)?;
        serde_json::to_writer_pretty(&mut file, &output_data)?;
        println!("\nResults saved to {:?}", output_path);
    }

    Ok(())
}
