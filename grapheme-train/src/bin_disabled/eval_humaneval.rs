//! HumanEval Evaluation Harness for GRAPHEME
//!
//! Implements proper HumanEval pass@k evaluation:
//! - Load HumanEval problems
//! - Generate n samples per problem
//! - Execute generated code with test cases
//! - Compute pass@1, pass@10, pass@100 scores
//!
//! HumanEval SOTA (as of 2024): 96.2% pass@1
//! Our target: Beat this with GRAPHEME's compile-and-verify approach!
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin eval_humaneval -- \
//!     --model checkpoints/unified_code_best.json \
//!     --data data/humaneval/problems.jsonl \
//!     --samples 200 --k 1,10,100

use clap::Parser;
use grapheme_code::CodeBrain;
use grapheme_core::{DomainBrain, GraphemeGraph, GraphTransformNet, Learnable};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "eval_humaneval")]
#[command(about = "Evaluate GRAPHEME model on HumanEval benchmark")]
struct Args {
    /// Path to trained model checkpoint
    #[arg(short, long)]
    model: PathBuf,

    /// Path to HumanEval problems JSONL
    #[arg(short, long)]
    data: PathBuf,

    /// Number of samples to generate per problem
    #[arg(short, long, default_value = "200")]
    samples: usize,

    /// K values for pass@k (comma-separated)
    #[arg(short, long, default_value = "1,10,100")]
    k: String,

    /// Temperature for sampling (0 = greedy)
    #[arg(long, default_value = "0.8")]
    temperature: f32,

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

/// Generate code for a prompt using the model
fn generate_code(
    model: &GraphTransformNet,
    prompt: &str,
    _temperature: f32,
) -> String {
    let input_graph = GraphemeGraph::from_text(prompt);
    let (_, pooling_result) = model.forward(&input_graph);
    model.decode(&pooling_result)
}

/// Execute generated code with test cases
fn execute_test(prompt: &str, generated: &str, test_code: &str, entry_point: &str) -> (bool, Option<String>) {
    // Build complete test file
    let full_code = format!(
        "{}\n{}\n\n{}\n\n# Run tests\ncheck({})\n",
        prompt, generated, test_code, entry_point
    );

    // Write to temp file
    let temp_path = format!("/tmp/humaneval_test_{}.py", std::process::id());
    match std::fs::write(&temp_path, &full_code) {
        Ok(_) => {}
        Err(e) => return (false, Some(format!("Failed to write temp file: {}", e))),
    }

    // Execute with Python
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
                (false, Some(stderr))
            }
        }
        Err(e) => (false, Some(format!("Execution failed: {}", e))),
    }
}

/// Compute pass@k using unbiased estimator
///
/// pass@k = 1 - C(n-c, k) / C(n, k)
/// where n = total samples, c = correct samples, k = pass threshold
fn pass_at_k(n: usize, c: usize, k: usize) -> f64 {
    if n < k {
        return if c > 0 { 1.0 } else { 0.0 };
    }

    if c == 0 {
        return 0.0;
    }

    if c >= k {
        // Compute using log-space to avoid overflow
        // pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
        let mut log_prod = 0.0;
        for i in 0..k {
            if n - c < i + 1 {
                // Not enough failures to fill k slots
                return 1.0;
            }
            log_prod += ((n - c - i) as f64).ln() - ((n - i) as f64).ln();
        }
        1.0 - log_prod.exp()
    } else {
        // c < k: need at least c successes in k draws
        // This case is rare for large c
        1.0 - (1..=k).fold(1.0, |acc, i| {
            if i > c { acc } else { acc * (n - c) as f64 / (n - i + 1) as f64 }
        })
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("HumanEval Evaluation Harness for GRAPHEME");
    println!("=========================================");
    println!("Target: Beat SOTA (96.2% pass@1)\n");

    // Parse k values
    let k_values: Vec<usize> = args.k
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    println!("Evaluating pass@k for k={:?}", k_values);

    // Load model
    println!("\nLoading model from {:?}...", args.model);
    let model = GraphTransformNet::load_from_file(&args.model)?;
    println!("Model loaded successfully");

    // Initialize CodeBrain for validation
    let code_brain = CodeBrain::new();

    // Load problems
    println!("\nLoading HumanEval problems from {:?}...", args.data);
    let problems = load_problems(&args.data)?;
    println!("Loaded {} problems", problems.len());

    // Evaluation
    let start = Instant::now();
    let mut all_results: Vec<ProblemResult> = Vec::new();

    for (prob_idx, problem) in problems.iter().enumerate() {
        let prob_start = Instant::now();

        if args.verbose {
            println!("\n--- Problem {}/{}: {} ---", prob_idx + 1, problems.len(), problem.task_id);
            println!("Entry point: {}", problem.entry_point);
        }

        // Generate samples in parallel
        let samples: Vec<SampleResult> = (0..args.samples)
            .into_par_iter()
            .map(|_| {
                let generated = generate_code(&model, &problem.prompt, args.temperature);

                // Validate syntax
                let syntax_valid = code_brain.can_process(&generated);

                // Execute test
                let (test_passed, error) = if syntax_valid {
                    execute_test(&problem.prompt, &generated, &problem.test, &problem.entry_point)
                } else {
                    (false, Some("Syntax error".to_string()))
                };

                SampleResult {
                    generated_code: generated,
                    syntax_valid,
                    test_passed,
                    error,
                }
            })
            .collect();

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
        if (prob_idx + 1) % 10 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (prob_idx + 1) as f64 / elapsed;
            let eta = (problems.len() - prob_idx - 1) as f64 / rate;
            println!(
                "Progress: {}/{} problems ({:.1}%), ETA: {:.0}s",
                prob_idx + 1,
                problems.len(),
                (prob_idx + 1) as f64 / problems.len() as f64 * 100.0,
                eta
            );
        }
    }

    let total_time = start.elapsed();

    // Compute pass@k scores
    println!("\n========== RESULTS ==========\n");

    for &k in &k_values {
        let pass_scores: Vec<f64> = all_results
            .iter()
            .map(|r| pass_at_k(r.num_samples, r.num_correct, k))
            .collect();

        let mean_pass = pass_scores.iter().sum::<f64>() / pass_scores.len() as f64;

        let indicator = if k == 1 {
            if mean_pass * 100.0 > 96.2 {
                " 🏆 BEATS SOTA!"
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
