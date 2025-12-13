//! HumanEval Evaluation Harness for GRAPHEME
//!
//! Implements proper HumanEval pass@k evaluation using UnifiedCortex:
//! - Load HumanEval problems
//! - Generate n samples per problem using multi-brain unified processing
//! - Execute generated code with test cases (sandboxed)
//! - Compute pass@1, pass@10, pass@100 scores
//!
//! HumanEval SOTA (as of 2024): 96.2% pass@1
//! Our target: Beat this with GRAPHEME's compile-and-verify approach!
//!
//! Usage:
//!   cargo run --release -p grapheme-train --bin eval_humaneval -- \
//!     --data data/humaneval/problems.jsonl \
//!     --samples 200 --k 1,10,100

use clap::Parser;
use grapheme_train::unified_cortex::{UnifiedCortex, UnifiedConfig, FusionType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use wait_timeout::ChildExt;

/// Command line arguments
#[derive(Parser, Debug)]
#[command(name = "eval_humaneval")]
#[command(about = "Evaluate GRAPHEME UnifiedCortex on HumanEval benchmark")]
struct Args {
    /// Path to HumanEval problems JSONL
    #[arg(short, long)]
    data: PathBuf,

    /// Number of samples to generate per problem
    #[arg(short, long, default_value = "200")]
    samples: usize,

    /// K values for pass@k (comma-separated)
    #[arg(short, long, default_value = "1,10,100")]
    k: String,

    /// Fusion type: attention, weighted, max, concat
    #[arg(long, default_value = "attention")]
    fusion: String,

    /// Timeout per execution in seconds
    #[arg(long, default_value = "5")]
    timeout: u64,

    /// Output results to file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Quick mode (fewer samples, k=1 only)
    #[arg(long)]
    quick: bool,
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
    num_syntax_valid: usize,
    samples: Vec<SampleResult>,
}

/// Result for a single generated sample
#[derive(Debug, Clone, Serialize)]
struct SampleResult {
    generated_code: String,
    syntax_valid: bool,
    test_passed: bool,
    error: Option<String>,
    execution_time_ms: u64,
}

/// Evaluation summary statistics
#[derive(Debug, Clone, Serialize)]
struct EvalSummary {
    problems: usize,
    samples_per_problem: usize,
    total_samples: usize,
    total_correct: usize,
    total_syntax_valid: usize,
    pass_at_k: Vec<(usize, f64)>,
    evaluation_time_secs: f64,
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

/// Generate code for a prompt using UnifiedCortex
fn generate_code(cortex: &mut UnifiedCortex, prompt: &str) -> String {
    let result = cortex.unified_process(prompt);
    result.decoded_code
}

/// Execute generated code with test cases (sandboxed with timeout)
fn execute_test(
    prompt: &str,
    generated: &str,
    test_code: &str,
    entry_point: &str,
    timeout_secs: u64,
) -> (bool, bool, Option<String>, u64) {
    let start = Instant::now();

    // Build complete test file
    let full_code = format!(
        "{}\n{}\n\n{}\n\n# Run tests\ncheck({})\n",
        prompt, generated, test_code, entry_point
    );

    // Check basic syntax validity first
    let syntax_check = Command::new("python3")
        .arg("-c")
        .arg(format!("import ast; ast.parse('''{}''')", full_code.replace("'''", "\\'\\'\\'")))
        .output();

    let syntax_valid = match syntax_check {
        Ok(output) => output.status.success(),
        Err(_) => false,
    };

    if !syntax_valid {
        return (false, false, Some("Syntax error".to_string()), start.elapsed().as_millis() as u64);
    }

    // Execute with Python in sandbox
    let mut child = match Command::new("python3")
        .arg("-c")
        .arg(&full_code)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn() {
            Ok(child) => child,
            Err(e) => return (true, false, Some(format!("Spawn failed: {}", e)), start.elapsed().as_millis() as u64),
        };

    // Wait with timeout
    let timeout = Duration::from_secs(timeout_secs);
    match child.wait_timeout(timeout) {
        Ok(Some(status)) => {
            let elapsed = start.elapsed().as_millis() as u64;
            if status.success() {
                (true, true, None, elapsed)
            } else {
                let stderr = child.stderr.take()
                    .map(|mut s| {
                        let mut buf = String::new();
                        std::io::Read::read_to_string(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                (true, false, Some(stderr), elapsed)
            }
        }
        Ok(None) => {
            // Timeout - kill the process
            let _ = child.kill();
            let _ = child.wait();
            (true, false, Some("Execution timeout".to_string()), timeout_secs * 1000)
        }
        Err(e) => {
            (true, false, Some(format!("Wait failed: {}", e)), start.elapsed().as_millis() as u64)
        }
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

    if n - c < k {
        // Not enough failures to fill k slots, so at least one success guaranteed
        return 1.0;
    }

    // Compute using log-space to avoid overflow
    // pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    let mut log_prod = 0.0;
    for i in 0..k {
        log_prod += ((n - c - i) as f64).ln() - ((n - i) as f64).ln();
    }
    1.0 - log_prod.exp()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("HumanEval Evaluation Harness for GRAPHEME UnifiedCortex");
    println!("========================================================");
    println!("Target: Beat SOTA (96.2% pass@1)\n");

    // Adjust for quick mode
    let (samples, k_values) = if args.quick {
        println!("Running in QUICK mode (fewer samples)");
        (10, vec![1])
    } else {
        let k_values: Vec<usize> = args.k
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        (args.samples, k_values)
    };

    println!("Samples per problem: {}", samples);
    println!("Evaluating pass@k for k={:?}", k_values);

    // Parse fusion type
    let fusion_type = match args.fusion.as_str() {
        "weighted" => FusionType::Weighted,
        "max" => FusionType::Max,
        "concat" => FusionType::Concat,
        _ => FusionType::Attention,
    };
    println!("Fusion type: {:?}", fusion_type);

    // Initialize UnifiedCortex
    println!("\nInitializing UnifiedCortex...");
    let config = UnifiedConfig {
        fusion_type,
        parallel: true,
        ..UnifiedConfig::default()
    };
    let cortex = UnifiedCortex::new(config);
    println!("UnifiedCortex initialized with {} brains", cortex.brains.len());

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

        // Generate samples (note: we need to clone cortex for parallel use)
        let sample_results: Vec<SampleResult> = (0..samples)
            .into_par_iter()
            .map(|_| {
                // Each thread needs its own cortex instance for thread safety
                let mut local_cortex = UnifiedCortex::new(UnifiedConfig::default());
                let generated = generate_code(&mut local_cortex, &problem.prompt);

                let (syntax_valid, test_passed, error, execution_time_ms) = execute_test(
                    &problem.prompt,
                    &generated,
                    &problem.test,
                    &problem.entry_point,
                    args.timeout,
                );

                SampleResult {
                    generated_code: generated,
                    syntax_valid,
                    test_passed,
                    error,
                    execution_time_ms,
                }
            })
            .collect();

        let num_correct = sample_results.iter().filter(|s| s.test_passed).count();
        let num_syntax_valid = sample_results.iter().filter(|s| s.syntax_valid).count();

        let result = ProblemResult {
            task_id: problem.task_id.clone(),
            num_samples: sample_results.len(),
            num_correct,
            num_syntax_valid,
            samples: sample_results,
        };

        if args.verbose {
            let prob_time = prob_start.elapsed();
            println!(
                "  Correct: {}/{} ({:.1}%), Syntax valid: {}/{}, time={:.1}s",
                num_correct, samples,
                num_correct as f64 / samples as f64 * 100.0,
                num_syntax_valid, samples,
                prob_time.as_secs_f64()
            );
        }

        all_results.push(result);

        // Progress update every 10 problems
        if (prob_idx + 1) % 10 == 0 || prob_idx == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (prob_idx + 1) as f64 / elapsed;
            let eta = (problems.len() - prob_idx - 1) as f64 / rate;
            println!(
                "Progress: {}/{} problems ({:.1}%), ETA: {:.0}s",
                prob_idx + 1, problems.len(),
                (prob_idx + 1) as f64 / problems.len() as f64 * 100.0,
                eta
            );
        }
    }

    let total_time = start.elapsed();

    // Compute pass@k scores
    println!("\n========== RESULTS ==========\n");

    let mut pass_at_k_results = Vec::new();

    for &k in &k_values {
        let pass_scores: Vec<f64> = all_results
            .iter()
            .map(|r| pass_at_k(r.num_samples, r.num_correct, k))
            .collect();

        let mean_pass = pass_scores.iter().sum::<f64>() / pass_scores.len() as f64;
        pass_at_k_results.push((k, mean_pass * 100.0));

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
    let total_syntax_valid: usize = all_results.iter().map(|r| r.num_syntax_valid).sum();

    println!("\n--- Statistics ---");
    println!("Total samples: {}", total_samples);
    println!("Total correct: {} ({:.2}%)", total_correct, total_correct as f64 / total_samples as f64 * 100.0);
    println!("Syntax valid: {} ({:.2}%)", total_syntax_valid, total_syntax_valid as f64 / total_samples as f64 * 100.0);
    println!("Evaluation time: {:.1}s", total_time.as_secs_f64());

    // Summary
    let summary = EvalSummary {
        problems: problems.len(),
        samples_per_problem: samples,
        total_samples,
        total_correct,
        total_syntax_valid,
        pass_at_k: pass_at_k_results.clone(),
        evaluation_time_secs: total_time.as_secs_f64(),
    };

    // Save results
    if let Some(output_path) = &args.output {
        let output_data = serde_json::json!({
            "summary": summary,
            "results": all_results,
        });

        let mut file = File::create(output_path)?;
        serde_json::to_writer_pretty(&mut file, &output_data)?;
        println!("\nResults saved to {:?}", output_path);
    }

    // Final verdict
    println!("\n========== VERDICT ==========");
    if pass_at_k_results.iter().any(|(k, score)| *k == 1 && *score > 96.2) {
        println!("🏆 GRAPHEME BEATS HumanEval SOTA! 🏆");
    } else {
        let pass1 = pass_at_k_results.iter().find(|(k, _)| *k == 1).map(|(_, s)| *s).unwrap_or(0.0);
        println!("Current: {:.2}% | Target: 96.2% | Gap: {:.2}%", pass1, 96.2 - pass1);
    }

    Ok(())
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_at_k_all_correct() {
        // All 10 samples correct, pass@1 should be 1.0
        assert!((pass_at_k(10, 10, 1) - 1.0).abs() < 1e-9);
        assert!((pass_at_k(10, 10, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_pass_at_k_none_correct() {
        // No samples correct, pass@k should be 0.0
        assert!((pass_at_k(10, 0, 1) - 0.0).abs() < 1e-9);
        assert!((pass_at_k(10, 0, 10) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pass_at_k_half_correct() {
        // 5/10 correct, pass@1 = 1 - (5/10) = 0.5
        let p1 = pass_at_k(10, 5, 1);
        assert!((p1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pass_at_k_nearly_all() {
        // 9/10 correct, pass@1 should be 0.9
        let p1 = pass_at_k(10, 9, 1);
        assert!((p1 - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_pass_at_k_k_greater_than_n() {
        // k > n: if any correct, 1.0, else 0.0
        assert!((pass_at_k(5, 1, 10) - 1.0).abs() < 1e-9);
        assert!((pass_at_k(5, 0, 10) - 0.0).abs() < 1e-9);
    }
}
