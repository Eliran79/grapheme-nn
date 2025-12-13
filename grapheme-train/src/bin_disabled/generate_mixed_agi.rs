//! Mixed AGI Dataset Generator
//!
//! Generates diverse training data across all modalities for unified AGI training:
//! - Math: arithmetic expressions
//! - Text: QA pairs
//! - TimeSeries: sequence prediction
//! - Vision: simple pattern classification
//!
//! Backend-168: Mixed Input Dataset Generator

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "generate_mixed_agi")]
#[command(about = "Generate mixed AGI training data across all modalities")]
struct Args {
    /// Number of samples per domain
    #[arg(short, long, default_value = "1000")]
    samples: usize,

    /// Output directory for generated data
    #[arg(short, long, default_value = "data/mixed_agi")]
    output: PathBuf,

    /// Random seed for reproducible generation
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Split into train/val/test (ratios: 0.8/0.1/0.1)
    #[arg(long)]
    split: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Only generate specific domain (math, text, timeseries, vision)
    #[arg(long)]
    domain: Option<String>,
}

/// Training example for mixed AGI dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedExample {
    pub id: String,
    pub domain: String,
    pub input_type: String,
    pub input: InputData,
    pub expected_output: String,
    pub metadata: ExampleMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputData {
    Text { text: String },
    Sequence { values: Vec<f32> },
    Image { width: usize, height: usize, pixels: Vec<f32> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleMetadata {
    pub difficulty: u8,
    pub tags: Vec<String>,
}

/// Simple LCG random number generator for reproducibility
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 1000000) as f32 / 1000000.0
    }

    fn next_range(&mut self, min: i32, max: i32) -> i32 {
        let range = (max - min + 1) as u64;
        min + (self.next_u64() % range) as i32
    }

    fn choose<T: Clone>(&mut self, items: &[T]) -> T {
        let idx = (self.next_u64() as usize) % items.len();
        items[idx].clone()
    }
}

/// Generate math examples
fn generate_math_examples(rng: &mut Rng, count: usize, verbose: bool) -> Vec<MixedExample> {
    let ops = ["+", "-", "*"];
    let mut examples = Vec::new();

    for i in 0..count {
        let a = rng.next_range(1, 50);
        let b = rng.next_range(1, 50);
        let op = rng.choose(&ops);

        let (expr, result) = match op {
            "+" => (format!("{} + {}", a, b), (a + b).to_string()),
            "-" => (format!("{} - {}", a, b), (a - b).to_string()),
            "*" => (format!("{} * {}", a, b), (a * b).to_string()),
            _ => continue,
        };

        examples.push(MixedExample {
            id: format!("math_{:06}", i),
            domain: "math".to_string(),
            input_type: "text".to_string(),
            input: InputData::Text { text: expr },
            expected_output: result,
            metadata: ExampleMetadata {
                difficulty: 1,
                tags: vec!["arithmetic".to_string()],
            },
        });
    }

    if verbose {
        println!("  Generated {} math examples", examples.len());
    }

    examples
}

/// Generate text QA examples
fn generate_text_examples(rng: &mut Rng, count: usize, verbose: bool) -> Vec<MixedExample> {
    let qa_pairs = [
        ("How many legs does a cat have?", "4"),
        ("What color is the sky?", "blue"),
        ("What animal says moo?", "cow"),
        ("What shape is a ball?", "round"),
        ("How many days in a week?", "7"),
        ("What color is grass?", "green"),
        ("How many months in a year?", "12"),
        ("What animal says woof?", "dog"),
        ("What shape is the sun?", "circle"),
        ("How many wheels on a bicycle?", "2"),
        ("What color is snow?", "white"),
        ("What animal has a trunk?", "elephant"),
        ("How many fingers on one hand?", "5"),
        ("What color is a banana?", "yellow"),
        ("What animal says meow?", "cat"),
        ("What shape is a box?", "cube"),
        ("How many hours in a day?", "24"),
        ("What color is an orange?", "orange"),
        ("What animal lays eggs?", "chicken"),
        ("How many sides on a triangle?", "3"),
    ];

    let mut examples = Vec::new();

    for i in 0..count {
        let (q, a) = rng.choose(&qa_pairs);

        examples.push(MixedExample {
            id: format!("text_{:06}", i),
            domain: "text".to_string(),
            input_type: "text".to_string(),
            input: InputData::Text { text: q.to_string() },
            expected_output: a.to_string(),
            metadata: ExampleMetadata {
                difficulty: 1,
                tags: vec!["qa".to_string(), "factoid".to_string()],
            },
        });
    }

    if verbose {
        println!("  Generated {} text examples", examples.len());
    }

    examples
}

/// Generate time series examples
fn generate_timeseries_examples(rng: &mut Rng, count: usize, verbose: bool) -> Vec<MixedExample> {
    let mut examples = Vec::new();

    for i in 0..count {
        let pattern_type = rng.next_range(0, 3);
        let (sequence, next_val) = match pattern_type {
            0 => {
                // Linear increasing: a, a+d, a+2d, ...
                let start = rng.next_range(1, 20) as f32;
                let step = rng.next_range(1, 5) as f32;
                let seq: Vec<f32> = (0..5).map(|j| start + j as f32 * step).collect();
                let next = start + 5.0 * step;
                (seq, next)
            }
            1 => {
                // Linear decreasing
                let start = rng.next_range(20, 50) as f32;
                let step = rng.next_range(1, 5) as f32;
                let seq: Vec<f32> = (0..5).map(|j| start - j as f32 * step).collect();
                let next = start - 5.0 * step;
                (seq, next)
            }
            2 => {
                // Doubling
                let start = rng.next_range(1, 5) as f32;
                let seq: Vec<f32> = (0..5).map(|j| start * 2.0_f32.powi(j)).collect();
                let next = start * 2.0_f32.powi(5);
                (seq, next)
            }
            _ => {
                // Fibonacci-like
                let a = rng.next_range(1, 5) as f32;
                let b = rng.next_range(1, 5) as f32;
                let seq = vec![a, b, a + b, a + 2.0 * b, 2.0 * a + 3.0 * b];
                let next = 3.0 * a + 5.0 * b;
                (seq, next)
            }
        };

        examples.push(MixedExample {
            id: format!("timeseries_{:06}", i),
            domain: "timeseries".to_string(),
            input_type: "sequence".to_string(),
            input: InputData::Sequence { values: sequence },
            expected_output: format!("{:.0}", next_val),
            metadata: ExampleMetadata {
                difficulty: 2,
                tags: vec!["prediction".to_string()],
            },
        });
    }

    if verbose {
        println!("  Generated {} timeseries examples", examples.len());
    }

    examples
}

/// Generate vision examples (4x4 patterns)
fn generate_vision_examples(rng: &mut Rng, count: usize, verbose: bool) -> Vec<MixedExample> {
    // Pre-defined patterns
    let patterns: Vec<(Vec<f32>, &str)> = vec![
        // Vertical line
        (vec![0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0], "vertical"),
        // Horizontal line
        (vec![0.0, 0.0, 0.0, 0.0,
              1.0, 1.0, 1.0, 1.0,
              0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0], "horizontal"),
        // Cross
        (vec![0.0, 1.0, 0.0, 0.0,
              1.0, 1.0, 1.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0], "cross"),
        // Square
        (vec![1.0, 1.0, 1.0, 1.0,
              1.0, 0.0, 0.0, 1.0,
              1.0, 0.0, 0.0, 1.0,
              1.0, 1.0, 1.0, 1.0], "square"),
        // Diagonal
        (vec![1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0], "diagonal"),
        // Dot (center)
        (vec![0.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 1.0, 0.0,
              0.0, 1.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 0.0], "dot"),
        // Corner (top-left)
        (vec![1.0, 1.0, 0.0, 0.0,
              1.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0], "corner"),
        // T-shape
        (vec![1.0, 1.0, 1.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0], "T"),
    ];

    let mut examples = Vec::new();

    for i in 0..count {
        let (base_pixels, label) = rng.choose(&patterns);

        // Add small noise to make each example unique
        let pixels: Vec<f32> = base_pixels
            .iter()
            .map(|&p| (p + rng.next_f32() * 0.1 - 0.05).clamp(0.0, 1.0))
            .collect();

        examples.push(MixedExample {
            id: format!("vision_{:06}", i),
            domain: "vision".to_string(),
            input_type: "image".to_string(),
            input: InputData::Image {
                width: 4,
                height: 4,
                pixels,
            },
            expected_output: label.to_string(),
            metadata: ExampleMetadata {
                difficulty: 2,
                tags: vec!["pattern".to_string(), "classification".to_string()],
            },
        });
    }

    if verbose {
        println!("  Generated {} vision examples", examples.len());
    }

    examples
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("==========================================");
    println!(" GRAPHEME Mixed AGI Dataset Generator");
    println!(" Backend-168: Multi-Modal Training Data");
    println!("==========================================\n");

    // Create output directory
    fs::create_dir_all(&args.output)?;

    let mut rng = Rng::new(args.seed);

    println!("Configuration:");
    println!("  Output directory: {:?}", args.output);
    println!("  Samples per domain: {}", args.samples);
    println!("  Seed: {}", args.seed);
    println!("  Split: {}", args.split);
    if let Some(ref domain) = args.domain {
        println!("  Domain filter: {}", domain);
    }
    println!();

    let start = Instant::now();

    // Generate examples for each domain
    let mut all_examples = Vec::new();

    let domains: Vec<&str> = if let Some(ref d) = args.domain {
        vec![d.as_str()]
    } else {
        vec!["math", "text", "timeseries", "vision"]
    };

    for domain in &domains {
        let domain_start = Instant::now();
        println!("Generating {} examples...", domain);

        let examples = match *domain {
            "math" => generate_math_examples(&mut rng, args.samples, args.verbose),
            "text" => generate_text_examples(&mut rng, args.samples, args.verbose),
            "timeseries" => generate_timeseries_examples(&mut rng, args.samples, args.verbose),
            "vision" => generate_vision_examples(&mut rng, args.samples, args.verbose),
            _ => {
                eprintln!("Unknown domain: {}", domain);
                continue;
            }
        };

        let elapsed = domain_start.elapsed();
        println!(
            "  {} examples in {:.2}s ({:.0} ex/s)",
            examples.len(),
            elapsed.as_secs_f64(),
            examples.len() as f64 / elapsed.as_secs_f64()
        );

        all_examples.extend(examples);
    }

    println!("\nTotal examples: {}", all_examples.len());

    // Save examples
    if args.split {
        // Shuffle examples
        let len = all_examples.len();
        for i in 0..len {
            let j = (rng.next_u64() as usize) % len;
            all_examples.swap(i, j);
        }

        // Split 80/10/10
        let train_end = (len as f32 * 0.8) as usize;
        let val_end = train_end + (len as f32 * 0.1) as usize;

        let train_examples = &all_examples[..train_end];
        let val_examples = &all_examples[train_end..val_end];
        let test_examples = &all_examples[val_end..];

        save_jsonl(&args.output.join("train.jsonl"), train_examples)?;
        save_jsonl(&args.output.join("val.jsonl"), val_examples)?;
        save_jsonl(&args.output.join("test.jsonl"), test_examples)?;

        println!(
            "Saved: train={}, val={}, test={}",
            train_examples.len(),
            val_examples.len(),
            test_examples.len()
        );
    } else {
        save_jsonl(&args.output.join("mixed_agi.jsonl"), &all_examples)?;
        println!("Saved to: {:?}", args.output.join("mixed_agi.jsonl"));
    }

    let elapsed = start.elapsed();
    println!(
        "\nGeneration complete in {:.2}s ({:.0} ex/s total)",
        elapsed.as_secs_f64(),
        all_examples.len() as f64 / elapsed.as_secs_f64()
    );

    println!("\n==========================================");
    println!(" Dataset Statistics");
    println!("==========================================");
    for domain in &domains {
        let count = all_examples.iter().filter(|e| e.domain == *domain).count();
        println!("  {}: {} examples", domain, count);
    }

    println!("\n==========================================");
    println!(" Backend-168 Complete: Mixed AGI Dataset");
    println!("==========================================");

    Ok(())
}

fn save_jsonl(path: &PathBuf, examples: &[MixedExample]) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for example in examples {
        let json = serde_json::to_string(example)?;
        writeln!(writer, "{}", json)?;
    }

    writer.flush()?;
    Ok(())
}
