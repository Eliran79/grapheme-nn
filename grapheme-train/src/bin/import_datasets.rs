//! Import External Datasets for GRAPHEME AGI Training - CORTEX MESH EDITION
//!
//! Converts multiple domain datasets into GRAPHEME knowledge base format:
//! - GSM8K: Math word problems
//! - SQuAD: Reading comprehension
//! - CODE: Programming patterns (HumanEval, MBPP)
//! - LAW: Legal cases and citations (CaseLaw)
//! - VISION: Image descriptions (COCO captions)
//!
//! Usage:
//!   import_datasets --gsm8k data/external/gsm8k --output checkpoints/math_kb.json
//!   import_datasets --code data/external/code --output checkpoints/code_kb.json
//!   import_datasets --law data/external/law --output checkpoints/law_kb.json
//!   import_datasets --generate-synthetic --output checkpoints/synthetic_kb.json

use clap::Parser;
use grapheme_train::{GraphKnowledgeBase, KnowledgeEntry};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "import_datasets")]
#[command(about = "Import external datasets into GRAPHEME knowledge base (CORTEX MESH)")]
struct Args {
    /// Path to GSM8K directory
    #[arg(long)]
    gsm8k: Option<PathBuf>,

    /// Path to SQuAD directory
    #[arg(long)]
    squad: Option<PathBuf>,

    /// Path to Code dataset directory (HumanEval/MBPP format)
    #[arg(long)]
    code: Option<PathBuf>,

    /// Path to Law dataset directory (CaseLaw format)
    #[arg(long)]
    law: Option<PathBuf>,

    /// Path to Vision dataset directory (COCO captions format)
    #[arg(long)]
    vision: Option<PathBuf>,

    /// Generate synthetic datasets for all cortices
    #[arg(long)]
    generate_synthetic: bool,

    /// Output knowledge base file
    #[arg(short, long)]
    output: PathBuf,

    /// Maximum entries per dataset
    #[arg(long, default_value = "5000")]
    max_entries: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

// GSM8K format
#[derive(Deserialize)]
struct Gsm8kEntry {
    question: String,
    answer: String,
}

// SQuAD format
#[derive(Deserialize)]
struct SquadData {
    data: Vec<SquadArticle>,
}

#[derive(Deserialize)]
struct SquadArticle {
    title: String,
    paragraphs: Vec<SquadParagraph>,
}

#[derive(Deserialize)]
struct SquadParagraph {
    context: String,
    qas: Vec<SquadQA>,
}

#[derive(Deserialize)]
struct SquadQA {
    question: String,
    answers: Vec<SquadAnswer>,
    is_impossible: Option<bool>,
}

#[derive(Deserialize)]
struct SquadAnswer {
    text: String,
}

// ============================================================================
// CODE DATASET FORMATS (HumanEval, MBPP)
// ============================================================================

#[derive(Deserialize)]
struct HumanEvalEntry {
    task_id: String,
    prompt: String,
    canonical_solution: String,
    entry_point: String,
}

#[derive(Deserialize)]
struct MbppEntry {
    task_id: i32,
    text: String,
    code: String,
}

// ============================================================================
// LAW DATASET FORMATS (CaseLaw, legal QA)
// ============================================================================

#[derive(Deserialize)]
struct LegalCaseEntry {
    case_name: Option<String>,
    citation: Option<String>,
    holding: Option<String>,
    facts: Option<String>,
}

// ============================================================================
// VISION DATASET FORMATS (COCO captions)
// ============================================================================

#[derive(Deserialize)]
struct CocoAnnotations {
    annotations: Vec<CocoCaption>,
}

#[derive(Deserialize)]
struct CocoCaption {
    image_id: i64,
    caption: String,
}

/// Extract final answer from GSM8K format (after ####)
fn extract_gsm8k_answer(answer: &str) -> String {
    if let Some(idx) = answer.rfind("####") {
        answer[idx + 4..].trim().to_string()
    } else {
        // Try to find the last number
        answer.split_whitespace()
            .filter(|s| s.parse::<f64>().is_ok() || s.chars().all(|c| c.is_numeric() || c == '.' || c == '-'))
            .last()
            .unwrap_or("unknown")
            .to_string()
    }
}

/// Import GSM8K dataset
fn import_gsm8k(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("train.jsonl");
    if !train_path.exists() {
        println!("GSM8K train.jsonl not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open GSM8K");
    let reader = BufReader::new(file);
    let mut count = 0;

    for (i, line) in reader.lines().enumerate() {
        if count >= max {
            break;
        }

        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let entry: Gsm8kEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let answer = extract_gsm8k_answer(&entry.answer);

        if verbose && count < 3 {
            println!("  GSM8K #{}: Q: {}...", i, &entry.question[..entry.question.len().min(50)]);
            println!("          A: {}", answer);
        }

        let kb_entry = KnowledgeEntry::new(
            format!("gsm8k_{}", i),
            "math_word_problem",
            &entry.question,
            &answer,
        )
        .with_confidence(0.9)
        .with_epoch(1);

        kb.add(kb_entry);
        count += 1;
    }

    println!("Imported {} GSM8K entries", count);
    count
}

/// Import SQuAD dataset
fn import_squad(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("train-v2.0.json");
    if !train_path.exists() {
        println!("SQuAD train-v2.0.json not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open SQuAD");
    let data: SquadData = serde_json::from_reader(file).expect("Failed to parse SQuAD");

    let mut count = 0;
    let mut qa_id = 0;

    'outer: for article in &data.data {
        for para in &article.paragraphs {
            for qa in &para.qas {
                if count >= max {
                    break 'outer;
                }

                // Skip impossible questions
                if qa.is_impossible.unwrap_or(false) || qa.answers.is_empty() {
                    continue;
                }

                let answer = &qa.answers[0].text;

                if verbose && count < 3 {
                    println!("  SQuAD #{}: Q: {}...", qa_id, &qa.question[..qa.question.len().min(50)]);
                    println!("          A: {}", answer);
                }

                // Create shorter context-based question
                let question = format!("{}", qa.question);

                let kb_entry = KnowledgeEntry::new(
                    format!("squad_{}", qa_id),
                    "reading_comprehension",
                    &question,
                    answer,
                )
                .with_confidence(0.95)
                .with_epoch(1);

                kb.add(kb_entry);
                count += 1;
                qa_id += 1;
            }
        }
    }

    println!("Imported {} SQuAD entries", count);
    count
}

// ============================================================================
// CODE CORTEX DATASET IMPORT
// ============================================================================

/// Import HumanEval dataset for code generation
fn import_humaneval(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("HumanEval.jsonl");
    if !train_path.exists() {
        println!("HumanEval.jsonl not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open HumanEval");
    let reader = BufReader::new(file);
    let mut count = 0;

    for line in reader.lines() {
        if count >= max {
            break;
        }

        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let entry: HumanEvalEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if verbose && count < 3 {
            println!("  HumanEval #{}: {}", entry.task_id, entry.entry_point);
        }

        let kb_entry = KnowledgeEntry::new(
            format!("humaneval_{}", entry.task_id),
            "code_generation",
            &entry.prompt,
            &entry.canonical_solution,
        )
        .with_confidence(0.95)
        .with_epoch(1);

        kb.add(kb_entry);
        count += 1;
    }

    println!("Imported {} HumanEval entries", count);
    count
}

/// Import MBPP dataset for code generation
fn import_mbpp(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("mbpp.jsonl");
    if !train_path.exists() {
        println!("mbpp.jsonl not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open MBPP");
    let reader = BufReader::new(file);
    let mut count = 0;

    for line in reader.lines() {
        if count >= max {
            break;
        }

        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let entry: MbppEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if verbose && count < 3 {
            println!("  MBPP #{}: {}...", entry.task_id, &entry.text[..entry.text.len().min(50)]);
        }

        let kb_entry = KnowledgeEntry::new(
            format!("mbpp_{}", entry.task_id),
            "code_generation",
            &entry.text,
            &entry.code,
        )
        .with_confidence(0.9)
        .with_epoch(1);

        kb.add(kb_entry);
        count += 1;
    }

    println!("Imported {} MBPP entries", count);
    count
}

/// Import code datasets
fn import_code(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let mut total = 0;
    total += import_humaneval(path, kb, max, verbose);
    total += import_mbpp(path, kb, max, verbose);
    total
}

// ============================================================================
// LAW CORTEX DATASET IMPORT
// ============================================================================

/// Import legal case dataset
fn import_law(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("cases.jsonl");
    if !train_path.exists() {
        println!("cases.jsonl not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open legal cases");
    let reader = BufReader::new(file);
    let mut count = 0;

    for (i, line) in reader.lines().enumerate() {
        if count >= max {
            break;
        }

        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        let entry: LegalCaseEntry = match serde_json::from_str(&line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Skip entries without essential fields
        let case_name = match entry.case_name {
            Some(n) => n,
            None => continue,
        };
        let holding = entry.holding.unwrap_or_else(|| "No holding".to_string());

        if verbose && count < 3 {
            println!("  Case #{}: {}", i, case_name);
        }

        // Create question about the case
        let question = format!("What is the holding in {}?", case_name);

        let kb_entry = KnowledgeEntry::new(
            format!("law_{}", i),
            "legal_case",
            &question,
            &holding,
        )
        .with_confidence(0.85)
        .with_epoch(1);

        kb.add(kb_entry);
        count += 1;
    }

    println!("Imported {} legal case entries", count);
    count
}

// ============================================================================
// VISION CORTEX DATASET IMPORT
// ============================================================================

/// Import COCO captions dataset
fn import_vision(path: &PathBuf, kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let train_path = path.join("captions_train2017.json");
    if !train_path.exists() {
        println!("captions_train2017.json not found at {:?}", train_path);
        return 0;
    }

    let file = File::open(&train_path).expect("Failed to open COCO captions");
    let data: CocoAnnotations = serde_json::from_reader(file).expect("Failed to parse COCO");

    let mut count = 0;

    for (i, caption) in data.annotations.iter().enumerate() {
        if count >= max {
            break;
        }

        if verbose && count < 3 {
            println!("  Image #{}: {}", caption.image_id, &caption.caption[..caption.caption.len().min(50)]);
        }

        // Create question about the image
        let question = format!("What is in image {}?", caption.image_id);

        let kb_entry = KnowledgeEntry::new(
            format!("vision_{}", i),
            "image_caption",
            &question,
            &caption.caption,
        )
        .with_confidence(0.8)
        .with_epoch(1);

        kb.add(kb_entry);
        count += 1;
    }

    println!("Imported {} COCO caption entries", count);
    count
}

// ============================================================================
// SYNTHETIC DATA GENERATION (for all cortices)
// ============================================================================

/// Generate synthetic data for all cortices
fn generate_synthetic_data(kb: &mut GraphKnowledgeBase, max: usize, verbose: bool) -> usize {
    let mut count = 0;

    // === MATH CORTEX SYNTHETIC DATA ===
    println!("Generating synthetic MATH data...");

    // Generate addition examples
    for a in [5, 10, 15, 25, 50, 100] {
        for b in [2, 3, 4, 5, 7, 10] {
            if count >= max { break; }
            let question = format!("What is {} plus {}?", a, b);
            let answer = format!("{}", a + b);
            kb.add(KnowledgeEntry::new(
                format!("synth_math_add_{}", count),
                "synthetic_math",
                &question,
                &answer,
            ).with_confidence(1.0).with_epoch(0));
            count += 1;
        }
    }

    // Generate multiplication examples
    for a in [5, 10, 15, 25, 50, 100] {
        for b in [2, 3, 4, 5, 7, 10] {
            if count >= max { break; }
            let question = format!("Calculate {} times {}.", a, b);
            let answer = format!("{}", a * b);
            kb.add(KnowledgeEntry::new(
                format!("synth_math_mul_{}", count),
                "synthetic_math",
                &question,
                &answer,
            ).with_confidence(1.0).with_epoch(0));
            count += 1;
        }
    }

    // Generate subtraction examples
    for a in [50, 100, 150, 200] {
        for b in [5, 10, 15, 25] {
            if count >= max { break; }
            let question = format!("What is {} minus {}?", a, b);
            let answer = format!("{}", a - b);
            kb.add(KnowledgeEntry::new(
                format!("synth_math_sub_{}", count),
                "synthetic_math",
                &question,
                &answer,
            ).with_confidence(1.0).with_epoch(0));
            count += 1;
        }
    }

    // Generate division examples
    for a in [20, 50, 100, 200] {
        for b in [2, 4, 5, 10] {
            if count >= max { break; }
            let question = format!("Divide {} by {}.", a, b);
            let answer = format!("{}", a / b);
            kb.add(KnowledgeEntry::new(
                format!("synth_math_div_{}", count),
                "synthetic_math",
                &question,
                &answer,
            ).with_confidence(1.0).with_epoch(0));
            count += 1;
        }
    }
    if verbose { println!("  Generated {} math entries", count); }

    // === CODE CORTEX SYNTHETIC DATA ===
    println!("Generating synthetic CODE data...");
    let code_examples = [
        ("Write a function to add two numbers", "def add(a, b):\n    return a + b"),
        ("Write a function to check if a number is even", "def is_even(n):\n    return n % 2 == 0"),
        ("Write a function to reverse a string", "def reverse(s):\n    return s[::-1]"),
        ("Write a function to find the maximum", "def find_max(lst):\n    return max(lst)"),
        ("Write a function to calculate factorial", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"),
        ("Write a function to check if string is palindrome", "def is_palindrome(s):\n    return s == s[::-1]"),
        ("Write a function to count words", "def count_words(s):\n    return len(s.split())"),
        ("Write a function to sum a list", "def sum_list(lst):\n    return sum(lst)"),
    ];

    let code_start = count;
    for (i, (prompt, code)) in code_examples.iter().enumerate() {
        if count >= max { break; }
        kb.add(KnowledgeEntry::new(
            format!("synth_code_{}", i),
            "synthetic_code",
            *prompt,
            *code,
        ).with_confidence(1.0).with_epoch(0));
        count += 1;
    }
    if verbose { println!("  Generated {} code entries", count - code_start); }

    // === LAW CORTEX SYNTHETIC DATA ===
    println!("Generating synthetic LAW data...");
    let law_examples = [
        ("What is the holding in Brown v. Board of Education?", "Separate educational facilities are inherently unequal and violate the Equal Protection Clause of the 14th Amendment."),
        ("What is the holding in Miranda v. Arizona?", "Police must inform suspects of their rights before custodial interrogation."),
        ("What is the holding in Marbury v. Madison?", "The Supreme Court has the power of judicial review to declare laws unconstitutional."),
        ("What is stare decisis?", "The legal principle that courts should follow precedent from prior decisions."),
        ("What is the difference between plaintiff and defendant?", "Plaintiff brings the lawsuit; defendant is the party being sued."),
        ("What is habeas corpus?", "A legal action requiring a person under arrest to be brought before a judge."),
        ("What is the burden of proof in criminal cases?", "The prosecution must prove guilt beyond a reasonable doubt."),
        ("What is due process?", "The right to fair treatment through the judicial system."),
    ];

    let law_start = count;
    for (i, (question, answer)) in law_examples.iter().enumerate() {
        if count >= max { break; }
        kb.add(KnowledgeEntry::new(
            format!("synth_law_{}", i),
            "synthetic_law",
            *question,
            *answer,
        ).with_confidence(1.0).with_epoch(0));
        count += 1;
    }
    if verbose { println!("  Generated {} law entries", count - law_start); }

    // === VISION CORTEX SYNTHETIC DATA ===
    println!("Generating synthetic VISION data...");
    let vision_examples = [
        ("Describe a cat", "A small domesticated feline with soft fur, whiskers, and pointed ears."),
        ("Describe a dog", "A domesticated mammal with four legs, a tail, and often used as a pet."),
        ("Describe a car", "A four-wheeled motor vehicle designed for transportation of people."),
        ("Describe a tree", "A perennial plant with a trunk, branches, and leaves that grows tall."),
        ("Describe a house", "A building for human habitation with walls, roof, and living spaces."),
        ("Describe the ocean", "A vast body of salt water covering most of Earth's surface."),
        ("Describe a mountain", "A large natural elevation of earth's surface rising high above sea level."),
        ("Describe a flower", "The reproductive structure of a plant, often colorful and fragrant."),
    ];

    let vision_start = count;
    for (i, (prompt, desc)) in vision_examples.iter().enumerate() {
        if count >= max { break; }
        kb.add(KnowledgeEntry::new(
            format!("synth_vision_{}", i),
            "synthetic_vision",
            *prompt,
            *desc,
        ).with_confidence(1.0).with_epoch(0));
        count += 1;
    }
    if verbose { println!("  Generated {} vision entries", count - vision_start); }

    println!("Generated {} total synthetic entries", count);
    count
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Dataset Importer - CORTEX MESH EDITION");
    println!("================================================\n");

    let mut kb = GraphKnowledgeBase::new();
    let mut total = 0;

    // Import GSM8K (Math Cortex)
    if let Some(ref gsm8k_path) = args.gsm8k {
        println!("[Math Cortex] Importing GSM8K from {:?}...", gsm8k_path);
        total += import_gsm8k(gsm8k_path, &mut kb, args.max_entries, args.verbose);
    }

    // Import SQuAD (Text/NLP Cortex)
    if let Some(ref squad_path) = args.squad {
        println!("\n[Text Cortex] Importing SQuAD from {:?}...", squad_path);
        total += import_squad(squad_path, &mut kb, args.max_entries, args.verbose);
    }

    // Import Code (Code Cortex)
    if let Some(ref code_path) = args.code {
        println!("\n[Code Cortex] Importing code datasets from {:?}...", code_path);
        total += import_code(code_path, &mut kb, args.max_entries, args.verbose);
    }

    // Import Law (Law Cortex)
    if let Some(ref law_path) = args.law {
        println!("\n[Law Cortex] Importing legal datasets from {:?}...", law_path);
        total += import_law(law_path, &mut kb, args.max_entries, args.verbose);
    }

    // Import Vision (Vision Cortex)
    if let Some(ref vision_path) = args.vision {
        println!("\n[Vision Cortex] Importing vision datasets from {:?}...", vision_path);
        total += import_vision(vision_path, &mut kb, args.max_entries, args.verbose);
    }

    // Generate synthetic data for all cortices
    if args.generate_synthetic {
        println!("\n[All Cortices] Generating synthetic training data...");
        total += generate_synthetic_data(&mut kb, args.max_entries, args.verbose);
    }

    if total == 0 {
        println!("No datasets imported.");
        println!("\nAvailable options:");
        println!("  --gsm8k <path>      Math word problems");
        println!("  --squad <path>      Reading comprehension");
        println!("  --code <path>       Programming problems (HumanEval/MBPP)");
        println!("  --law <path>        Legal cases");
        println!("  --vision <path>     Image captions (COCO)");
        println!("  --generate-synthetic Generate synthetic data for all cortices");
        return Ok(());
    }

    // Save knowledge base
    println!("\nSaving to {:?}...", args.output);
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    kb.save(&args.output)?;

    // Print stats
    let stats = kb.stats();
    println!("\n╔════════════════════════════════════════════════╗");
    println!("║        CORTEX MESH KNOWLEDGE BASE              ║");
    println!("╠════════════════════════════════════════════════╣");
    println!("║  Total entries: {:>30} ║", stats.total_entries);
    println!("║  Avg question nodes: {:>25.1} ║", stats.avg_question_nodes);
    println!("║  Avg answer nodes: {:>27.1} ║", stats.avg_answer_nodes);
    println!("╠════════════════════════════════════════════════╣");
    println!("║  By Cortex/Topic:                              ║");
    for (topic, count) in &stats.entries_by_topic {
        println!("║    • {:20}: {:>5} entries       ║", topic, count);
    }
    println!("╚════════════════════════════════════════════════╝");

    println!("\nDone! Use with:");
    println!("  cargo run -p grapheme-train --bin agi_infer -- \\");
    println!("    --model checkpoints/checkpoint_level1_final.json \\");
    println!("    --kb {:?}", args.output);

    Ok(())
}
