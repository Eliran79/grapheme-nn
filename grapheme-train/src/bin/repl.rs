//! Interactive REPL for GRAPHEME model testing.
//!
//! Allows entering expressions for evaluation by a trained model.

use clap::Parser;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "repl")]
#[command(about = "Interactive GRAPHEME model testing", long_about = None)]
struct Args {
    /// Path to trained model file
    #[arg(short, long)]
    model: PathBuf,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Interactive REPL");
    println!("=========================");
    println!("Model: {:?}", args.model);
    println!("Type expressions to evaluate, 'help' for commands, 'quit' to exit\n");

    // TODO: Load model using grapheme_train

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("grapheme> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "quit" | "exit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "help" | "h" | "?" => {
                println!("Commands:");
                println!("  help, h, ?   - Show this help");
                println!("  quit, exit, q - Exit REPL");
                println!("  <expression> - Evaluate math expression");
                println!("\nExamples:");
                println!("  (+ 2 3)");
                println!("  (* (+ 1 2) 4)");
                println!("  (derive (^ x 2) x)");
            }
            _ => {
                // TODO: Implement actual model inference
                println!("[TODO] Model inference not yet implemented");
                println!("Input: {}", input);
                println!("This binary provides the CLI interface for backend-089");
            }
        }
        println!();
    }

    Ok(())
}
