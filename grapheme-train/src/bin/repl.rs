//! Interactive REPL for GRAPHEME model testing.
//!
//! Allows entering expressions for evaluation by a trained model.

use clap::Parser;
use grapheme_core::GraphTransformNet;
use grapheme_train::{quick_eval, quick_symbolic, Pipeline, PipelineMode};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "repl")]
#[command(about = "Interactive GRAPHEME model testing", long_about = None)]
struct Args {
    /// Path to trained model file (optional - uses engine-only mode if not provided)
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Enable verbose output (show intermediate steps)
    #[arg(short, long)]
    verbose: bool,

    /// Show timing information
    #[arg(short, long)]
    timing: bool,
}

struct ReplState {
    pipeline: Pipeline,
    variables: HashMap<String, f64>,
    verbose: bool,
    timing: bool,
    history: Vec<String>,
}

impl ReplState {
    fn new(model: Option<GraphTransformNet>, verbose: bool, timing: bool) -> Self {
        let pipeline = if let Some(net) = model {
            Pipeline::new()
                .with_transform_net(net)
                .with_mode(PipelineMode::Inference)
        } else {
            Pipeline::new().with_mode(PipelineMode::Inference)
        };

        Self {
            pipeline,
            variables: HashMap::new(),
            verbose,
            timing,
            history: Vec::new(),
        }
    }

    fn set_variable(&mut self, name: &str, value: f64) {
        self.variables.insert(name.to_string(), value);
        self.pipeline.bind(name, value);
        println!("  {} = {}", name, value);
    }

    fn clear_variables(&mut self) {
        self.variables.clear();
        self.pipeline.clear_bindings();
        println!("  Variables cleared");
    }

    fn show_variables(&self) {
        if self.variables.is_empty() {
            println!("  No variables defined");
        } else {
            println!("  Variables:");
            for (name, value) in &self.variables {
                println!("    {} = {}", name, value);
            }
        }
    }

    fn evaluate(&mut self, input: &str) {
        self.history.push(input.to_string());

        let start = Instant::now();

        // Try pipeline first
        let result = self.pipeline.process(input);

        let elapsed = start.elapsed();

        if self.verbose {
            println!("  Steps:");
            for step in &result.steps {
                println!("    {}", step);
            }
        }

        if result.success() {
            println!("  = {}", result.result_string());
        } else if !result.errors.is_empty() {
            // Fall back to quick_eval for simple expressions
            if let Some(val) = quick_eval(input) {
                println!("  = {}", val);
            } else if let Some(symbolic) = quick_symbolic(input) {
                println!("  = {}", symbolic);
            } else {
                println!("  Error: {}", result.errors.join(", "));
            }
        } else {
            println!("  No result");
        }

        if self.timing {
            println!("  ({:.3}ms)", elapsed.as_secs_f64() * 1000.0);
        }
    }
}

fn print_help() {
    println!("Commands:");
    println!("  help, h, ?         - Show this help");
    println!("  quit, exit, q      - Exit REPL");
    println!("  let <var> = <val>  - Set a variable (e.g., let x = 5)");
    println!("  vars               - Show all defined variables");
    println!("  clear              - Clear all variables");
    println!("  history            - Show command history");
    println!("  verbose [on|off]   - Toggle verbose mode");
    println!("  timing [on|off]    - Toggle timing display");
    println!("  <expression>       - Evaluate math expression");
    println!();
    println!("Expression formats:");
    println!("  Polish notation:   (+ 2 3), (* (+ 1 2) 4)");
    println!("  Infix notation:    2 + 3, (1 + 2) * 4");
    println!("  Natural language:  what is 2 plus 3");
    println!();
    println!("Math functions:");
    println!("  (sin x), (cos x), (tan x), (log x), (exp x), (sqrt x)");
    println!("  (^ x 2)          - x squared");
    println!("  (derive (^ x 2) x) - differentiate x^2 with respect to x");
    println!("  (integrate x x)  - integrate x with respect to x");
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("GRAPHEME Interactive REPL");
    println!("=========================");

    // Load model if provided
    let model = if let Some(ref model_path) = args.model {
        println!("Loading model from: {:?}", model_path);
        let start = Instant::now();
        let model = GraphTransformNet::load_from_file(model_path)?;
        println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());
        Some(model)
    } else {
        println!("Running in engine-only mode (no model loaded)");
        None
    };

    println!();
    println!("Type 'help' for commands, 'quit' to exit");
    println!();

    let mut state = ReplState::new(model, args.verbose, args.timing);

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("grapheme> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            println!();
            break; // EOF
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Parse commands
        let lower = input.to_lowercase();
        match lower.as_str() {
            "quit" | "exit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "help" | "h" | "?" => {
                print_help();
            }
            "vars" | "variables" => {
                state.show_variables();
            }
            "clear" => {
                state.clear_variables();
            }
            "history" => {
                if state.history.is_empty() {
                    println!("  No history");
                } else {
                    println!("  History:");
                    for (i, cmd) in state.history.iter().enumerate() {
                        println!("    {}: {}", i + 1, cmd);
                    }
                }
            }
            "verbose on" => {
                state.verbose = true;
                println!("  Verbose mode: on");
            }
            "verbose off" => {
                state.verbose = false;
                println!("  Verbose mode: off");
            }
            "verbose" => {
                state.verbose = !state.verbose;
                println!("  Verbose mode: {}", if state.verbose { "on" } else { "off" });
            }
            "timing on" => {
                state.timing = true;
                println!("  Timing display: on");
            }
            "timing off" => {
                state.timing = false;
                println!("  Timing display: off");
            }
            "timing" => {
                state.timing = !state.timing;
                println!("  Timing display: {}", if state.timing { "on" } else { "off" });
            }
            _ => {
                // Check for let statement
                if lower.starts_with("let ") {
                    let rest = &input[4..].trim();
                    if let Some((name, value_str)) = rest.split_once('=') {
                        let name = name.trim();
                        let value_str = value_str.trim();
                        match value_str.parse::<f64>() {
                            Ok(value) => state.set_variable(name, value),
                            Err(_) => {
                                // Try to evaluate the expression
                                if let Some(value) = quick_eval(value_str) {
                                    state.set_variable(name, value);
                                } else {
                                    println!("  Error: Cannot parse '{}' as a number or expression", value_str);
                                }
                            }
                        }
                    } else {
                        println!("  Usage: let <variable> = <value>");
                    }
                } else {
                    // Evaluate as expression
                    state.evaluate(input);
                }
            }
        }
        println!();
    }

    Ok(())
}
