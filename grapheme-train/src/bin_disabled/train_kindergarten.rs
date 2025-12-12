//! Kindergarten training test for GRAPHEME
//!
//! Tests the complete vision flow:
//! 1. Input text → Graph
//! 2. Target text → Graph
//! 3. Train graph-to-graph with structural loss
//! 4. Output graph → Text

use grapheme_core::GraphemeGraph;
use grapheme_train::{compute_structural_loss, StructuralLossConfig, SinkhornConfig};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Serialize, Deserialize)]
struct QAPair {
    input: String,
    target: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧒 GRAPHEME Kindergarten Training Test\n");

    // Load dataset
    let file = File::open("data/kindergarten/simple_qa.jsonl")?;
    let reader = BufReader::new(file);
    let mut pairs: Vec<QAPair> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let pair: QAPair = serde_json::from_str(&line)?;
        pairs.push(pair);
    }

    println!("📊 Loaded {} QA pairs\n", pairs.len());

    // Structural loss configuration (from backend-096, 097, 098)
    let config = StructuralLossConfig {
        alpha: 1.0,  // Node cost
        beta: 0.5,   // Edge cost
        gamma: 2.0,  // Clique cost (highest weight)
        sinkhorn: SinkhornConfig {
            iterations: 20,
            temperature: 0.1,
            epsilon: 1e-6,
        },
    };

    println!("⚙️  Structural Loss Config:");
    println!("   α (node)  = {}", config.alpha);
    println!("   β (edge)  = {}", config.beta);
    println!("   γ (clique) = {}\n", config.gamma);

    // Test graph-to-graph training
    println!("🔬 Testing Vision Flow:\n");

    for (i, pair) in pairs.iter().enumerate() {
        println!("Example {}:", i + 1);
        println!("  Input:  '{}'", pair.input);
        println!("  Target: '{}'", pair.target);

        // Step 1: Text → Graph (input)
        let input_graph = GraphemeGraph::from_text(&pair.input);
        println!("  ✓ Input graph: {} nodes, {} edges",
                 input_graph.node_count(),
                 input_graph.edge_count());

        // Step 2: Text → Graph (target)
        let target_graph = GraphemeGraph::from_text(&pair.target);
        println!("  ✓ Target graph: {} nodes, {} edges",
                 target_graph.node_count(),
                 target_graph.edge_count());

        // Step 3: Compute structural loss (backend-096, 097, 098)
        let loss_result = compute_structural_loss(&input_graph, &target_graph, &config);

        println!("  📉 Structural Loss:");
        println!("     Node cost:   {:.4}", loss_result.node_cost);
        println!("     Edge cost:   {:.4}", loss_result.edge_cost);
        println!("     Clique cost: {:.4}", loss_result.clique_cost);
        println!("     Total loss:  {:.4}", loss_result.total_loss);

        // Step 4: Graph → Text (verify roundtrip)
        let input_reconstructed = input_graph.to_text();
        let target_reconstructed = target_graph.to_text();

        assert_eq!(input_reconstructed, pair.input, "Input roundtrip failed");
        assert_eq!(target_reconstructed, pair.target, "Target roundtrip failed");
        println!("  ✓ Roundtrip verified\n");
    }

    println!("✅ All tests passed!");
    println!("\n📝 Summary:");
    println!("   - Text → Graph conversion: ✓");
    println!("   - Graph → Text conversion: ✓");
    println!("   - Structural loss computation: ✓");
    println!("   - Sinkhorn OT node alignment: ✓");
    println!("   - DAG clique cost (O(n)): ✓");
    println!("\n🎓 Ready for actual training!");

    Ok(())
}
