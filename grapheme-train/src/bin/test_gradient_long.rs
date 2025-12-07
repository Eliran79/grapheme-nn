use grapheme_core::{GraphemeGraph, GraphTransformNet};
use grapheme_train::{compute_structural_loss, StructuralLossConfig, SinkhornConfig};

const EMBED_DIM: usize = 64;

fn main() {
    println!("⬇️  Extended Gradient Descent Test (100 epochs)");
    println!("================================================\n");

    let training_examples = [
        ("abc", "ab"),
        ("xyz", "xy"),
        ("123", "12"),
    ];

    let structural_config = StructuralLossConfig {
        alpha: 1.0,
        beta: 0.5,
        gamma: 2.0,
        sinkhorn: SinkhornConfig::default(),
    };

    let mut model = GraphTransformNet::new(256, EMBED_DIM, 128, 2);
    let lr = 0.001;
    let epochs = 100;

    let initial_loss = compute_average_loss(&model, &training_examples, &structural_config);
    println!("Initial loss: {:.6}\n", initial_loss);

    let mut losses = Vec::new();
    let mut min_loss = f32::INFINITY;
    let mut max_loss = f32::NEG_INFINITY;
    let mut min_epoch = 0;

    for epoch in 0..epochs {
        model.zero_grad();

        for (input, target) in &training_examples {
            let input_graph = GraphemeGraph::from_text(input);
            let target_graph = GraphemeGraph::from_text(target);

            let (predicted, pooling_result) = model.forward(&input_graph);
            let loss_result = compute_structural_loss(&predicted, &target_graph, &structural_config);

            model.backward(&input_graph, &pooling_result, &loss_result.node_gradients, EMBED_DIM);
        }

        model.step(lr);

        let curr_loss = compute_average_loss(&model, &training_examples, &structural_config);
        losses.push(curr_loss);
        
        if curr_loss < min_loss {
            min_loss = curr_loss;
            min_epoch = epoch;
        }
        max_loss = max_loss.max(curr_loss);

        if epoch % 10 == 0 || epoch == epochs - 1 {
            let delta = if epoch > 0 { curr_loss - losses[epoch - 1] } else { 0.0 };
            println!("Epoch {:3}: loss={:.6} (Δ={:+.6})", epoch, curr_loss, delta);
        }
    }

    let final_loss = losses[losses.len() - 1];
    let total_change = final_loss - initial_loss;
    let percent_change = (total_change / initial_loss) * 100.0;

    println!("\n=== Summary ===");
    println!("Initial:  {:.6}", initial_loss);
    println!("Final:    {:.6}", final_loss);
    println!("Minimum:  {:.6} (epoch {})", min_loss, min_epoch);
    println!("Maximum:  {:.6}", max_loss);
    println!("Change:   {:+.6} ({:+.2}%)", total_change, percent_change);
    println!("Best improvement: {:.6} ({:.2}%)", initial_loss - min_loss, ((initial_loss - min_loss) / initial_loss) * 100.0);
    
    if min_loss < initial_loss {
        println!("\n✓ SUCCESS: Loss decreased - gradients working!");
    } else {
        println!("\n✗ No improvement");
    }
}

fn compute_average_loss(
    model: &GraphTransformNet,
    examples: &[(&str, &str)],
    config: &StructuralLossConfig,
) -> f32 {
    let total_loss: f32 = examples
        .iter()
        .map(|(input, target)| {
            let input_graph = GraphemeGraph::from_text(input);
            let target_graph = GraphemeGraph::from_text(target);

            let (predicted, _) = model.forward(&input_graph);
            compute_structural_loss(&predicted, &target_graph, config).total_loss
        })
        .sum();

    total_loss / examples.len() as f32
}
