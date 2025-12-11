//! Time Series Training Script for GRAPHEME
//!
//! This script demonstrates training the neuromorphic GRAPHEME architecture
//! on time series data for forecasting tasks.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin train_timeseries
//! cargo run --bin train_timeseries -- --epochs 500 --window-size 15
//! ```

use grapheme_time::{
    create_training_pairs, generate_sine_wave, NormalizationParams, TimeBrain, TimeSeriesConfig,
    TimeSeriesTrainer,
};

/// Training configuration
struct TrainConfig {
    epochs: usize,
    window_size: usize,
    hidden_nodes: usize,
    learning_rate: f32,
    skip_connections: bool,
    skip_interval: usize,
    series_length: usize,
    train_split: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            window_size: 10,
            hidden_nodes: 4,
            learning_rate: 0.01,
            skip_connections: true,
            skip_interval: 3,
            series_length: 500,
            train_split: 0.8,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==========================================");
    println!(" GRAPHEME Time Series Forecasting Demo");
    println!("==========================================\n");

    let config = TrainConfig::default();

    println!("Configuration:");
    println!("  Epochs:          {}", config.epochs);
    println!("  Window size:     {}", config.window_size);
    println!("  Hidden nodes:    {}", config.hidden_nodes);
    println!("  Learning rate:   {}", config.learning_rate);
    println!("  Skip connections: {}", config.skip_connections);
    println!("  Series length:   {}", config.series_length);
    println!();

    // Generate synthetic sine wave data
    println!("Generating synthetic time series (sine wave)...");
    let series = generate_sine_wave(config.series_length, 0.1, 1.0, 0.0);
    println!("  Generated {} data points", series.len());

    // Split into train/test
    let split_idx = (series.len() as f32 * config.train_split) as usize;
    let (train_series, test_series) = series.split_at(split_idx);
    println!("  Train: {} points, Test: {} points", train_series.len(), test_series.len());

    // Create training pairs
    let (train_windows, train_targets) = create_training_pairs(train_series, config.window_size)?;
    let (test_windows, test_targets) = create_training_pairs(test_series, config.window_size)?;
    println!("  Train pairs: {}, Test pairs: {}", train_windows.len(), test_windows.len());
    println!();

    // Create TimeBrain with configuration
    let ts_config = TimeSeriesConfig::default()
        .with_window_size(config.window_size)
        .with_hidden_nodes(config.hidden_nodes)
        .with_skip_connections(config.skip_connections)
        .with_skip_interval(config.skip_interval);

    let mut brain = TimeBrain::new(ts_config);

    // Compute and set normalization parameters from training data
    let norm_params = NormalizationParams::from_data(
        train_series,
        grapheme_time::NormalizationMethod::MinMax,
    );
    brain.set_normalization(norm_params.clone());

    // Create trainer
    let mut trainer = TimeSeriesTrainer::new(brain, config.learning_rate);

    println!("Training...");
    println!("------------------------------------------");

    let mut best_test_mse = f32::INFINITY;

    for epoch in 1..=config.epochs {
        let mut train_loss_sum = 0.0;
        let mut train_count = 0;

        // Train on all windows
        for (window, &target) in train_windows.iter().zip(train_targets.iter()) {
            match trainer.train_step(window, target) {
                Ok(loss) => {
                    train_loss_sum += loss;
                    train_count += 1;
                }
                Err(e) => {
                    eprintln!("Training error: {:?}", e);
                }
            }
        }

        let train_mse = if train_count > 0 {
            train_loss_sum / train_count as f32
        } else {
            0.0
        };

        // Evaluate on test set periodically
        if epoch % 10 == 0 || epoch == 1 || epoch == config.epochs {
            let (test_mse, sample_preds) =
                evaluate(&trainer, &test_windows, &test_targets);

            if test_mse < best_test_mse {
                best_test_mse = test_mse;
            }

            println!(
                "Epoch {:4}/{}: Train MSE = {:.6}, Test MSE = {:.6}{}",
                epoch,
                config.epochs,
                train_mse,
                test_mse,
                if test_mse == best_test_mse { " *" } else { "" }
            );

            // Show sample predictions at certain epochs
            if epoch == 1 || epoch == config.epochs {
                println!("  Sample predictions (first 5):");
                for (i, (pred, actual)) in sample_preds.iter().take(5).enumerate() {
                    println!(
                        "    [{:2}] Predicted: {:+.4}, Actual: {:+.4}, Error: {:+.4}",
                        i,
                        pred,
                        actual,
                        pred - actual
                    );
                }
            }
        }
    }

    println!("------------------------------------------\n");

    // Final evaluation
    let (final_test_mse, final_preds) = evaluate(&trainer, &test_windows, &test_targets);

    // Compute naive baseline (predict previous value)
    let naive_mse: f32 = test_windows
        .iter()
        .zip(test_targets.iter())
        .map(|(window, &target)| {
            let prev = window.last().copied().unwrap_or(0.0);
            (prev - target).powi(2)
        })
        .sum::<f32>()
        / test_windows.len() as f32;

    println!("Final Results:");
    println!("  Final Test MSE: {:.6}", final_test_mse);
    println!("  Best Test MSE:  {:.6}", best_test_mse);
    println!("  Naive Baseline: {:.6}", naive_mse);
    println!(
        "  Improvement:    {:.2}%",
        (1.0 - final_test_mse / naive_mse) * 100.0
    );
    println!();

    // Show prediction quality
    let good_preds = final_preds
        .iter()
        .filter(|(pred, actual)| (pred - actual).abs() < 0.1)
        .count();
    let total_preds = final_preds.len();

    println!("Prediction Quality:");
    println!(
        "  Predictions within 0.1 of actual: {}/{} ({:.1}%)",
        good_preds,
        total_preds,
        (good_preds as f32 / total_preds as f32) * 100.0
    );

    if final_test_mse < naive_mse {
        println!("\nModel beats naive baseline.");
    } else {
        println!("\nModel did not beat naive baseline (may need more training or tuning).");
    }

    println!("\nTraining complete.");

    Ok(())
}

/// Evaluate model on test set, returns (MSE, sample_predictions)
fn evaluate(
    trainer: &TimeSeriesTrainer,
    windows: &[Vec<f32>],
    targets: &[f32],
) -> (f32, Vec<(f32, f32)>) {
    let mut mse_sum = 0.0;
    let mut predictions = Vec::new();

    for (window, &target) in windows.iter().zip(targets.iter()) {
        match trainer.predict(window) {
            Ok(pred) => {
                let error = (pred - target).powi(2);
                mse_sum += error;
                predictions.push((pred, target));
            }
            Err(_) => {
                // Use naive prediction on error
                let prev = window.last().copied().unwrap_or(0.0);
                let error = (prev - target).powi(2);
                mse_sum += error;
                predictions.push((prev, target));
            }
        }
    }

    let mse = if !windows.is_empty() {
        mse_sum / windows.len() as f32
    } else {
        0.0
    };

    (mse, predictions)
}
