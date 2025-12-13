//! Fashion-MNIST Dataset Loader
//!
//! Downloads and parses the Fashion-MNIST dataset for clothing image classification.
//!
//! Dataset: https://github.com/zalandoresearch/fashion-mnist
//! Format: Binary IDX files (same format as MNIST)

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Fashion-MNIST class labels
pub const FASHION_LABELS: [&str; 10] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
];

/// Fashion-MNIST example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FashionMnistExample {
    pub id: String,
    pub label: u8,
    pub label_name: String,
    pub pixels: Vec<f32>,
}

/// Fashion-MNIST dataset statistics
#[derive(Debug, Clone)]
pub struct FashionMnistStats {
    pub total_examples: usize,
    pub by_class: [usize; 10],
    pub avg_pixel_value: f32,
}

impl std::fmt::Display for FashionMnistStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fashion-MNIST Dataset Statistics:")?;
        writeln!(f, "  Total examples: {}", self.total_examples)?;
        writeln!(f, "  By class:")?;
        for (i, count) in self.by_class.iter().enumerate() {
            writeln!(f, "    {}: {} ({})", i, FASHION_LABELS[i], count)?;
        }
        writeln!(f, "  Avg pixel value: {:.3}", self.avg_pixel_value)
    }
}

/// Fashion-MNIST dataset loader
pub struct FashionMnistLoader {
    data_dir: PathBuf,
}

impl FashionMnistLoader {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    /// Load Fashion-MNIST dataset from IDX binary files
    pub fn load(&self, split: &str) -> Result<Vec<FashionMnistExample>, Box<dyn std::error::Error>> {
        let (images_file, labels_file) = match split {
            "train" => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            "test" | "t10k" => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
            _ => return Err(format!("Unknown split: {}. Use 'train' or 'test'", split).into()),
        };

        let images_path = self.data_dir.join(images_file);
        let labels_path = self.data_dir.join(labels_file);

        // Try with .gz extension if uncompressed not found
        let images_path = if images_path.exists() {
            images_path
        } else {
            let gz_path = self.data_dir.join(format!("{}.gz", images_file));
            if gz_path.exists() {
                return Err(format!(
                    "Compressed files found. Please decompress first:\n  gunzip {:?}",
                    gz_path
                ).into());
            }
            return Err(format!(
                "Fashion-MNIST images not found: {:?}. Download from https://github.com/zalandoresearch/fashion-mnist",
                images_path
            ).into());
        };

        let labels_path = if labels_path.exists() {
            labels_path
        } else {
            return Err(format!(
                "Fashion-MNIST labels not found: {:?}",
                labels_path
            ).into());
        };

        // Read images
        let images = self.read_idx_images(&images_path)?;
        let labels = self.read_idx_labels(&labels_path)?;

        if images.len() != labels.len() {
            return Err(format!(
                "Image count ({}) doesn't match label count ({})",
                images.len(),
                labels.len()
            ).into());
        }

        // Combine into examples
        let examples: Vec<FashionMnistExample> = images
            .into_iter()
            .zip(labels.into_iter())
            .enumerate()
            .map(|(idx, (pixels, label))| FashionMnistExample {
                id: format!("fashion_{}_{:05}", split, idx),
                label,
                label_name: FASHION_LABELS[label as usize].to_string(),
                pixels,
            })
            .collect();

        Ok(examples)
    }

    /// Read IDX3 image file (magic number 2051)
    fn read_idx_images(&self, path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header = [0u8; 16];
        reader.read_exact(&mut header)?;

        let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        if magic != 2051 {
            return Err(format!("Invalid magic number: {} (expected 2051)", magic).into());
        }

        let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
        let rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
        let cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

        if rows != 28 || cols != 28 {
            return Err(format!("Unexpected image dimensions: {}x{}", rows, cols).into());
        }

        let image_size = rows * cols;
        let mut images = Vec::with_capacity(num_images);

        for _ in 0..num_images {
            let mut buffer = vec![0u8; image_size];
            reader.read_exact(&mut buffer)?;

            // Normalize pixels to [0, 1]
            let pixels: Vec<f32> = buffer.iter().map(|&p| p as f32 / 255.0).collect();
            images.push(pixels);
        }

        Ok(images)
    }

    /// Read IDX1 label file (magic number 2049)
    fn read_idx_labels(&self, path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header = [0u8; 8];
        reader.read_exact(&mut header)?;

        let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        if magic != 2049 {
            return Err(format!("Invalid magic number: {} (expected 2049)", magic).into());
        }

        let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;

        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)?;

        // Validate labels
        for &label in &labels {
            if label >= 10 {
                return Err(format!("Invalid label: {} (expected 0-9)", label).into());
            }
        }

        Ok(labels)
    }

    /// Convert to GRAPHEME training format (JSONL)
    pub fn to_grapheme_format<P: AsRef<Path>>(
        &self,
        examples: &[FashionMnistExample],
        output_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        for example in examples {
            // Subsample pixels for embedding (28x28 -> subset for efficiency)
            let subsampled: Vec<f32> = example.pixels
                .iter()
                .step_by(4)  // Take every 4th pixel -> 196 values
                .take(64)    // Limit to 64 for embedding
                .copied()
                .collect();

            let training_example = serde_json::json!({
                "id": example.id,
                "domain": "vision",
                "input_type": "image",
                "input": {
                    "pixels": subsampled,
                    "width": 28,
                    "height": 28,
                    "channels": 1
                },
                "expected_output": example.label_name,
                "metadata": {
                    "source": "fashion_mnist",
                    "label": example.label,
                    "tags": ["vision", "classification", "fashion"]
                }
            });

            writeln!(writer, "{}", serde_json::to_string(&training_example)?)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Filter examples by class
    pub fn filter_by_class(&self, examples: &[FashionMnistExample], classes: &[u8]) -> Vec<FashionMnistExample> {
        examples
            .iter()
            .filter(|e| classes.contains(&e.label))
            .cloned()
            .collect()
    }

    /// Get dataset statistics
    pub fn stats(&self, examples: &[FashionMnistExample]) -> FashionMnistStats {
        let mut by_class = [0usize; 10];
        let mut total_pixel_sum = 0.0f64;
        let mut total_pixels = 0usize;

        for example in examples {
            by_class[example.label as usize] += 1;
            total_pixel_sum += example.pixels.iter().map(|&p| p as f64).sum::<f64>();
            total_pixels += example.pixels.len();
        }

        FashionMnistStats {
            total_examples: examples.len(),
            by_class,
            avg_pixel_value: if total_pixels > 0 {
                (total_pixel_sum / total_pixels as f64) as f32
            } else {
                0.0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fashion_labels() {
        assert_eq!(FASHION_LABELS.len(), 10);
        assert_eq!(FASHION_LABELS[0], "T-shirt/top");
        assert_eq!(FASHION_LABELS[9], "Ankle boot");
    }

    #[test]
    fn test_stats_empty() {
        let loader = FashionMnistLoader::new("data/fashion");
        let stats = loader.stats(&[]);
        assert_eq!(stats.total_examples, 0);
        assert_eq!(stats.avg_pixel_value, 0.0);
    }

    #[test]
    fn test_filter_by_class() {
        let loader = FashionMnistLoader::new("data/fashion");

        let examples = vec![
            FashionMnistExample {
                id: "1".to_string(),
                label: 0,
                label_name: "T-shirt/top".to_string(),
                pixels: vec![0.5; 784],
            },
            FashionMnistExample {
                id: "2".to_string(),
                label: 5,
                label_name: "Sandal".to_string(),
                pixels: vec![0.3; 784],
            },
            FashionMnistExample {
                id: "3".to_string(),
                label: 0,
                label_name: "T-shirt/top".to_string(),
                pixels: vec![0.7; 784],
            },
        ];

        let filtered = loader.filter_by_class(&examples, &[0]);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|e| e.label == 0));
    }
}
