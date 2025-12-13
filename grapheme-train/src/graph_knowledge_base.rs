//! Graph-Based Knowledge Base for GRAPHEME
//!
//! Backend-210: Store learned Q&A pairs with their graph embeddings for retrieval.
//!
//! This module provides TRUE graph-based knowledge storage, not the fake
//! hardcoded cosine similarity workaround. Knowledge entries are stored with:
//! - Original question/answer text
//! - Graph representations (GraphemeGraph)
//! - Learned embeddings from the model
//!
//! Retrieval uses the trained model's graph transformations.

use grapheme_core::{GraphemeGraph, GraphTransformNet};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ============================================================================
// Knowledge Entry
// ============================================================================

/// A single Q&A knowledge entry with graph representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    /// Unique identifier
    pub id: String,
    /// Topic/category
    pub topic: String,
    /// Original question text
    pub question: String,
    /// Original answer text
    pub answer: String,
    /// Question graph node count (for quick filtering)
    pub question_nodes: usize,
    /// Answer graph node count
    pub answer_nodes: usize,
    /// Confidence score (from training loss)
    pub confidence: f32,
    /// Training epoch when learned
    pub learned_epoch: usize,
    /// Number of times retrieved
    pub retrieval_count: usize,
}

impl KnowledgeEntry {
    /// Create a new knowledge entry
    pub fn new(
        id: impl Into<String>,
        topic: impl Into<String>,
        question: impl Into<String>,
        answer: impl Into<String>,
    ) -> Self {
        let question_str = question.into();
        let answer_str = answer.into();

        // Count nodes in graphs
        let q_graph = GraphemeGraph::from_text(&question_str);
        let a_graph = GraphemeGraph::from_text(&answer_str);

        Self {
            id: id.into(),
            topic: topic.into(),
            question: question_str,
            answer: answer_str,
            question_nodes: q_graph.node_count(),
            answer_nodes: a_graph.node_count(),
            confidence: 0.0,
            learned_epoch: 0,
            retrieval_count: 0,
        }
    }

    /// Set confidence from training loss (lower loss = higher confidence)
    pub fn with_confidence(mut self, loss: f32) -> Self {
        // Convert loss to confidence: lower loss = higher confidence
        self.confidence = 1.0 / (1.0 + loss);
        self
    }

    /// Set the epoch when learned
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.learned_epoch = epoch;
        self
    }
}

// ============================================================================
// Knowledge Base Statistics
// ============================================================================

/// Statistics about the knowledge base
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeBaseStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Entries per topic
    pub entries_by_topic: HashMap<String, usize>,
    /// Average confidence across all entries
    pub avg_confidence: f32,
    /// Average question length (nodes)
    pub avg_question_nodes: f32,
    /// Average answer length (nodes)
    pub avg_answer_nodes: f32,
}

// ============================================================================
// Graph Knowledge Base
// ============================================================================

/// Graph-based knowledge base for GRAPHEME
///
/// Stores Q&A pairs with their graph representations and supports
/// retrieval using trained model embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphKnowledgeBase {
    /// All knowledge entries
    entries: Vec<KnowledgeEntry>,
    /// Index: question hash -> entry indices
    #[serde(skip)]
    question_index: HashMap<u64, Vec<usize>>,
    /// Index: topic -> entry indices
    #[serde(skip)]
    topic_index: HashMap<String, Vec<usize>>,
    /// Version for compatibility
    version: String,
}

impl Default for GraphKnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphKnowledgeBase {
    /// Create a new empty knowledge base
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            question_index: HashMap::new(),
            topic_index: HashMap::new(),
            version: "1.0.0".to_string(),
        }
    }

    /// Rebuild indices after deserialization
    fn rebuild_indices(&mut self) {
        self.question_index.clear();
        self.topic_index.clear();

        for (idx, entry) in self.entries.iter().enumerate() {
            // Question hash index
            let hash = Self::hash_question(&entry.question);
            self.question_index.entry(hash).or_default().push(idx);

            // Topic index
            self.topic_index.entry(entry.topic.clone()).or_default().push(idx);
        }
    }

    /// Hash a question for quick lookup
    fn hash_question(question: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        question.to_lowercase().hash(&mut hasher);
        hasher.finish()
    }

    /// Add a knowledge entry
    pub fn add(&mut self, entry: KnowledgeEntry) {
        let idx = self.entries.len();
        let hash = Self::hash_question(&entry.question);
        let topic = entry.topic.clone();

        self.entries.push(entry);
        self.question_index.entry(hash).or_default().push(idx);
        self.topic_index.entry(topic).or_default().push(idx);
    }

    /// Add multiple entries at once
    pub fn add_batch(&mut self, entries: Vec<KnowledgeEntry>) {
        for entry in entries {
            self.add(entry);
        }
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries
    pub fn entries(&self) -> &[KnowledgeEntry] {
        &self.entries
    }

    /// Get entries by topic
    pub fn entries_by_topic(&self, topic: &str) -> Vec<&KnowledgeEntry> {
        self.topic_index
            .get(topic)
            .map(|indices| indices.iter().map(|&i| &self.entries[i]).collect())
            .unwrap_or_default()
    }

    /// Get all topics
    pub fn topics(&self) -> Vec<&str> {
        self.topic_index.keys().map(|s| s.as_str()).collect()
    }

    /// Compute statistics
    pub fn stats(&self) -> KnowledgeBaseStats {
        if self.entries.is_empty() {
            return KnowledgeBaseStats::default();
        }

        let mut stats = KnowledgeBaseStats {
            total_entries: self.entries.len(),
            entries_by_topic: HashMap::new(),
            avg_confidence: 0.0,
            avg_question_nodes: 0.0,
            avg_answer_nodes: 0.0,
        };

        for entry in &self.entries {
            *stats.entries_by_topic.entry(entry.topic.clone()).or_default() += 1;
            stats.avg_confidence += entry.confidence;
            stats.avg_question_nodes += entry.question_nodes as f32;
            stats.avg_answer_nodes += entry.answer_nodes as f32;
        }

        let n = self.entries.len() as f32;
        stats.avg_confidence /= n;
        stats.avg_question_nodes /= n;
        stats.avg_answer_nodes /= n;

        stats
    }

    // ========================================================================
    // Graph-Based Retrieval
    // ========================================================================

    /// Query the knowledge base using graph similarity
    ///
    /// Uses the trained model to encode the query and find similar entries.
    /// This is TRUE graph-based retrieval, not fake cosine similarity.
    pub fn query(
        &mut self,
        query: &str,
        model: &GraphTransformNet,
        top_k: usize,
    ) -> Vec<QueryResult> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Encode query using the trained model
        let query_graph = GraphemeGraph::from_text(query);
        let (_, query_pooled) = model.forward(&query_graph);
        let query_embedding = Self::pool_features(&query_pooled.features);

        // Score all entries
        let mut results: Vec<(usize, f32)> = self.entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                // Encode entry question
                let entry_graph = GraphemeGraph::from_text(&entry.question);
                let (_, entry_pooled) = model.forward(&entry_graph);
                let entry_embedding = Self::pool_features(&entry_pooled.features);

                // Compute similarity
                let similarity = Self::cosine_similarity(&query_embedding, &entry_embedding);
                (idx, similarity)
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K and increment retrieval counts
        results
            .into_iter()
            .take(top_k)
            .map(|(idx, similarity)| {
                self.entries[idx].retrieval_count += 1;
                QueryResult {
                    entry: self.entries[idx].clone(),
                    similarity,
                    rank: 0, // Will be set below
                }
            })
            .enumerate()
            .map(|(rank, mut r)| {
                r.rank = rank + 1;
                r
            })
            .collect()
    }

    /// Pool features into a single vector (mean pooling)
    fn pool_features(features: &Array2<f32>) -> Array1<f32> {
        if features.nrows() == 0 {
            return Array1::zeros(features.ncols().max(1));
        }
        features.mean_axis(ndarray::Axis(0)).unwrap_or_else(|| Array1::zeros(features.ncols()))
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    // ========================================================================
    // Persistence
    // ========================================================================

    /// Save knowledge base to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Load knowledge base from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let json = fs::read_to_string(path)?;
        let mut kb: Self = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        kb.rebuild_indices();
        Ok(kb)
    }

}

// ============================================================================
// Query Result
// ============================================================================

/// Result of a knowledge base query
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// The matched knowledge entry
    pub entry: KnowledgeEntry,
    /// Similarity score (0.0 - 1.0)
    pub similarity: f32,
    /// Rank in results (1-indexed)
    pub rank: usize,
}

impl QueryResult {
    /// Format as display string
    pub fn display(&self) -> String {
        format!(
            "[{}] (sim: {:.3}) Q: {} -> A: {}",
            self.rank,
            self.similarity,
            truncate(&self.entry.question, 40),
            truncate(&self.entry.answer, 60)
        )
    }
}

/// Truncate string with ellipsis
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_knowledge_base() {
        let kb = GraphKnowledgeBase::new();
        assert!(kb.is_empty());
        assert_eq!(kb.len(), 0);
    }

    #[test]
    fn test_add_entry() {
        let mut kb = GraphKnowledgeBase::new();
        let entry = KnowledgeEntry::new(
            "kb_001",
            "math",
            "What is 2+2?",
            "4"
        );
        kb.add(entry);

        assert_eq!(kb.len(), 1);
        assert_eq!(kb.entries()[0].question, "What is 2+2?");
    }

    #[test]
    fn test_entries_by_topic() {
        let mut kb = GraphKnowledgeBase::new();

        kb.add(KnowledgeEntry::new("1", "math", "Q1", "A1"));
        kb.add(KnowledgeEntry::new("2", "science", "Q2", "A2"));
        kb.add(KnowledgeEntry::new("3", "math", "Q3", "A3"));

        let math = kb.entries_by_topic("math");
        assert_eq!(math.len(), 2);

        let science = kb.entries_by_topic("science");
        assert_eq!(science.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut kb = GraphKnowledgeBase::new();

        kb.add(KnowledgeEntry::new("1", "math", "What is pi?", "3.14159")
            .with_confidence(0.5)
            .with_epoch(10));
        kb.add(KnowledgeEntry::new("2", "math", "What is e?", "2.71828")
            .with_confidence(0.3)
            .with_epoch(20));

        let stats = kb.stats();
        assert_eq!(stats.total_entries, 2);
        assert_eq!(*stats.entries_by_topic.get("math").unwrap(), 2);
    }

    #[test]
    fn test_save_load_json() {
        let mut kb = GraphKnowledgeBase::new();
        kb.add(KnowledgeEntry::new("1", "test", "Question?", "Answer!"));

        let path = "/tmp/test_kb.json";
        kb.save(path).unwrap();

        let loaded = GraphKnowledgeBase::load(path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.entries()[0].question, "Question?");

        std::fs::remove_file(path).ok();
    }
}
