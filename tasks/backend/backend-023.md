---
id: backend-023
title: Implement online continuous learning system
status: todo
priority: high
tags:
- backend
- infrastructure
- learning
dependencies:
- api-003
- backend-022
assignee: developer
created: 2025-12-05T22:24:18.184157773Z
estimate: ~
complexity: 5
area: backend
---

# Implement online continuous learning system

## Context
AGI requires continuous learning - the ability to learn from new data without forgetting old knowledge. Current ML systems require full retraining; GRAPHEME must support incremental updates.

**Key challenges**:
- Catastrophic forgetting (new learning erases old)
- Concept drift (data distribution changes)
- Efficiency (can't retrain from scratch)

## Objectives
- Enable learning from streaming data
- Prevent catastrophic forgetting
- Support knowledge consolidation
- Integrate with memory system
- Persist learning state across sessions

## Tasks
- [ ] Define `OnlineLearner` trait
- [ ] Implement experience replay buffer
- [ ] Implement elastic weight consolidation (EWC)
- [ ] Add knowledge distillation for compression
- [ ] Implement sleep/consolidation phase
- [ ] Create incremental curriculum
- [ ] Add drift detection
- [ ] Integrate with persistence system
- [ ] Write continual learning benchmarks

## Acceptance Criteria
✅ **Continual Learning:**
- New examples don't erase old knowledge
- Performance on old tasks remains stable
- Knowledge transfers to new tasks

✅ **Efficiency:**
- O(1) memory per example (replay buffer bounded)
- Update time proportional to batch, not dataset
- Periodic consolidation, not continuous

✅ **Persistence:**
- Learning state survives restart
- Can resume from any checkpoint
- Incremental saves during learning

## Technical Notes

### Online Learner Trait
```rust
pub trait OnlineLearner: Send + Sync {
    type Example;
    type Model;

    /// Learn from single example (may buffer internally)
    fn learn_one(&mut self, example: Self::Example);

    /// Learn from batch
    fn learn_batch(&mut self, batch: &[Self::Example]) {
        for example in batch {
            self.learn_one(example.clone());
        }
    }

    /// Trigger consolidation (call periodically or on idle)
    fn consolidate(&mut self);

    /// Get current model
    fn model(&self) -> &Self::Model;

    /// Save learning state
    fn save_state<W: Write>(&self, writer: W) -> Result<(), Error>;

    /// Load learning state
    fn load_state<R: Read>(&mut self, reader: R) -> Result<(), Error>;
}
```

### Experience Replay Buffer
```rust
pub struct ReplayBuffer<E> {
    buffer: VecDeque<E>,
    capacity: usize,
    strategy: ReplayStrategy,
}

pub enum ReplayStrategy {
    /// Random sampling (uniform)
    Uniform,
    /// Prioritized by loss/surprise
    Prioritized { priorities: Vec<f32> },
    /// Reservoir sampling for streaming
    Reservoir { seen: usize },
}

impl<E: Clone> ReplayBuffer<E> {
    pub fn add(&mut self, example: E) {
        if self.buffer.len() >= self.capacity {
            self.evict();
        }
        self.buffer.push_back(example);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<E> {
        match &self.strategy {
            ReplayStrategy::Uniform => {
                self.buffer.iter()
                    .choose_multiple(&mut thread_rng(), batch_size)
                    .cloned()
                    .collect()
            }
            ReplayStrategy::Prioritized { priorities } => {
                // Weighted sampling by priority
                let dist = WeightedIndex::new(priorities).unwrap();
                (0..batch_size)
                    .map(|_| self.buffer[dist.sample(&mut thread_rng())].clone())
                    .collect()
            }
            ReplayStrategy::Reservoir { .. } => {
                // Reservoir sampling already maintains representative sample
                self.buffer.iter().take(batch_size).cloned().collect()
            }
        }
    }
}
```

### Elastic Weight Consolidation
```rust
/// Prevent forgetting by penalizing changes to important weights
pub struct EWC {
    /// Fisher information for each parameter
    fisher: HashMap<String, Tensor>,
    /// Previous optimal parameters
    old_params: HashMap<String, Tensor>,
    /// Importance weight
    lambda: f32,
}

impl EWC {
    /// Compute Fisher information from current task
    pub fn compute_fisher(&mut self, model: &Model, dataset: &Dataset) {
        for (name, param) in model.parameters() {
            // Fisher = E[grad log p(y|x)]²
            let fisher = dataset.par_iter()
                .map(|example| {
                    let grad = model.gradient(example);
                    grad.pow(2)
                })
                .reduce(|| Tensor::zeros_like(param), |a, b| a + b)
                / dataset.len() as f32;

            self.fisher.insert(name.clone(), fisher);
            self.old_params.insert(name.clone(), param.clone());
        }
    }

    /// EWC regularization loss
    pub fn penalty(&self, model: &Model) -> f32 {
        let mut loss = 0.0;
        for (name, param) in model.parameters() {
            if let (Some(fisher), Some(old)) = (self.fisher.get(name), self.old_params.get(name)) {
                loss += (fisher * (param - old).pow(2)).sum();
            }
        }
        self.lambda * loss / 2.0
    }
}
```

### Sleep/Consolidation Phase
```rust
pub struct ConsolidationScheduler {
    /// Consolidate after this many examples
    examples_threshold: usize,
    /// Or after this duration
    time_threshold: Duration,
    /// Last consolidation time
    last_consolidation: Instant,
    /// Examples since last consolidation
    examples_since: usize,
}

impl ConsolidationScheduler {
    pub fn should_consolidate(&self) -> bool {
        self.examples_since >= self.examples_threshold
            || self.last_consolidation.elapsed() >= self.time_threshold
    }
}

impl<E> OnlineLearner for GraphemeLearner<E> {
    fn consolidate(&mut self) {
        // 1. Replay old examples mixed with new
        let replay_batch = self.replay_buffer.sample(self.batch_size / 2);
        let recent_batch = self.recent_buffer.drain(..).take(self.batch_size / 2);
        let mixed_batch: Vec<_> = replay_batch.into_iter()
            .chain(recent_batch)
            .collect();

        // 2. Train with EWC penalty
        let loss = self.compute_loss(&mixed_batch) + self.ewc.penalty(&self.model);
        self.optimizer.step(loss);

        // 3. Update Fisher information periodically
        if self.consolidation_count % 10 == 0 {
            self.ewc.compute_fisher(&self.model, &self.replay_buffer);
        }

        // 4. Save checkpoint
        self.checkpoint_manager.save_checkpoint(&self.model, self.epoch)?;

        self.scheduler.last_consolidation = Instant::now();
        self.scheduler.examples_since = 0;
        self.consolidation_count += 1;
    }
}
```

### Drift Detection
```rust
pub struct DriftDetector {
    /// Baseline statistics
    baseline_stats: DataStats,
    /// Current window statistics
    current_stats: DataStats,
    /// Drift threshold (KL divergence)
    threshold: f32,
}

impl DriftDetector {
    pub fn detect_drift(&self) -> Option<DriftReport> {
        let divergence = kl_divergence(&self.baseline_stats, &self.current_stats);
        if divergence > self.threshold {
            Some(DriftReport {
                divergence,
                recommendation: DriftAction::RetriggerFisher,
            })
        } else {
            None
        }
    }
}
```

### Files to Create
- `grapheme-train/src/online.rs`: OnlineLearner trait + impl
- `grapheme-train/src/replay.rs`: Experience replay buffer
- `grapheme-train/src/ewc.rs`: Elastic weight consolidation
- `grapheme-train/src/consolidation.rs`: Sleep/consolidation phase
- `grapheme-train/src/drift.rs`: Drift detection

## Testing
- [ ] Test learning without forgetting (accuracy on old tasks)
- [ ] Test replay buffer maintains diversity
- [ ] Test EWC prevents weight drift
- [ ] Test persistence across restarts
- [ ] Benchmark memory usage during streaming

## Updates
- 2025-12-05: Task created for AGI infrastructure

## Session Handoff (AI: Complete this when marking task done)
**For the next session/agent working on dependent tasks:**

### Dependencies & Integration
- Depends on: api-003 (Memory for replay), backend-022 (persistence)
- Integrates with: Training pipeline
- Enables: True continual learning for AGI
