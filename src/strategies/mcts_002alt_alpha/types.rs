use crate::strategies::mcts_002_alpha::PolicyGuidedNode;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Minimum exploration rate to prevent collapse to zero
pub const MIN_EXPLORATION_RATE: f64 = 0.1;

/// Maximum exploration rate to prevent unbounded growth
pub const MAX_EXPLORATION_RATE: f64 = 10.0;

/// Queue implementation for bidirectional search
pub struct Queue<T> {
    items: VecDeque<T>,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            items: VecDeque::new(),
        }
    }

    pub fn enqueue(&mut self, item: T) {
        self.items.push_back(item);
    }

    pub fn dequeue(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn size(&self) -> usize {
        self.items.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidirectionalPolicyNode {
    #[serde(flatten)]
    pub base: PolicyGuidedNode,
    pub g: f64,                 // A* cost from start
    pub h: f64,                 // A* heuristic to goal
    pub f: f64,                 // A* f = g + h
    pub parent: Option<String>, // For path reconstruction
    // Match TypeScript exactly - 'forward' | 'backward' only
    #[serde(rename = "direction")]
    pub direction: Option<String>, // Must be "forward" or "backward" only
    #[serde(rename = "searchDepth")]
    pub search_depth: Option<usize>, // Depth within the search
    #[serde(rename = "meetingPoint")]
    pub meeting_point: Option<bool>, // If true, node is a meeting point
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidirectionalStats {
    #[serde(rename = "forwardExplorationRate")]
    pub forward_exploration_rate: f64,
    #[serde(rename = "backwardExplorationRate")]
    pub backward_exploration_rate: f64,
    #[serde(rename = "meetingPoints")]
    pub meeting_points: usize,
    #[serde(rename = "pathQuality")]
    pub path_quality: f64,
}
