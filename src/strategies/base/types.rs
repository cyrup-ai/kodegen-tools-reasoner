//! Type definitions for base reasoning strategy
//!
//! Contains error types, async wrappers, and the Strategy trait.

use crate::types::{ReasoningRequest, ReasoningResponse, StrategyMetrics, ThoughtNode};
use futures::Stream;
use lazy_static::lazy_static;
use regex::Regex;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

lazy_static! {
    /// Regex for logical connectors (therefore, because, if, then, thus, hence, so)
    /// Used in calculate_logical_score() for thought quality evaluation.
    pub static ref LOGICAL_CONNECTORS: Regex =
        Regex::new(r"\b(therefore|because|if|then|thus|hence|so)\b")
            .expect("Failed to compile logical connectors regex");

    /// Regex for mathematical/logical expressions (+, -, *, /, =, <, >)
    /// Used in calculate_logical_score() for thought quality evaluation.
    pub static ref MATH_EXPRESSIONS: Regex =
        Regex::new(r"[+\-*/=<>]")
            .expect("Failed to compile mathematical expressions regex");

    /// Regex for novelty calculation (sentence terminators and reasoning markers)
    /// Used in calculate_novelty() for linguistic complexity scoring.
    /// Made public for use in mcts_002_alpha strategy.
    pub static ref NOVELTY_MARKERS: Regex =
        Regex::new(r"[.!?;]|therefore|because|if|then")
            .expect("Failed to compile novelty markers regex");
}

/// Expected embedding dimension for Stella 400M model
pub const EXPECTED_EMBEDDING_DIM: usize = 400;

/// Memory pressure threshold: trigger aggressive eviction if available memory < 15%
pub const MEMORY_PRESSURE_THRESHOLD: f64 = 0.15;

/// Target cache size reduction during memory pressure (reduce to 25% of max)
pub const MEMORY_PRESSURE_CACHE_REDUCTION: f64 = 0.25;

/// Error type for reasoning operations with severity categorization
#[derive(Debug, Clone)]
pub enum ReasoningError {
    /// Fatal error - cannot continue processing
    Fatal(String),

    /// Recoverable error - can continue with degraded functionality
    Recoverable(String),

    /// Generic error (preserved for backward compatibility)
    Other(String),
}

impl ReasoningError {
    pub fn is_fatal(&self) -> bool {
        matches!(self, ReasoningError::Fatal(_))
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(self, ReasoningError::Recoverable(_))
    }
}

impl fmt::Display for ReasoningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal(msg) => write!(f, "Fatal reasoning error: {}", msg),
            Self::Recoverable(msg) => write!(f, "Recoverable reasoning error: {}", msg),
            Self::Other(msg) => write!(f, "Reasoning error: {}", msg),
        }
    }
}

impl std::error::Error for ReasoningError {}

/// A convenience type alias for reasoning results
pub type ReasoningResult<T> = Result<T, ReasoningError>;

//==============================================================================
// AsyncTask - Generic awaitable type for all single-value operations
//==============================================================================

/// Generic awaitable future for any operation that returns a single value
pub struct AsyncTask<T> {
    rx: oneshot::Receiver<ReasoningResult<T>>,
}

impl<T> AsyncTask<T> {
    /// Creates a new AsyncTask from a receiver
    pub fn new(rx: oneshot::Receiver<ReasoningResult<T>>) -> Self {
        Self { rx }
    }

    /// Creates an AsyncTask from a direct value
    ///
    /// Used in BaseStrategy::clear() default implementation (line 921 in this file).
    /// Rust's dead code analysis cannot trace usage in default trait implementations.
    ///
    /// APPROVED BY DAVID MAPLE on 2025-10-27
    #[allow(dead_code)]
    pub fn from_value(value: T) -> Self {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(value));
        Self { rx }
    }
}

impl<T> Future for AsyncTask<T> {
    type Output = ReasoningResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(ReasoningError::Other("Task failed".into()))),
            Poll::Pending => Poll::Pending,
        }
    }
}

//==============================================================================
// TaskStream - Generic stream type for all multi-value operations
//==============================================================================

/// Generic stream for any operation that returns multiple values
pub struct TaskStream<T> {
    inner: ReceiverStream<ReasoningResult<T>>,
}

impl<T> TaskStream<T> {
    /// Creates a new stream from a receiver
    pub fn new(rx: mpsc::Receiver<ReasoningResult<T>>) -> Self {
        Self {
            inner: ReceiverStream::new(rx),
        }
    }

    /// Creates a stream that produces an error
    pub fn from_error(error: ReasoningError) -> Self {
        let (tx, rx) = mpsc::channel(1);
        let _ = tx.try_send(Err(error));
        Self::new(rx)
    }
}

impl<T> Stream for TaskStream<T> {
    type Item = ReasoningResult<T>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

// Type aliases for backward compatibility with existing code
pub type AsyncPath = AsyncTask<Vec<ThoughtNode>>;

/// Type alias for clear operation results.
///
/// Return type for Strategy::clear() trait method (line 205) and all implementations.
/// Rust's dead code analysis cannot trace trait method return types.
///
/// APPROVED BY DAVID MAPLE on 2025-10-27
#[allow(dead_code)]
pub type ClearedSignal = AsyncTask<()>;

pub type MetricStream = TaskStream<StrategyMetrics>;
pub type Reasoning = TaskStream<ReasoningResponse>;
pub type Metric = StrategyMetrics;

/// Strategy trait without async_trait
pub trait Strategy: Send + Sync {
    /// Process a thought with the selected strategy
    fn process_thought(&self, request: ReasoningRequest) -> Reasoning;

    /// Get the best reasoning path found by this strategy
    fn get_best_path(&self) -> AsyncPath;

    /// Get strategy metrics
    fn get_metrics(&self) -> MetricStream;

    /// Clear strategy state
    ///
    /// Called via trait objects in wrapper strategies (mcts_002alt_alpha.rs:859).
    /// Rust's dead code analysis cannot trace trait object method calls.
    ///
    /// APPROVED BY DAVID MAPLE on 2025-10-27
    #[allow(dead_code)]
    fn clear(&self) -> ClearedSignal;
}
