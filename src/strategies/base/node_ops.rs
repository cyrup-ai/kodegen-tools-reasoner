//! Node operations and retry logic
//!
//! Handles reading, writing, and retrying node operations with the state manager.

use super::BaseStrategy;
use super::types::{AsyncTask, ReasoningError};
use crate::types::ThoughtNode;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

impl BaseStrategy {
    pub fn get_node(&self, id: &str) -> AsyncTask<Option<ThoughtNode>> {
        let state_manager = Arc::clone(&self.state_manager);
        let id = id.to_string();

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            let result = state_manager.get_node(&id).await;
            let _ = tx.send(Ok(result));
        });

        AsyncTask::new(rx)
    }

    pub fn save_node(&self, node: ThoughtNode) -> AsyncTask<()> {
        let state_manager = Arc::clone(&self.state_manager);
        let node = node.clone();

        let (tx, rx) = oneshot::channel();

        tokio::spawn(async move {
            state_manager.save_node(node).await;
            let _ = tx.send(Ok(()));
        });

        AsyncTask::new(rx)
    }

    /// Save node with exponential backoff retry logic
    pub async fn save_node_with_retry(
        &self,
        node: ThoughtNode,
        max_retries: Option<u32>,
    ) -> Result<(), ReasoningError> {
        let max_retries = max_retries.unwrap_or(crate::types::CONFIG.max_retry_attempts);
        let mut attempt = 0;
        let mut delay = Duration::from_millis(crate::types::CONFIG.initial_retry_delay_ms);
        let max_delay = Duration::from_millis(crate::types::CONFIG.max_retry_delay_ms);

        loop {
            match self.save_node(node.clone()).await {
                Ok(_) => return Ok(()),
                Err(e) if e.is_fatal() => {
                    // Fatal errors should not be retried
                    return Err(e);
                }
                Err(e) if attempt < max_retries => {
                    tracing::warn!(
                        "Save node {} failed (attempt {}/{}): {}",
                        node.id,
                        attempt + 1,
                        max_retries,
                        e
                    );
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(max_delay); // Capped exponential backoff
                    attempt += 1;
                }
                Err(e) => {
                    return Err(ReasoningError::Recoverable(format!(
                        "Failed to save node {} after {} attempts: {}",
                        node.id, max_retries, e
                    )));
                }
            }
        }
    }

    /// Get node with retry logic for transient failures
    pub async fn get_node_with_retry(
        &self,
        node_id: &str,
        max_retries: Option<u32>,
    ) -> Result<Option<ThoughtNode>, ReasoningError> {
        let max_retries = max_retries.unwrap_or(crate::types::CONFIG.max_retry_attempts);
        let mut attempt = 0;
        let mut delay = Duration::from_millis(crate::types::CONFIG.initial_retry_delay_ms);
        let max_delay = Duration::from_millis(crate::types::CONFIG.max_retry_delay_ms);

        loop {
            match self.get_node(node_id).await {
                Ok(result) => return Ok(result),
                Err(e) if e.is_fatal() => {
                    // Fatal errors should not be retried
                    return Err(e);
                }
                Err(e) if e.is_recoverable() && attempt < max_retries => {
                    tracing::warn!(
                        "Get node {} failed (attempt {}/{}): {}",
                        node_id,
                        attempt + 1,
                        max_retries,
                        e
                    );
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(max_delay); // Capped exponential backoff
                    attempt += 1;
                }
                Err(e) => {
                    return Err(ReasoningError::Recoverable(format!(
                        "Failed to get node {} after {} attempts: {}",
                        node_id, max_retries, e
                    )));
                }
            }
        }
    }

    /// Helper to log and handle channel send failures
    pub fn log_channel_send_error<T>(
        result: Result<(), tokio::sync::mpsc::error::SendError<T>>,
        context: &str,
    ) {
        if result.is_err() {
            tracing::error!("Failed to send {} (receiver dropped)", context);
        }
    }
}
