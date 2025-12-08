mod common;

use anyhow::Context;
use serde_json::json;
use tracing::info;
use kodegen_config::{REASONER};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    info!("Starting tools-reasoner MCP client demo");

    let (conn, mut server) = common::connect_to_local_http_server().await?;

    let workspace_root = common::find_workspace_root()
        .context("Failed to find workspace root")?;
    let log_path = workspace_root.join("tmp/mcp-client/reasoner.log");
    let client = common::LoggingClient::new(conn.client(), log_path)
        .await
        .context("Failed to create logging client")?;

    info!("Connected to server: {:?}", client.server_info());

    let result = run_reasoner_demos(&client).await;

    conn.close().await?;
    server.shutdown().await?;

    result
}

async fn run_reasoner_demos(client: &common::LoggingClient) -> anyhow::Result<()> {
    demo_beam_search(client).await?;
    demo_mcts(client).await?;
    demo_mcts_alpha(client).await?;
    demo_mcts_alt_alpha(client).await?;
    demo_branching(client).await?;
    Ok(())
}

async fn demo_beam_search(client: &common::LoggingClient) -> anyhow::Result<()> {
    info!("=== Beam Search Strategy Demo ===");
    
    let response = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Consider different algorithms for finding the kth largest element in an unsorted array",
            "thought_number": 1,
            "total_thoughts": 3,
            "next_thought_needed": true,
            "strategy_type": "beam_search",
            "beam_width": 3
        })
    ).await.context("Thought 1 failed")?;
    
    let text1 = response.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in Thought 1 response"))?
        .text.clone();
    
    info!("Thought 1 response: {}", text1);
    
    let response2 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "QuickSelect offers O(n) average case - partition around pivot until we find kth position",
            "thought_number": 2,
            "total_thoughts": 3,
            "next_thought_needed": true,
            "strategy_type": "beam_search"
        })
    ).await.context("Thought 2 failed")?;
    
    let text2 = response2.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in Thought 2 response"))?
        .text.clone();
    
    info!("Thought 2 response: {}", text2);
    
    let response3 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "QuickSelect is optimal - O(n) time, O(1) space. Min-heap is O(n log k) but uses O(k) space.",
            "thought_number": 3,
            "total_thoughts": 3,
            "next_thought_needed": false,
            "strategy_type": "beam_search"
        })
    ).await.context("Thought 3 failed")?;
    
    let text3 = response3.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in Thought 3 response"))?
        .text.clone();
    
    info!("✅ Beam search demo complete");
    info!("Final response: {}", text3);
    
    Ok(())
}

async fn demo_mcts(client: &common::LoggingClient) -> anyhow::Result<()> {
    info!("=== MCTS Strategy Demo ===");
    
    let response = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Design a distributed caching strategy that balances consistency and performance",
            "thought_number": 1,
            "total_thoughts": 3,
            "next_thought_needed": true,
            "strategy_type": "mcts",
            "num_simulations": 20
        })
    ).await.context("MCTS Thought 1 failed")?;
    
    let text1 = response.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in MCTS Thought 1 response"))?
        .text.clone();
    
    info!("MCTS Thought 1 response: {}", text1);
    
    let response2 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Use write-through caching with Redis for consistency, CDN edge caching for read performance",
            "thought_number": 2,
            "total_thoughts": 3,
            "next_thought_needed": true,
            "strategy_type": "mcts",
            "num_simulations": 20
        })
    ).await.context("MCTS Thought 2 failed")?;
    
    let text2 = response2.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in MCTS Thought 2 response"))?
        .text.clone();
    
    info!("MCTS Thought 2 response: {}", text2);
    
    let response3 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Implement cache invalidation via pub/sub with TTL fallback for stale data protection",
            "thought_number": 3,
            "total_thoughts": 3,
            "next_thought_needed": false,
            "strategy_type": "mcts",
            "num_simulations": 20
        })
    ).await.context("MCTS Thought 3 failed")?;
    
    let text3 = response3.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in MCTS Thought 3 response"))?
        .text.clone();
    
    info!("✅ MCTS demo complete");
    info!("Final response: {}", text3);
    
    Ok(())
}

async fn demo_mcts_alpha(client: &common::LoggingClient) -> anyhow::Result<()> {
    info!("=== MCTS 002 Alpha Strategy Demo (High Exploration) ===");
    
    let response = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Optimize database query performance from 200ms to under 50ms",
            "thought_number": 1,
            "total_thoughts": 2,
            "next_thought_needed": true,
            "strategy_type": "mcts_002_alpha",
            "num_simulations": 20
        })
    ).await.context("MCTS Alpha Thought 1 failed")?;
    
    let text1 = response.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in MCTS Alpha Thought 1 response"))?
        .text.clone();
    
    info!("MCTS Alpha Thought 1 response: {}", text1);
    
    let response2 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Add composite indexes, implement query result caching, use connection pooling, and partition large tables",
            "thought_number": 2,
            "total_thoughts": 2,
            "next_thought_needed": false,
            "strategy_type": "mcts_002_alpha",
            "num_simulations": 20
        })
    ).await.context("MCTS Alpha Thought 2 failed")?;
    
    let text2 = response2.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in MCTS Alpha Thought 2 response"))?
        .text.clone();
    
    info!("✅ MCTS 002 Alpha demo complete");
    info!("Final response: {}", text2);
    
    Ok(())
}

async fn demo_mcts_alt_alpha(client: &common::LoggingClient) -> anyhow::Result<()> {
    info!("=== MCTS 002 Alt Alpha Strategy Demo (Length Rewarding) ===");
    
    let response = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Analyze the trade-offs between microservices and monolithic architecture for a new e-commerce platform",
            "thought_number": 1,
            "total_thoughts": 2,
            "next_thought_needed": true,
            "strategy_type": "mcts_002alt_alpha",
            "num_simulations": 20
        })
    ).await.context("MCTS Alt Alpha Thought 1 failed")?;
    
    let text1 = response.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in MCTS Alt Alpha Thought 1 response"))?
        .text.clone();
    
    info!("MCTS Alt Alpha Thought 1 response: {}", text1);
    
    let response2 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Start with modular monolith for faster development, clear service boundaries, easier debugging. Migrate to microservices gradually as scale demands, using domain-driven design principles to identify service boundaries naturally",
            "thought_number": 2,
            "total_thoughts": 2,
            "next_thought_needed": false,
            "strategy_type": "mcts_002alt_alpha",
            "num_simulations": 20
        })
    ).await.context("MCTS Alt Alpha Thought 2 failed")?;
    
    let text2 = response2.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in MCTS Alt Alpha Thought 2 response"))?
        .text.clone();
    
    info!("✅ MCTS 002 Alt Alpha demo complete");
    info!("Final response: {}", text2);
    
    Ok(())
}

async fn demo_branching(client: &common::LoggingClient) -> anyhow::Result<()> {
    info!("=== Branching Demo (Multiple Solution Approaches) ===");
    
    let parent_response = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "We need to design a real-time collaborative text editor similar to Google Docs",
            "thought_number": 1,
            "total_thoughts": 5,
            "next_thought_needed": true,
            "strategy_type": "beam_search"
        })
    ).await.context("Parent thought failed")?;
    
    let parent_text = parent_response.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in parent response"))?
        .text.clone();
        
    let parent: serde_json::Value = serde_json::from_str(&parent_text)
        .context("Failed to parse parent response JSON")?;
    
    let parent_id = parent["nodeId"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No nodeId in parent response"))?;
    
    info!("Parent node ID: {}", parent_id);
    
    let branch1 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Approach 1: Use Operational Transformation (OT) like Google Docs - transforms operations to resolve conflicts, proven at scale, complex implementation",
            "thought_number": 2,
            "total_thoughts": 5,
            "next_thought_needed": true,
            "parent_id": parent_id,
            "strategy_type": "beam_search"
        })
    ).await.context("Branch 1 failed")?;
    
    let branch1_text = branch1.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in Branch 1 response"))?
        .text.clone();
    
    info!("Branch 1 (OT approach): {}", branch1_text);
    
    let branch2 = client.call_tool(
        kodegen_config::REASONER,
        json!({
            "thought": "Approach 2: Use CRDT (Conflict-free Replicated Data Types) - mathematically guaranteed convergence, simpler to reason about, larger payload size",
            "thought_number": 2,
            "total_thoughts": 5,
            "next_thought_needed": true,
            "parent_id": parent_id,
            "strategy_type": "beam_search"
        })
    ).await.context("Branch 2 failed")?;
    
    let branch2_text = branch2.content
        .first()
        .and_then(|c| c.as_text())
        .ok_or_else(|| anyhow::anyhow!("No text content in Branch 2 response"))?
        .text.clone();
    
    info!("Branch 2 (CRDT approach): {}", branch2_text);
    
    info!("✅ Branching demo complete - explored 2 solution paths from common parent");
    
    Ok(())
}
