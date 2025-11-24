<div align="center">
  <img src="assets/img/banner.png" alt="Kodegen AI Banner" width="100%" />
</div>

# kodegen-tools-reasoner

**Memory-efficient, Blazing-Fast MCP tools for code generation agents with advanced reasoning capabilities.**

[![License](https://img.shields.io/badge/license-Apache%202.0%20OR%20MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)

## Overview

`kodegen-tools-reasoner` is a high-performance MCP (Model Context Protocol) server that provides sophisticated reasoning strategies for AI agents. It implements multiple search algorithms including Beam Search, Monte Carlo Tree Search (MCTS), and experimental variants designed for complex problem-solving with branching and revision support.

### Key Features

- 🚀 **Multiple Reasoning Strategies**: Beam Search, MCTS, and experimental variants (MCTS 002 Alpha, MCTS 002 Alt Alpha)
- 🧠 **Local Semantic Scoring**: Uses Stella 400M embeddings via Candle for semantic coherence analysis
- 🔄 **Branching & Revision Support**: Explore multiple solution paths from a common parent
- ⚡ **High Performance**: LRU caching with memory pressure detection and lock-free atomic telemetry
- 🌐 **MCP Protocol**: Standard Model Context Protocol for seamless agent integration
- 📊 **Comprehensive Statistics**: Real-time metrics for all reasoning strategies

## Installation

### Prerequisites

- Rust nightly toolchain
- Cargo

### Build from Source

```bash
# Clone the repository
git clone https://github.com/cyrup-ai/kodegen-tools-reasoner.git
cd kodegen-tools-reasoner

# Build the project
cargo build --release

# Run the server
cargo run --release
```

The server will start on the default port (30453) and expose the MCP tool interface.

## Usage

### Running the Server

```bash
# Run in development mode
cargo run

# Run in release mode
cargo run --release

# Run with logging
RUST_LOG=info cargo run
```

### Running Examples

```bash
# Run the comprehensive reasoner demo
cargo run --example reasoner_demo
```

## Reasoning Strategies

### 1. Beam Search (Default)

Breadth-first exploration that maintains the top N paths simultaneously.

**Best for**: Balanced exploration, general problem-solving

**Parameters**:
- `beamWidth`: Number of paths to maintain (default: 3)

```json
{
  "thought": "Analyzing algorithm complexity",
  "thoughtNumber": 1,
  "totalThoughts": 5,
  "nextThoughtNeeded": true,
  "strategyType": "beam_search",
  "beamWidth": 3
}
```

### 2. MCTS (Monte Carlo Tree Search)

Standard MCTS with UCB1/PUCT selection for exploration-exploitation balance.

**Best for**: Decision trees, game-like problems, optimization

**Parameters**:
- `numSimulations`: Number of rollouts per thought (default: 50)

```json
{
  "thought": "Design distributed caching strategy",
  "thoughtNumber": 1,
  "totalThoughts": 3,
  "nextThoughtNeeded": true,
  "strategyType": "mcts",
  "numSimulations": 100
}
```

### 3. MCTS 002 Alpha

MCTS with 10% higher exploration bonus for creative problem-solving.

**Best for**: Creative solutions, exploring novel approaches

```json
{
  "thought": "Optimize database query performance",
  "thoughtNumber": 1,
  "totalThoughts": 2,
  "nextThoughtNeeded": true,
  "strategyType": "mcts_002_alpha",
  "numSimulations": 50
}
```

### 4. MCTS 002 Alt Alpha

MCTS variant that rewards longer, more detailed reasoning paths.

**Best for**: Detailed analysis, thorough explanations

```json
{
  "thought": "Analyze microservices vs monolithic architecture",
  "thoughtNumber": 1,
  "totalThoughts": 2,
  "nextThoughtNeeded": true,
  "strategyType": "mcts_002alt_alpha",
  "numSimulations": 50
}
```

## MCP Tool API

### Tool: `reasoner`

Process thoughts step-by-step with advanced reasoning strategies.

#### Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `thought` | string | Yes | Current reasoning step text |
| `thoughtNumber` | integer | Yes | Current thought index (1-based) |
| `totalThoughts` | integer | Yes | Estimated total thoughts needed |
| `nextThoughtNeeded` | boolean | Yes | Whether more reasoning is required |
| `strategyType` | string | No | Reasoning strategy (default: "beam_search") |
| `beamWidth` | integer | No | Paths to maintain for beam search (default: 3) |
| `numSimulations` | integer | No | MCTS rollouts per thought (default: 50) |
| `parentId` | string | No | Parent node ID for branching thoughts |

#### Output Response

```json
{
  "nodeId": "uuid-v4",
  "thought": "echoed input thought",
  "score": 0.85,
  "depth": 2,
  "isComplete": false,
  "nextThoughtNeeded": true,
  "possiblePaths": 3,
  "bestScore": 0.92,
  "strategyUsed": "beam_search",
  "thoughtNumber": 2,
  "totalThoughts": 5,
  "stats": {
    "totalNodes": 15,
    "averageScore": 0.78,
    "maxDepth": 3,
    "branchingFactor": 2.1,
    "strategyMetrics": {}
  }
}
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────┐
│         MCP Tool Interface              │
│           (reasoner)                    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│           Reasoner                      │
│  (Route requests to strategies)         │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────────────┬──────────────┐
         │                   │              │
┌────────▼────────┐  ┌──────▼──────┐  ┌───▼──────────┐
│  BeamSearch     │  │    MCTS     │  │ MCTS Variants│
│                 │  │             │  │              │
└────────┬────────┘  └──────┬──────┘  └───┬──────────┘
         │                  │              │
         └──────────────────┴──────────────┘
                       │
              ┌────────▼────────┐
              │  BaseStrategy   │
              │ (Stella 400M    │
              │  Embeddings)    │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  StateManager   │
              │ (LRU Cache +    │
              │  HashMap)       │
              └─────────────────┘
```

### Key Components

- **Reasoner**: Orchestrates strategy selection and result aggregation
- **StateManager**: Thread-safe dual-layer caching (LRU + HashMap) with strict lock ordering
- **Strategies**: Pluggable reasoning algorithms (Beam Search, MCTS, variants)
- **BaseStrategy**: Shared embedding-based semantic scoring using local Stella 400M model

## Configuration

Key configuration constants in `src/types.rs`:

```rust
pub const CONFIG: Config = Config {
    beam_width: 3,           // Top paths to maintain
    max_depth: 5,            // Maximum reasoning depth
    min_score: 0.5,          // Viability threshold
    temperature: 0.7,        // Thought diversity
    cache_size: 1000,        // LRU cache entries
    default_strategy: "beam_search",
    num_simulations: 50,     // MCTS rollouts
    // ... additional MCTS weights
};
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

### Code Quality

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check

# Run linter
cargo clippy

# Run linter with all warnings
cargo clippy -- -W clippy::all
```

### Building for Different Targets

```bash
# Build for native target
cargo build --release

# Build for WASM
cargo build --target wasm32-unknown-unknown
```

## Performance

The reasoner is optimized for high-performance operation:

- **LRU Caching**: Hot-path optimization with configurable cache size
- **Atomic Telemetry**: Lock-free cache statistics tracking
- **Memory Pressure Detection**: Automatic cache adjustment using `sysinfo`
- **Concurrent Access**: Thread-safe StateManager with deadlock-preventing lock ordering
- **Local Embeddings**: No network latency - Stella 400M runs locally via Candle

### Cache Statistics

Monitor embedding cache performance:

```rust
let stats = reasoner.get_cache_stats();
println!("Hit rate: {:.2}%", stats.hits as f64 / (stats.hits + stats.misses) as f64 * 100.0);
println!("Memory usage: {} bytes", stats.size_bytes);
println!("Evictions: {}", stats.evictions);
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `cargo fmt` and `cargo clippy`
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Links

- Homepage: [https://kodegen.ai](https://kodegen.ai)
- Repository: [https://github.com/cyrup-ai/kodegen-tools-reasoner](https://github.com/cyrup-ai/kodegen-tools-reasoner)
- MCP Protocol: [Model Context Protocol](https://modelcontextprotocol.io/)

## Acknowledgments

Built by [KODEGEN.ᴀɪ](https://kodegen.ai) for high-performance AI agent reasoning.
