# Spectrue Engine

**Open Source AI Fact-Checking Core**

The transparent, hallucination-resistant analysis engine behind Spectrue. 
This library provides the core logic for multi-agent fact-checking, RAG (Retrieval-Augmented Generation), and deep verification.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

## Features

- **Multi-Agent Architecture**: Orchestrates Oracle, Analyst, and Verifier agents.
- **Smart Waterfall Search**: Optimized search strategy (Oracle → RAG → Tier 1 → Deep Check).
- **Hallucination Resistance**: Strict source verification with 'Aletheia-X' prompts.
- **Content-Aware Localization**: Detects content language and uses native sources.
- **RGBA Analysis**: Returns orthogonal scores for Relevance, Veracity, Bias, and Authority.

## Installation

```bash
pip install spectrue-core-engine
```

## Usage

```python
from spectrue_core.engine import SpectrueEngine
from spectrue_core.config import SpectrueConfig

# Initialize configuration
config = SpectrueConfig(
    openai_api_key="sk-...",
    tavily_api_key="tvly-..."
)

# Initialize engine
engine = SpectrueEngine(config)

# Analyze a claim
result = await engine.analyze(
    text="NASA discovered a new moon orbiting Earth.",
    context_lang="en"
)

print(result.veracity_score)
print(result.rationale)
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the **GNU Affero General Public License v3 (AGPLv3)** - see the [LICENSE](LICENSE) file for details.
