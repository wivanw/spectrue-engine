# Spectrue Engine

**Open Source AI Fact-Checking Core**

The transparent, hallucination-resistant analysis engine behind Spectrue. 
This library provides the core logic for multi-agent fact-checking, web-based verification, and deep analysis.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- **Multi-Agent Architecture**: Orchestrates Oracle, Analyst, and Verifier agents
- **Smart Waterfall Search**: Optimized strategy (Oracle → Tier 1 → Deep Dive)
- **Hallucination Resistance**: Strict source verification with 'Aletheia-X' prompts
- **Content-Aware Localization**: Detects content language and uses native sources
- **RGBA Analysis**: Returns orthogonal scores for Relevance, Veracity, Bias, and Authority
- **Cloud-Native**: Fully web-based verification (Tavily + Google Fact Check)

## 📋 Requirements

- **Python**: 3.10 or higher
- **Dependencies**: See [pyproject.toml](pyproject.toml)

## 🚀 Installation

### From PyPI (when published)
```bash
pip install spectrue-engine
```

### From GitHub (Latest)
```bash
pip install git+https://github.com/wivanw/spectrue-engine.git
```

### For Development
```bash
git clone https://github.com/wivanw/spectrue-engine.git
cd spectrue-engine
pip install -e ".[dev]"
```

## 💡 Usage

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
result = await engine.analyze_text(
    text="NASA discovered a new moon orbiting Earth.",
    lang="en"
)

print(f"Veracity: {result['verified_score']:.2f}")
print(f"Confidence: {result['confidence_score']:.2f}")
print(f"Analysis: {result['rationale']}")
```

## 🏗️ Architecture

```
spectrue_core/
├── engine.py              # Main entry point
├── config.py              # Configuration management
├── agents/                # LLM agents (fact checker, query generator)
├── analysis/              # Text analysis and parsing
├── verification/          # Fact verification logic
└── tools/                 # Search tools (Tavily, Google FC)
```

## 🔧 Configuration

Configure via `SpectrueConfig`:

```python
config = SpectrueConfig(
    openai_api_key="...",           # Required for analysis
    tavily_api_key="...",           # Required for search
    openai_model="gpt-4o",          # Default model
    min_confidence_threshold=0.7,   # Minimum confidence
    max_search_depth=3              # Search recursion depth
)
```

Or use environment variables with `SPECTRUE_` prefix:
```bash
export SPECTRUE_OPENAI_API_KEY="sk-..."
export SPECTRUE_TAVILY_API_KEY="tvly-..."
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests: `pytest`
5. Lint: `ruff check .`
6. Submit a Pull Request

## �� License

This project is licensed under the **GNU Affero General Public License v3 (AGPLv3)**.

This means:
- ✅ You can use it in your projects
- ✅ You can modify and distribute it
- ⚠️ If you run a modified version as a service, you **must** share your source code

See [LICENSE](LICENSE) for full details.

## 🛡️ Security

Found a security issue? Please email **wivanw@gmail.com** instead of opening a public issue.

See [SECURITY.md](SECURITY.md) for our security policy.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/wivanw/spectrue-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/wivanw/spectrue-engine/discussions)
- **Email**: wivanw@gmail.com

## 🙏 Acknowledgments

Built with support from:
- NGI Zero Commons Fund
- Open Source community

---

**Made with ❤️ for transparency in AI**
