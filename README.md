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

## 🔄 Verification Pipeline

The core verification process follows this pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT (URL or Text)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. CLAIM EXTRACTION                                            │
│     • LLM extracts atomic verifiable claims                     │
│     • Each claim gets: normalized_text, topic_group,            │
│       check_worthiness, search_strategy                         │
│     • "Search Strategist" approach: LLM reasons about           │
│       intent, authority, language, risks (Chain of Thought)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. ORACLE CHECK (Hybrid Mode)                                  │
│     • Smart Validator: LLM compares claim vs fact-check         │
│     • JACKPOT (>0.9): Stop pipeline immediately                 │
│     • EVIDENCE (0.5-0.9): Add to evidence pack (Tier A)         │
│     • MISS (<0.5): Proceed to web search                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. QUERY SELECTION (Coverage Engine)                           │
│     • Topic-Aware Round-Robin: Ensures 100% topic coverage      │
│     • 3-Pass Selection:                                         │
│       1. Coverage: One CORE query per topic (Space, Economy...) │
│       2. Depth: Add NUMERIC/ATTRIBUTION if budget allows        │
│       3. Fill: Remaining slots                                  │
│     • Safety: 90% fuzzy dedup + LLM/Tavily gambling guards      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. SEARCH WATERFALL                                            │
│     • Tier 1: Trusted domains (Reuters, AP, gov sites)          │
│     • Tier 2: General search (if T1 insufficient)               │
│     • CSE Fallback: Google Custom Search (if Tavily empty)      │
│     • Extracts full text from top results                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. STANCE CLUSTERING                                           │
│     • LLM maps search results to claims                         │
│     • Assigns stance: support | contradict | irrelevant         │
│     • Calculates relevance score per source-claim pair          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. EVIDENCE PACK BUILDING                                      │
│     • Structures evidence for LLM scorer                        │
│     • Computes per-claim metrics:                               │
│       - independent_domains, primary_present, official_present  │
│       - stance_distribution, coverage                           │
│     • Sets confidence constraints based on evidence quality     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. SCORING (LLM)                                               │
│     • Generates verdict per-claim                               │
│     • Aggregates to verified_score (importance-weighted)        │
│     • Applies Hard Caps (Python, not LLM):                      │
│       - < 2 independent domains → max 0.65                      │
│       - Numeric claim no primary → max 0.60                     │
│     • Core claim refuted → global cap 0.25                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT (Result)                           │
│  verified_score, confidence_score, danger_score,                │
│  rationale, claim_verdicts, sources, caps_applied               │
└─────────────────────────────────────────────────────────────────┘
```

## 🧠 Design Philosophy

### LLM as Search Strategist

**При роботі з кодом пошукової системи треба опиратись не на евристику і конкретні приклади, а делегувати ці задачі LLM для покращення результатів.**

This means:
- ❌ **NO hardcoded `if/else`** for "if science → search English"
- ❌ **NO domain-specific heuristics** like keyword lists
- ✅ **LLM reasons** about intent, authority, language, risks
- ✅ **Chain of Thought prompts** force LLM to explain before generating
- ✅ **Python only for**: filtering, caps enforcement, API calls

**Why?** LLM generalizes to new domains (K-Pop → Korean, Cricket → Hindi) without code changes.

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
├── agents/                # LLM agents
│   └── skills/            # Modular skills
│       ├── claims.py      # Claim extraction + Search Strategist
│       ├── clustering.py  # Stance clustering
│       ├── scoring.py     # Evidence scoring + Hard Caps
│       └── query.py       # Query generation (legacy)
├── verification/          # Verification pipeline
│   ├── pipeline.py        # Main orchestrator
│   ├── evidence.py        # Evidence pack builder
│   ├── evidence_pack.py   # Data structures (TypedDicts)
│   └── search_mgr.py      # Search tool orchestration
└── tools/                 # Search tools
    ├── search_tool.py     # Tavily API
    ├── google_fact_check.py  # Google Fact Check API
    └── google_cse_search.py  # Google Custom Search
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

## 📜 License

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
