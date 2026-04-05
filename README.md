# Spectrue Engine

<p align="center">
  <strong>Open Source AI Fact-Checking Core</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg" alt="License: AGPL v3"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/wivanw/spectrue-engine/actions"><img src="https://img.shields.io/github/actions/workflow/status/wivanw/spectrue-engine/ci.yml?branch=main&label=CI" alt="CI Status"></a>
  <a href="https://codecov.io/gh/wivanw/spectrue-engine"><img src="https://img.shields.io/codecov/c/github/wivanw/spectrue-engine" alt="Coverage"></a>
  <a href="https://github.com/wivanw/spectrue-engine/releases"><img src="https://img.shields.io/github/v/release/wivanw/spectrue-engine?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  The transparent, hallucination-resistant analysis engine behind Spectrue.<br>
  Multi-agent fact-checking • Web-based verification • Deep analysis
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="docs/API.md">API Docs</a> •
  <a href="docs/ARCHITECTURE.md">Architecture</a> •
  <a href="#-contributing">Contributing</a>
</p>

---


## ✨ Features

- **Layered Architecture (DDD)**: Strict separation of Domain, Use Cases, Adapters, and Pipeline layers.
- **Step-Based DAG Pipeline**: Asynchronous Directed Acyclic Graph orchestration with thin, reusable steps.
- **Claim-Centric Orchestration**: Each claim gets metadata-driven verification routing.
- **Progressive Widening Search**: Cost-aware phases with early exit when evidence is sufficient.
- **Multi-Agent Architecture**: Orchestrates Oracle, Analyst, and Verifier agents.
- **Hallucination Resistance**: Strict source verification with 'Aletheia-X' prompts.
- **RGBA Analysis**: Returns orthogonal scores for Danger, Veracity, Honesty, and Explainability.
- **Fail-Soft Architecture**: Graceful degradation on component failures.


## 📚 Documentation

- **Core Architecture**: `docs/ARCHITECTURE.md` (Layers, Boundaries, Design Principles)
- **Algorithms & Contracts**: `docs/ALGORITHMS.md` (Bayesian scoring, EAL, Graph propagation)
- **Deep Mode**: `docs/DEEP_MODE.md` (Per-claim judging, deep v2 clustered retrieval)
- **Resource Accounting**: `docs/RESOURCE_ACCOUNTING.md` (Cost metering, credits)
- **Trace Debugging**: `docs/TRACE_GUIDE.md` (How to read execution traces)

## 🔄 Verification Pipeline (DAG)

The engine executes verification as a **Directed Acyclic Graph (DAG)** of thin, focused steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT (URL or Text)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. PREPARATION & CLEANING                                      │
│     • MeteringSetup: Initialize cost tracking                    │
│     • PrepareInput: Normalize text and locale                    │
│     • ArticleCleaner: Markdown-aware cleaning                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CLAIM EXTRACTION & GRAPH                                    │
│     • ExtractClaims: LLM decomposes text into atomic claims      │
│     • ClaimGraph: Build semantic dependency graph                │
│     • ClaimClusters: Group similar claims (Deep v2)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. RETRIEVAL ORCHESTRATION                                     │
│     • TargetSelection: Bayesian EVOI selection (which to verify) │
│     • BuildQueries: Multi-phase query generation                 │
│     • OracleFlow: Smart fact-check validation                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. WEB SEARCH & RERANKING                                      │
│     • WebSearch: Parallel retrieval (Tavily, Google)             │
│     • Rerank: Relevance filtering and duplicate removal           │
│     • FetchChunks: Deep content acquisition                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. EVIDENCE PROCESSING                                         │
│     • EvidenceCollect: Payload assembly and chunking             │
│     • EvidenceGating: Semantic relevance filtering               │
│     • StanceAnnotate: Support / Refute / Context classification  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. ANALYSIS & JUDGING                                          │
│     • Standard: Global batch scoring + RGBA aggregation          │
│     • Deep: Per-claim independent judging (ClaimFrames)          │
│     • SummarizeEvidence: Stance-based categorization             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. ASSEMBLY & COSTING                                          │
│     • ResultAssembly: Final response serialization               │
│     • CostSummary: Accurate fractional SC accounting             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT (Result)                           │
└─────────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture (DDD)

The codebase is organized into four strictly decoupled layers:

### 📦 Layer 1: Domain (`spectrue_core/domain/`)
Pure business logic and models. Zero dependencies on external services or orchestration.
- **`claims/`**: Models, extraction logic, and graph algorithms.
- **`evidence/`**: Deduplication, clustering, and corroboration rules.
- **`verification/`**: Stance evaluation and Bayesian verdict updates.

### 📦 Layer 2: Use Cases (`spectrue_core/use_cases/`)
Coordinates domain logic and adapters to perform application-level operations.
- **`claims/`**, **`evidence/`**, **`verification/`**: Flow orchestration.

### 📦 Layer 3: Adapters (`spectrue_core/adapters/`)
Interface boundaries for external systems.
- **`llm/`**: Skill-based LLM adapters (Claims, Queries, Scoring).
- **`retrieval/`**: Search provider integrations (Tavily, Google).
- **`graph/`**: NetworkX and embedding service adapters.

### 📦 Layer 4: Pipeline (`spectrue_core/pipeline/`)
The execution engine for the verification DAG.
- **`steps/`**: "Thin" orchestration units (<= 50 lines) that invoke use cases.
- **`dag.py`**: Topological sort and parallel execution logic.
- **`factory.py`**: Mode-to-steps mapping (General vs Deep).

---

## 🎯 Claim-Centric Orchestration

The engine uses metadata-driven routing to optimize verification:

### ClaimMetadata

Each claim is enriched with metadata at extraction time:

```python
ClaimMetadata(
    verification_target="reality",  # What to verify
    claim_role="core",              # Role in document
    check_worthiness=0.9,           # Priority (0-1)
    search_locale_plan=SearchLocalePlan(
        primary="en",
        fallback=["uk"]
    ),
    retrieval_policy=RetrievalPolicy(
        channels_allowed=["authoritative", "reputable_news"]
    ),
    metadata_confidence="high"
)
```

### Verification Targets

| Target | Description | Example |
|--------|-------------|---------|
| `reality` | Verify factual accuracy | "Biden won 2024" |
| `attribution` | Verify who said what | "Elon Musk said..." |
| `existence` | Verify source/doc exists | "According to the report..." |
| `none` | Not verifiable (skip) | Horoscopes, predictions |

### Evidence Sufficiency

The engine stops searching when one of these rules is satisfied:

| Rule | Condition | Example |
|------|-----------|---------|
| **Rule 1** | 1 authoritative source (gov/edu) with quote | CDC confirms vaccine safety |
| **Rule 2** | 2 independent reputable sources with quotes | Reuters + AP both report |
| **Rule 3** | 1 origin source (for attribution claims) | Original tweet found |

## 🧠 Design Philosophy

### LLM as Search Strategist

**When working with search system code, rely on LLM reasoning rather than heuristics or hardcoded examples for better results.**

This means:
- ❌ **NO hardcoded `if/else`** for "if science → search English"
- ❌ **NO domain-specific heuristics** like keyword lists
- ✅ **LLM reasons** about intent, authority, language, risks
- ✅ **Chain of Thought prompts** force LLM to explain before generating
- ✅ **Python only for**: filtering, caps enforcement, API calls

**Why?** LLM generalizes to new domains (K-Pop → Korean, Cricket → Hindi) without code changes.

### Fail-Soft Architecture

The engine is designed to gracefully degrade:
- **Low confidence metadata**: Inject Phase A-light (minimal search)
- **Search failure**: Continue to next phase, don't crash
- **LLM failure**: Return partial results with reduced confidence

### Resource Accounting

The engine provides cost-aware execution with transparent measurement:
- **Deterministic accounting**: All resource consumption tracked with `Decimal` precision
- **Continuous measurement**: No intermediate rounding during a verification run
- **Transparent finalization**: Callers receive exact fractional values and may apply settlement rules

See [docs/RESOURCE_ACCOUNTING.md](docs/RESOURCE_ACCOUNTING.md) for full semantics.

## 📋 Requirements

- **Python**: 3.10–3.12 (3.10+ supported)
- **Dependencies**: See [pyproject.toml](pyproject.toml)

### Required API Keys

| Key | Purpose | Required |
|-----|---------|----------|
| `OPENAI_API_KEY` | LLM analysis (GPT-5) | Yes |
| `TAVILY_API_KEY` | Web search | Yes |
| `DEEPSEEK_API_KEY` | Deep reasoning (optional) | Optional |
| `GOOGLE_FACT_CHECK_KEY` | Oracle fact-check | Optional |

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

### Basic Usage

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

### With Claim Orchestration

```python
from spectrue_core.use_cases.verification.verdict import VerdictUseCase
from spectrue_core.pipeline.factory import PipelineFactory
from spectrue_core.pipeline.mode import PipelineMode

# Build execution plan and run pipeline
factory = PipelineFactory()
pipeline = factory.build(PipelineMode.GENERAL_MODE)

# Context handles state across steps
ctx = await pipeline.execute(text="...", lang="en")

print(f"Result: {ctx.verdict}")
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."

# Optional
export GOOGLE_FACT_CHECK_KEY="..."  # For Oracle
export SPECTRUE_ENGINE_DEBUG=true   # Enable debug logging

# Feature Flags
export SPECTRUE_MAX_CONCURRENT_SEARCHES=3    # Parallel search limit

# Trace Configuration
export TRACE_SAFE_PAYLOADS=false   # Sanitize logs (default: false)
export TRACE_MAX_HEAD_CHARS=120    # Truncation limit
```

## 🧪 Testing

```bash
# Run core suite
pytest tests/unit tests/test_*.py \
  tests/integration/test_orchestration.py \
  tests/integration/test_calibration.py \
  tests/integration/test_verification_pipeline.py

# Run specific test suite
pytest tests/unit/test_orchestrator.py -v
pytest tests/unit/test_sufficiency.py -v
pytest tests/integration/test_orchestration.py -v

# With coverage
pytest --cov=spectrue_core
```

**Current Test Coverage**: 59 orchestration tests + existing test suite

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests: `pytest`
5. Lint: `ruff check .`
6. Submit a Pull Request

## 🧰 Open Source Maintainer Checklist

Use this checklist to keep the engine “open-source ready” (reproducible, reviewable, and stable for external users).

### Releases
- Update `CHANGELOG.md` (Keep a Changelog; one entry per release).
- Bump version in `pyproject.toml` following SemVer (breaking changes require a major bump).
- Tag the release and ensure CI is green on the tag.

### Tests (No-Network Core Suite)
- Keep a **core test suite** that runs without network access or secrets (unit + key integration tests).
- Any test that requires network must be explicitly isolated/marked and not part of the core suite.
- Add regression tests for bug fixes, especially for pipeline/search “shape” changes.

### Documentation
- Keep `docs/ARCHITECTURE.md` consistent with the current module structure and terminology/contracts (Document, Claim, Claim metadata, ClaimRole, VerificationTarget, SearchLocalePlan, RetrievalPolicy, Evidence, Sufficiency).
- Update `docs/API.md` when public-facing data contracts change.
- Prefer additive/backward-compatible schema changes; document migrations when unavoidable.

### Compatibility & Contracts
- Do not break public entrypoints/imports; use thin wrappers + re-exports when refactoring.
- Keep canonical shapes stable (e.g. search returns `(context_text, sources)`; normalize provider fields like `link→url`, `snippet→content`).

### Security & Licensing
- Never commit secrets or trace artifacts with sensitive content.
- Ensure new files follow the repository’s license header pattern and do not introduce incompatible code/licenses.

## Project Principles

Spectrue is designed as non-authoritative analytical infrastructure.
The system explicitly surfaces uncertainty, missing context, and conflicting evidence instead of collapsing them into verdicts or confidence scores.

Key documents:
- [Independence & Free Access Philosophy](docs/principles/INDEPENDENCE_AND_FREE_ACCESS.md)
- [Governance & Non-Influence Guarantees](docs/principles/GOVERNANCE_AND_NON_INFLUENCE.md)
- [What Spectrue Is Not](docs/principles/WHAT_SPECTRUE_IS_NOT.md)

> **Note**: The public website and UI may lag behind the current engine capabilities.
> Claim-level analysis and full traceability are implemented in the engine and documented here.

## 📜 License

This project is licensed under the **GNU Affero General Public License v3 (AGPLv3)**.

This means:
- ✅ You can use it in your projects
- ✅ You can modify and distribute it
- ⚠️ If you run a modified version as a service, you **must** share your source code

See [LICENSE](LICENSE) for full details.

## 🛡️ Security

Found a security issue? Please email **admin@spectrue.net** instead of opening a public issue.

See [SECURITY.md](SECURITY.md) for our security policy.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/wivanw/spectrue-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/wivanw/spectrue-engine/discussions)
- **Email**: admin@spectrue.net

## 🙏 Acknowledgments

Built with support from:
- NGI Zero Commons Fund
- Open Source community

---

**Made with ❤️ for transparency in AI**