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

- **Claim-Centric Orchestration**: Each claim gets metadata-driven verification routing
- **Progressive Widening Search**: Cost-aware phases with early exit when evidence is sufficient
- **Multi-Agent Architecture**: Orchestrates Oracle, Analyst, and Verifier agents
- **Hallucination Resistance**: Strict source verification with 'Aletheia-X' prompts
- **Smart Waterfall Search**: Optimized strategy (Oracle → Tier 1 → Deep Dive)
- **Content-Aware Localization**: Detects content language and uses native sources
- **RGBA Analysis**: Returns orthogonal scores for Danger, Veracity, Honesty, and Explainability
- **Fail-Soft Architecture**: Graceful degradation on component failures

## 🔄 Verification Pipeline

The core verification process follows this pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT (URL or Text)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. CLAIM EXTRACTION + METADATA                                 │
│     • LLM extracts atomic verifiable claims                     │
│     • Each claim gets ClaimMetadata:                            │
│       - verification_target: reality|attribution|existence|none │
│       - claim_role: core|support|context|meta                   │
│       - search_locale_plan: primary + fallback languages        │
│       - retrieval_policy: allowed evidence channels             │
│       - metadata_confidence: high|medium|low                    │
│     • "Search Strategist" approach: LLM reasons about           │
│       intent, authority, language, risks (Chain of Thought)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. ORCHESTRATOR → EXECUTION PLAN                               │
│     • ClaimOrchestrator builds ExecutionPlan per claim          │
│     • Phases based on metadata:                                 │
│       - Phase A: Primary locale, authoritative sources, k=3    │
│       - Phase B: +local media, advanced depth, k=5             │
│       - Phase C: Fallback locale (e.g., English), k=3          │
│       - Phase D: All channels, deep search, k=7                │
│       - Phase A-light: Fail-open for low-confidence, k=2       │
│     • verification_target=none → 0 phases (skip search)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. ORACLE CHECK (Hybrid Mode)                                  │
│     • Smart Validator: LLM compares claim vs fact-check         │
│     • JACKPOT (>0.9): Stop pipeline immediately                 │
│     • EVIDENCE (0.5-0.9): Add to evidence pack (Tier A)         │
│     • MISS (<0.5): Proceed to web search                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. PROGRESSIVE WIDENING (PhaseRunner)                          │
│     • Execute phases sequentially: A → B → C → D                │
│     • After each phase: check evidence sufficiency              │
│     • Sufficiency Rules:                                        │
│       Rule 1: 1 authoritative source with quote = STOP          │
│       Rule 2: 2 independent reputable sources = STOP            │
│       Rule 3: 1 origin source (for attribution) = STOP          │
│     • Early exit: Skip remaining phases when sufficient         │
│     • Parallel execution within each phase (semaphore-limited)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. STANCE CLUSTERING                                           │
│     • LLM maps search results to claims                         │
│     • Assigns stance: support | contradict | context            │
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
│  7. WEIGHTED RGBA SCORING                                       │
│     • Quote Highlighting: "📌 QUOTE" markers for key evidence   │
│     • Generates verdict per-claim with semantic scale           │
│     • Aggregates with role-based weighting:                     │
│       - CORE claims: weight=1.0                                 │
│       - CONTEXT claims (horoscopes, predictions): weight=0.0    │
│       - ATTRIBUTION claims: weight=0.7                          │
│     • Result: Context claims don't dilute factual scores        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT (Result)                           │
│  verified_score, danger_score, style_score, explainability,     │
│  rationale, claim_verdicts, sources, phase_trace                │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Claim-Centric Orchestration (M80)

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

## 📋 Requirements

- **Python**: 3.10–3.12 (3.10+ supported)
- **Dependencies**: See [pyproject.toml](pyproject.toml)

### Required API Keys

| Key | Purpose | Required |
|-----|---------|----------|
| `OPENAI_API_KEY` | LLM analysis (GPT-5) | Yes |
| `TAVILY_API_KEY` | Web search | Yes |
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

### With Claim Orchestration (M80)

```python
from spectrue_core.verification.orchestrator import ClaimOrchestrator
from spectrue_core.verification.phase_runner import PhaseRunner
from spectrue_core.verification.execution_plan import BudgetClass

# Build execution plan
orchestrator = ClaimOrchestrator()
plan = orchestrator.build_execution_plan(claims, BudgetClass.STANDARD)

# Run progressive widening
runner = PhaseRunner(search_manager, max_concurrent=3)
evidence = await runner.run_all_claims(claims, plan)

# Evidence is keyed by claim_id
for claim_id, sources in evidence.items():
    print(f"Claim {claim_id}: {len(sources)} sources found")
```

### Checking Evidence Sufficiency

```python
from spectrue_core.verification.sufficiency import evidence_sufficiency
from spectrue_core.schema.claim_metadata import VerificationTarget

result = evidence_sufficiency(
    claim_id="c1",
    sources=search_results,
    verification_target=VerificationTarget.REALITY
)

if result.status == "sufficient":
    print(f"✓ Stopped early: {result.rule_matched}")
else:
    print(f"Continue searching: {result.reason}")
```

## 🏗️ Architecture

```
spectrue_core/
├── engine.py              # Main entry point
├── config.py              # Configuration management
├── runtime_config.py      # Feature flags & tunables
│
├── agents/                # LLM agents
│   └── skills/            # Modular skills
│       ├── claims.py      # Claim extraction + metadata
│       ├── clustering.py  # Stance clustering
│       ├── scoring.py     # Evidence scoring
│       └── relevance.py   # Semantic gating
│
├── schema/                # Data types
│   ├── claim_metadata.py  # M80: ClaimMetadata, VerificationTarget
│   ├── claims.py          # ClaimUnit, Assertion
│   ├── verdict.py         # StructuredVerdict
│   └── serialization.py   # Canonical JSON-safe serialization helpers
│
├── verification/          # Verification pipeline
│   ├── pipeline.py        # Main orchestrator
│   ├── orchestrator.py    # M80: ClaimOrchestrator
│   ├── execution_plan.py  # M80: Phase, ExecutionPlan
│   ├── phase_runner.py    # M80: PhaseRunner
│   ├── sufficiency.py     # M80: Evidence sufficiency
│   ├── rgba_aggregation.py# M80: Weighted RGBA
│   ├── evidence.py        # Evidence pack builder
│   ├── evidence_pack.py   # Data structures
│   └── search_mgr.py      # Search orchestration
│
├── graph/                 # ClaimGraph (M72)
│   ├── claim_graph.py     # Build pipeline orchestration
│   ├── candidates.py      # B-stage: candidate generation
│   ├── ranking.py         # Ranking (PageRank)
│   ├── quality_gates.py   # Gate checks (kept_ratio bounds)
│   └── embedding_util.py  # Embedding client
│
├── utils/                 # Utilities
│   ├── trace.py           # Debug tracing (safe payloads)
│   └── trust_utils.py     # Source reputation
│
└── tools/                 # External APIs
    ├── search_tool.py     # Tavily API
    ├── google_fact_check.py  # Google Fact Check
    └── google_cse_search.py  # Google Custom Search
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
export FEATURE_CLAIM_ORCHESTRATION=true  # Enable M80 orchestration
export M80_MAX_CONCURRENT_SEARCHES=3     # Parallel search limit

# Trace Configuration
export TRACE_SAFE_PAYLOADS=true    # Sanitize logs (default: true)
export TRACE_MAX_HEAD_CHARS=120    # Truncation limit
```

### Programmatic Configuration

```python
config = SpectrueConfig(
    openai_api_key="...",           # Required for analysis
    tavily_api_key="...",           # Required for search
    openai_model="gpt-5",            # Model for analysis
    min_confidence_threshold=0.7,   # Minimum confidence
    max_search_depth=3              # Search recursion depth
)
```

## 🧪 Testing

```bash
# Run offline core suite (no network, no secrets)
export SPECTRUE_TEST_OFFLINE=1
pytest tests/unit tests/test_*.py \
  tests/integration/test_m80_orchestration.py \
  tests/integration/test_m81_calibration.py \
  tests/integration/test_verification_pipeline.py

# Run specific test suite
pytest tests/unit/test_orchestrator.py -v
pytest tests/unit/test_sufficiency.py -v
pytest tests/integration/test_m80_orchestration.py -v

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
