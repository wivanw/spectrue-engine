# Architecture Overview

This document describes the high-level architecture of Spectrue Engine.

## Table of Contents

- [Design Principles](#design-principles)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Module Structure](#module-structure)
- [Extension Points](#extension-points)

---

## Design Principles

### 1. LLM-First Search Strategy

Instead of hardcoded heuristics, the engine delegates search decisions to LLMs:

- **No domain-specific rules**: No "if science → search English" logic
- **Chain of Thought**: LLM explains reasoning before generating queries
- **Generalization**: New domains work without code changes

### 2. Claim-Centric Processing

Each claim is treated as an independent verification unit:

- **Metadata-driven routing**: `ClaimMetadata` determines search strategy
- **Independent phases**: Each claim has its own execution plan
- **Early exit**: Stop searching when evidence is sufficient

### Terminology & Contracts (M80)

Use these terms consistently in code, traces, and documentation:

- **Document**: input text/article being analyzed
- **Claim**: atomic statement to verify
- **Claim metadata**: non-verdict fields that control verification
- **ClaimRole**: `core` / `support` / `context` / `meta` / `attribution` / `aggregated` / `subclaim`
- **VerificationTarget**: `reality` / `attribution` / `existence` / `none`
- **SearchLocalePlan**: `{ primary, fallback[] }` (search locale; UI locale is explanations only)
- **RetrievalPolicy**:
  - `channels_allowed`: `authoritative` / `reputable_news` / `local_media` / `social` / `low_reliability_web`
  - `use_policy_by_channel`: `support_ok` or `lead_only` (social/low_reliability_web are lead-only by default)
- **Evidence**: quote/passage from a source linked to a claim
- **Verdict/Label**: Supported / Refuted / NEI (or product equivalents derived from scores)
- **Sufficiency**: “evidence is sufficient → stop early”

### 3. Fail-Soft Architecture

The engine gracefully degrades on failures:

- **Search failures**: Continue to next phase, don't crash
- **LLM failures**: Return partial results with reduced confidence
- **Low confidence**: Inject minimal search phase (fail-open)

### 4. Weighted Aggregation

Not all claims are equal:

- **Core claims**: Full weight in RGBA score (1.0)
- **Context claims**: Zero weight (horoscopes, predictions)
- **Importance weighting**: Higher check_worthiness = more impact

### 5. Cost-Aware Resource Accounting

The engine measures all resource consumption during verification:

- **Unified units**: Search and LLM operations normalized to SC (Spectrue Credit)
- **Deterministic measurement**: Usage tracked with `Decimal` precision
- **Continuous accounting**: No intermediate rounding during a run
- **Transparent finalization**: Exact fractional values available to the caller

> **Separation of concerns:** The engine measures resources; economic policy (pricing, billing) lives outside the engine.

See [Resource Accounting](./RESOURCE_ACCOUNTING.md) for detailed semantics.

---

## Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        SpectrueEngine                           │
│  High-level facade for analysis                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ValidationPipeline                         │
│  Orchestrates the verification flow                             │
└─────────────────────────────────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  ClaimExtractor │ │  SearchManager  │ │  ScoringSkill   │
│  (LLM)          │ │  (Tavily/CSE)   │ │  (LLM)          │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │
          ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│ ClaimOrchestrator│ │   PhaseRunner   │
│ (M80)           │ │   (M80)         │
└─────────────────┘ └─────────────────┘
```

### Step-Based Pipeline (M114-M115)

A new composable pipeline architecture that decomposes `ValidationPipeline.execute()` into discrete Steps with DAG execution:

```
┌─────────────────────────────────────────────────────────────────┐
│                       DAGPipeline                                │
│  Executes StepNodes with dependency resolution + parallel exec  │
└─────────────────────────────────────────────────────────────────┘
                              │
  ┌───────────────┬───────────┼───────────┬──────────────┐
  ▼               ▼           ▼           ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Metering  │ │Prepare   │ │Extract   │ │Search    │ │Evidence  │
│SetupStep │ │InputStep │ │ClaimsStep│ │FlowStep  │ │FlowStep  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Key Components:**

| Component | Purpose | Location |
|-----------|---------|----------|
| `PipelineMode` | Mode invariants (normal/deep) | `pipeline/mode.py` |
| `Step` Protocol | Composable unit of work | `pipeline/core.py` |
| `PipelineContext` | Immutable threading context | `pipeline/core.py` |
| `DAGPipeline` | Dependency-aware executor | `pipeline/dag.py` |
| `StepNode` | Step with depends_on/optional | `pipeline/dag.py` |
| `PipelineFactory` | Builds pipelines for modes | `pipeline/factory.py` |

**Invariant Steps**: Gate checks that fail fast on invalid input
- `AssertSingleClaimStep` (normal mode)
- `AssertSingleLanguageStep` (normal mode)
- `AssertNonEmptyClaimsStep` (all modes)

**Decomposed Steps**: Native implementations replacing legacy
- `MeteringSetupStep`, `PrepareInputStep`, `ExtractClaimsStep`
- `ClaimGraphStep`, `TargetSelectionStep`, `SearchFlowStep`
- `EvidenceFlowStep`, `OracleFlowStep`, `ResultAssemblyStep`

**Migration**: Enable via `use_step_pipeline: true` feature flag.

### SpectrueEngine

Entry point for external consumers. Provides simple `analyze_text()` API.

**Location**: `spectrue_core/engine.py`

### ValidationPipeline

Central orchestrator that coordinates:
1. Content resolution (URL → text)
2. Claim extraction
3. Search execution
4. Evidence building
5. Scoring

**Location**: `spectrue_core/verification/pipeline.py` (thin orchestration) +
focused submodules:
- `spectrue_core/verification/pipeline_input.py` (URL/input helpers)
- `spectrue_core/verification/pipeline_queries.py` (query selection helpers)
- `spectrue_core/verification/pipeline_oracle.py`
- `spectrue_core/verification/pipeline_claim_graph.py`
- `spectrue_core/verification/pipeline_search.py`
- `spectrue_core/verification/pipeline_evidence.py`

### ClaimGraph (M72) — split modules (M88)

The ClaimGraph module is intentionally split so the builder reads as a pipeline:
- `spectrue_core/graph/candidates.py`: B-stage candidate generation (embeddings + adjacency).
- `spectrue_core/graph/quality_gates.py`: gate checks (topic-aware kept_ratio).
- `spectrue_core/graph/ranking.py`: ranking (PageRank).
- `spectrue_core/graph/claim_graph.py`: orchestrates the steps and owns caches.

### ClaimOrchestrator (M80)

Builds `ExecutionPlan` for each claim based on:
- `verification_target`: What to verify
- `claim_role`: Role in the document structure
- `search_locale_plan`: Language strategy
- `budget_class`: Available search budget
- `retrieval_policy`: Allowed evidence channels + usage policy (support_ok vs lead_only)

**Location**: `spectrue_core/verification/orchestrator.py`

### PhaseRunner (M80)

Executes progressive widening:
1. Run Phase A for all claims (parallel)
2. Check sufficiency for each claim
3. Continue to Phase B, C, D for insufficient claims
4. Early exit when sufficient

Sufficiency is based on **evidence chunks** (quote/snippet/content) and tier rules; it must not be driven by
search `results_count`/`avg_relevance` heuristics. Lead-only channels (social/low_reliability_web) do not satisfy
sufficiency rules.

**Location**: `spectrue_core/verification/phase_runner.py`

### SearchManager

Abstracts search APIs:
- Tavily (primary)
- Google Custom Search (fallback)
- Google Fact Check (oracle)

**Location**: `spectrue_core/verification/search_mgr.py`

**Canonical shapes**:
- Search returns `(context_text, sources)` (`SearchResponse`) — `spectrue_core/verification/types.py`
- Provider source normalization lives in `spectrue_core/verification/source_utils.py`

### Skills (LLM Agents)

Modular LLM-powered components:

| Skill | Purpose |
|-------|---------|
| `ClaimExtractionSkill` | Extract claims with metadata |
| `ClusteringSkill` | Map sources to claims |
| `ScoringSkill` | Generate RGBA verdicts |
| `RelevanceSkill` | Semantic gating |
| `EdgeTypingSkill` | ClaimGraph relations |

**Location**: `spectrue_core/agents/skills/`

---

## Data Flow

### Phase 1: Input Processing

```
URL/Text → ContentResolver → Clean Text
                                ↓
                         LanguageDetector
                                ↓
                          Cleaned Text + Lang
```

### Phase 2: Claim Extraction

```
Clean Text → ClaimExtractionSkill (LLM)
                    ↓
             List[Claim] with:
             - text, normalized_text
             - topic_group, importance
             - ClaimMetadata
```

### Phase 3: Orchestration (M80)

```
Claims → ClaimOrchestrator
              ↓
         ExecutionPlan:
         - Claim "c1": [Phase A, Phase B]
         - Claim "c2": [] (verification_target=none)
         - Claim "c3": [Phase A, Phase B, Phase C]
```

### Phase 4: Search Execution

```
ExecutionPlan → PhaseRunner
                    ↓
              For each phase:
              1. Execute search (parallel within phase)
              2. Check sufficiency
              3. Early exit if sufficient
                    ↓
              Evidence: {claim_id: [sources]}
```

### Phase 5: Scoring

```
Claims + Evidence → ScoringSkill (LLM)
                         ↓
                  StructuredVerdict:
                  - claim_verdicts[]
                  - verified_score, danger_score
                  - rationale
```

### Phase 6: Aggregation

```
claim_verdicts → aggregate_weighted()
                      ↓
                AggregatedRGBA:
                - verified (weighted by role)
                - danger, style, explainability
```

---

## Module Structure

```
spectrue_core/
├── engine.py              # Public entry point
├── config.py              # Configuration classes
├── runtime_config.py      # Feature flags, tunables
│
├── agents/
│   ├── fact_checker_agent.py
│   ├── skills/
│   │   ├── base_skill.py     # Skill base class
│   │   ├── claims.py         # Claim extraction
│   │   ├── claim_metadata_parser.py  # M80: Claim metadata parsing helpers
│   │   ├── clustering.py     # Source-claim mapping
│   │   ├── scoring.py        # RGBA scoring
│   │   ├── relevance.py      # Semantic gating
│   │   ├── edge_typing.py    # ClaimGraph edges
│   │   └── query.py          # Query generation
│   └── locales/              # Prompt templates
│
├── schema/
│   ├── claim_metadata.py     # M80: ClaimMetadata types
│   ├── claims.py             # ClaimUnit, Assertion
│   ├── evidence.py           # EvidenceItem
│   ├── serialization.py      # Canonical schema serialization helpers
│   ├── verdict.py            # StructuredVerdict
│   └── verdict_contract.py   # Public verdict schema
│
├── pipeline/                  # M114-M115: Step-based architecture
│   ├── __init__.py           # Module exports
│   ├── mode.py               # PipelineMode, NORMAL_MODE, DEEP_MODE
│   ├── core.py               # Step Protocol, Pipeline, PipelineContext
│   ├── dag.py                # DAGPipeline, StepNode, topological sort
│   ├── errors.py             # PipelineViolation, PipelineExecutionError
│   ├── factory.py            # PipelineFactory (mode → steps mapping)
│   ├── executor.py           # execute_pipeline, validate_claims_for_mode
│   └── steps/
│       ├── __init__.py       # Step exports
│       ├── invariants.py     # AssertSingleClaimStep, etc.
│       ├── legacy.py         # LegacyPhaseRunnerStep wrappers
│       └── decomposed.py     # Native Steps (MeteringSetup, etc.)
│
├── verification/
│   ├── pipeline.py           # Main orchestrator
│   ├── pipeline_input.py     # URL/input helpers
│   ├── pipeline_queries.py   # Query selection helpers
│   ├── pipeline_oracle.py    # Oracle flow
│   ├── pipeline_claim_graph.py # ClaimGraph gating + enrichment
│   ├── pipeline_search.py    # Search orchestration
│   ├── pipeline_evidence.py  # Evidence pack assembly + scoring glue
│   ├── orchestrator.py       # M80: ClaimOrchestrator
│   ├── execution_plan.py     # M80: Phase, ExecutionPlan
│   ├── types.py              # Canonical types (SearchResponse, Source)
│   ├── source_utils.py       # Source normalization helpers
│   ├── phase_runner.py       # M80: PhaseRunner
│   ├── sufficiency.py        # M80: Evidence sufficiency
│   ├── rgba_aggregation.py   # M80: Weighted aggregation
│   ├── evidence.py           # EvidencePack builder
│   ├── evidence_pack.py      # TypedDicts
│   └── search_mgr.py         # Search orchestration
│
├── graph/
│   ├── claim_graph.py        # M72: ClaimGraph builder
│   ├── quality_gates.py      # Graph quality gates (extracted helpers)
│   ├── embedding_util.py     # Embedding client
│   └── types.py              # Graph data types
│
├── tools/
│   ├── web_search_tool.py    # High-level web search facade (cache + enrichment)
│   ├── tavily_client.py      # Tavily HTTP client + payload shaping
│   ├── search_result_normalizer.py  # Provider result normalization
│   ├── cache_utils.py        # Cache helpers (diskcache)
│   ├── search_tool.py        # Back-compat wrapper
│   ├── url_utils.py          # Shared URL helpers
│   ├── google_fact_check.py  # Oracle
│   └── google_cse_search.py  # CSE fallback
│
└── utils/
    ├── trace.py              # Debug tracing
    ├── trust_utils.py        # Domain reputation
    └── text_chunking.py      # Text processing
```

---

## Extension Points

### Adding a New Skill

1. Create `spectrue_core/agents/skills/my_skill.py`:

```python
from spectrue_core.agents.skills.base_skill import BaseSkill

class MySkill(BaseSkill):
    async def execute(self, input_data):
        prompt = self._build_prompt(input_data)
        response = await self.llm.call(prompt)
        return self._parse_response(response)
```

2. Add prompt templates to `locales/`
3. Register in pipeline or agent

### Adding a New Search Provider

1. Create `spectrue_core/tools/my_search.py`:

```python
class MySearchTool:
    async def search(self, query: str, **kwargs) -> list[dict]:
        # Call external API
        return results
```

2. Register in `SearchManager`

### Adding a New Sufficiency Rule

1. Edit `spectrue_core/verification/sufficiency.py`:

```python
# Add rule check
if my_condition(sources):
    return SufficiencyResult(
        status=SufficiencyStatus.SUFFICIENT,
        reason="My rule satisfied",
        rule_matched="Rule4"
    )
```

### Custom Aggregation

1. Create custom aggregation function:

```python
from spectrue_core.verification.rgba_aggregation import ClaimScore

def my_aggregate(scores: list[ClaimScore]) -> dict:
    # Custom weighting logic
    return {"verified": weighted_avg, ...}
```

---

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `FEATURE_CLAIM_ORCHESTRATION` | `false` | Enable M80 orchestration |
| `CLAIM_GRAPH_ENABLED` | `true` | Enable ClaimGraph (M72) |
| `TRACE_SAFE_PAYLOADS` | `true` | Sanitize trace logs |

Configure via environment or `EngineRuntimeConfig`:

```python
from spectrue_core.runtime_config import EngineRuntimeConfig

config = EngineRuntimeConfig.load_from_env()
if config.features.claim_orchestration:
    # Use PhaseRunner
```

---

## Performance Considerations

### Parallelism

- Claims within same phase execute in parallel
- Semaphore limits concurrent searches (default: 3)
- Phases execute sequentially (waterfall)

### Early Exit

- Check sufficiency after each phase
- 1 authoritative source → stop immediately
- Saves API calls and latency

### Caching

- Embedding cache for ClaimGraph
- Search response cache (configurable TTL)
- LLM response cache (same input = same output)

---

## Testing Strategy

### Unit Tests

- Test individual components in isolation
- Mock LLM responses
- Mock search APIs

### Integration Tests

- Test component interactions
- Use real (mocked) data flows
- Verify trace events

### End-to-End Tests

- Full pipeline execution
- Real API calls (in CI with credentials)
- Performance benchmarks
