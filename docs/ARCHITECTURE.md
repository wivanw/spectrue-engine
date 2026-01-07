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

### Terminology & Contracts

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
│                 │ │                 │
└─────────────────┘ └─────────────────┘
```

### Step-Based Pipeline

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
| `DAGExecutionState` | Step status/timing for DAG runs | `pipeline/execution_state.py` |
| DAG metadata constants | Shared DAG keys/statuses | `pipeline/constants.py` |
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

**Execution Visibility**: DAG runs record ordered layers and step-level status/timing.
Summaries are emitted via trace events and attached to results as
`dag_execution_summary` and `dag_execution_state` for debugging.

### Bayesian Claim Posterior Model

Unified scoring model that replaces double-counting patterns in verdict calculation.

**Problem Solved**: Previously, evidence was counted twice:
1. `aggregate_claim_verdict()` computed score from evidence (0.85 for quotes)
2. `_compute_article_g_from_anchor()` applied formula with already-inflated score

**Solution**: Single log-odds formula in `scoring/claim_posterior.py`:

```
ℓ_post = ℓ_prior + α·ℓ_llm + β·ℓ_evidence
p_post = σ(ℓ_post)
```

Where:
- `ℓ_prior = logit(p_prior)` — from tier/domain quality
- `ℓ_llm = logit(p_llm)` — from LLM verdict_score (noisy observation)
- `ℓ_evidence = Σ w_j · s_j` — weighted sum of stance signals
- `α, β` — calibration parameters from `SearchPolicyProfile`

**Key Components:**

| Component | Purpose | Location |
|-----------|---------|----------|
| `compute_claim_posterior()` | Unified posterior calculation | `scoring/claim_posterior.py` |
| `EvidenceItem` | Single evidence signal | `scoring/claim_posterior.py` |
| `PosteriorParams` | Calibration parameters | `scoring/claim_posterior.py` |
| `posterior_alpha`, `posterior_beta` | Policy weights | `verification/search_policy.py` |

**Invariants:**
- Weak/neutral evidence cannot inflate score to tier-prior level
- Evidence counted exactly once
- All signals in log-odds space (additive updates)

### Deep Mode Efficiency

Optimized deep mode from N+1 pipeline runs to 2:

**Before (N+1 runs)**:
```
1. Initial pipeline (extract_claims_only=True) → N claims
2. For each claim: verify_fact() → N pipeline runs
Total: N+1 pipeline runs
```

**After (2 runs)**:
```
1. Initial pipeline (extract_claims_only=True) → N claims
2. Single verify_fact() with preloaded_claims → 1 pipeline run
Total: 2 pipeline runs
```

**Key Changes:**
- `preloaded_claims` parameter in `verifier.verify_fact()`
- `ExtractClaimsStep` skips extraction when claims pre-exist
- ~50% cost reduction, ~40% latency reduction

### Coverage Skeleton Extraction

Two-phase claim extraction for comprehensive coverage:

**Phase 1: Skeleton Extraction**
- Extract events, measurements, quotes, policies with `raw_span`
- Regex-based coverage analyzers (time mentions, number mentions, quote spans)
- Coverage validation with tolerance threshold

**Phase 2: Skeleton → Claims**
- Convert skeleton items to claim-compatible dicts
- Per-type converters with basic validation
- `skeleton_item_id` traceability for debugging

**Key Components:**

| Component | Purpose | Location |
|-----------|---------|----------|
| `SkeletonEvent` | Event with entities, verb, time/location | `agents/skills/coverage_skeleton.py` |
| `SkeletonMeasurement` | Metric with quantity mentions | `agents/skills/coverage_skeleton.py` |
| `SkeletonQuote` | Quote with speaker entities | `agents/skills/coverage_skeleton.py` |
| `SkeletonPolicy` | Policy/regulation with action | `agents/skills/coverage_skeleton.py` |
| `skeleton_to_claims()` | Converter function | `agents/skills/coverage_skeleton.py` |
| `validate_skeleton_coverage()` | Coverage validation | `agents/skills/coverage_skeleton.py` |

**Trace Events:**
- `claims.skeleton.created` — skeleton item counts
- `claims.coverage.warning` — low coverage detected
- `claims.skeleton.to_claims` — conversion stats

### Search Escalation Policy

Multi-pass Tavily search with evidence-driven escalation:

```
Pass A (cheap): basic depth, 3 results, Q1/Q2
    ↓ (if insufficient)
Pass B (expand): basic depth, 6 results, Q2/Q3
    ↓ (if insufficient)
Pass C (deep): advanced depth, 6 results, Q1/Q2
    ↓ (if insufficient)
Pass D (relax): basic depth, 6 results, no domain restriction
```

**Query Variants (Q1/Q2/Q3):**
- Q1 (anchor_tight): entities + seed terms + date anchor
- Q2 (anchor_medium): entities + seed terms (no date)
- Q3 (broad): fewer terms for wider recall

**Topic Selection:**
- `falsifiable_by` in {reputable_news, official_statement} → "news"
- `falsifiable_by` in {dataset, scientific_publication} + explicit date → "news"
- Default → "news" (better for domain allowlists)

**Key Components:**

| Component | Purpose | Location |
|-----------|---------|----------|
| `QueryVariant` | Deterministic query from claim fields | `verification/search/search_escalation.py` |
| `EscalationPass` | Pass config (depth, results, domain relaxation) | `verification/search/search_escalation.py` |
| `RetrievalOutcome` | Observable quality signals | `verification/search/search_escalation.py` |
| `build_query_variants()` | Build Q1/Q2/Q3 from claim | `verification/search/search_escalation.py` |
| `select_topic_from_claim()` | Topic selection from falsifiability | `verification/search/search_escalation.py` |

**Trace Events:**
- `search.topic.selected` — topic and reason codes
- `search.escalation` — pass executed with params and outcome
- `search.stop` — early stop reason
- `search.summary` — end-of-retrieval summary

### Claim Verifiability Contract

Deterministic validation enforcing verifiable claims:

**Required Fields:**
- `claim_text` — non-empty
- `subject_entities` — at least 1
- `retrieval_seed_terms` — at least 3
- `falsifiability.is_falsifiable` — must be true
- `time_anchor` — required for event/measurement (exempt: quote, policy, ranking)

**Validation Function:**
```python
ok, reason_codes = validate_core_claim(claim)
# reason_codes: ['not_falsifiable', 'missing_subject_entities', ...]
```

**Time Anchor Exempt Predicates:**
- `quote` — verifiable via source attribution
- `policy` — verifiable via public records
- `ranking` — verifiable via latest data
- `existence` — verifiable by finding the entity

**Key Components:**

| Component | Purpose | Location |
|-----------|---------|----------|
| `validate_core_claim()` | Deterministic validation | `agents/skills/claims.py` |
| `ExtractionStats` | Track extracted/dropped/emitted | `agents/skills/claims.py` |
| `TIME_ANCHOR_EXEMPT_PREDICATES` | Predicates not requiring time anchor | `agents/skills/claims.py` |

**Trace Events:**
- `claim.dropped` — claim failed validation with reason codes
- `claims.extraction_stats` — extraction/drop/emit counts

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

### ClaimGraph

The ClaimGraph module is intentionally split so the builder reads as a pipeline:
- `spectrue_core/graph/candidates.py`: B-stage candidate generation (embeddings + adjacency).
- `spectrue_core/graph/quality_gates.py`: gate checks (topic-aware kept_ratio).
- `spectrue_core/graph/ranking.py`: ranking (PageRank).
- `spectrue_core/graph/claim_graph.py`: orchestrates the steps and owns caches.

### ClaimOrchestrator

Builds `ExecutionPlan` for each claim based on:
- `verification_target`: What to verify
- `claim_role`: Role in the document structure
- `search_locale_plan`: Language strategy
- `budget_class`: Available search budget
- `retrieval_policy`: Allowed evidence channels + usage policy (support_ok vs lead_only)

**Location**: `spectrue_core/verification/orchestrator.py`

### PhaseRunner

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

Abstracts search APIs and manages Evidence Acquisition Ladder (EAL):
- Tavily (primary search + extract)
- Google Custom Search (fallback)
- Google Fact Check (oracle)

**Evidence Acquisition Ladder (EAL)**:
- `apply_evidence_acquisition_ladder()` — enriches sources with full content and quotes
- Uses Bayesian EVOI model for budget allocation
- Per-claim budget trackers for deep mode (prevents budget pollution between parallel claims)

**Budget Trackers**:
- `inline_budget_tracker` — for inline source verification
- `get_claim_budget_tracker(claim_id)` — returns per-claim tracker for deep mode

**Location**: `spectrue_core/verification/search/search_mgr.py`

**Canonical shapes**:
- Search returns `(context_text, sources)` (`SearchResponse`) — `spectrue_core/verification/types.py`
- Provider source normalization lives in `spectrue_core/verification/search/source_utils.py`

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

### Phase 3: Orchestration

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
│   │   ├── claims.py         # Claim extraction + validation
│   │   ├── claims_prompts.py # Claim extraction prompts
│   │   ├── coverage_skeleton.py  # Coverage skeleton extraction
│   │   ├── claim_metadata_parser.py  # Claim metadata parsing helpers
│   │   ├── clustering.py     # Source-claim mapping
│   │   ├── scoring.py        # RGBA scoring
│   │   ├── relevance.py      # Semantic gating
│   │   ├── edge_typing.py    # ClaimGraph edges
│   │   └── query.py          # Query generation
│   └── locales/              # Prompt templates
│
├── schema/
│   ├── claim_metadata.py     # ClaimMetadata types
│   ├── claims.py             # ClaimUnit, Assertion
│   ├── evidence.py           # EvidenceItem
│   ├── serialization.py      # Canonical schema serialization helpers
│   ├── verdict.py            # StructuredVerdict
│   └── verdict_contract.py   # Public verdict schema
│
├── pipeline/                  # Step-based architecture
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
│   ├── orchestrator.py       # ClaimOrchestrator
│   ├── execution_plan.py     # Phase, ExecutionPlan
│   ├── types.py              # Canonical types (SearchResponse, Source)
│   ├── source_utils.py       # Source normalization helpers
│   ├── phase_runner.py       # PhaseRunner
│   ├── sufficiency.py        # Evidence sufficiency
│   ├── rgba_aggregation.py   # Weighted aggregation
│   ├── evidence.py           # EvidencePack builder
│   ├── evidence_pack.py      # TypedDicts
│   ├── search_mgr.py         # Search orchestration
│   └── search/
│       ├── search_escalation.py   # Search escalation policy
│       └── search_policy_adapter.py  # Policy enforcement
│
├── graph/
│   ├── claim_graph.py        # ClaimGraph builder
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
| `TRACE_SAFE_PAYLOADS` | `false` | Sanitize trace logs (redact PII/secrets) |

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
