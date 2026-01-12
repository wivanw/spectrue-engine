# Spectrue Engine API Reference

This document describes the public API of Spectrue Engine.

## Table of Contents

- [Engine](#engine)
- [Configuration](#configuration)
- [Claim Metadata](#claim-metadata)
- [Orchestrator](#orchestrator)
- [Phase Runner](#phase-runner)
- [Sufficiency](#sufficiency)
- [RGBA Aggregation](#rgba-aggregation)

---

## Engine

The main entry point for Spectrue Engine.

### `SpectrueEngine`

```python
from spectrue_core.engine import SpectrueEngine
from spectrue_core.config import SpectrueConfig

config = SpectrueConfig(
    openai_api_key="sk-...",
    tavily_api_key="tvly-..."
)

engine = SpectrueEngine(config)
```

### Methods

#### `analyze_text(text, lang, mode, progress_callback)`

Analyze text for factual accuracy.

**Parameters:**
- `text` (str): The text to analyze
- `lang` (str): Language code (e.g., "en", "uk", "de")
- `mode` (str): Analysis mode - "lite" or "deep"
- `progress_callback` (Callable, optional): Progress callback function

**Returns:** `dict` with:
- `verified_score` (float): Factual accuracy score (0-1)
- `danger_score` (float): Potential harm score (0-1)
- `style_score` (float): Presentation quality score (0-1)
- `explainability_score` (float): How well sources explain the verdict (0-1)
- `rationale` (str): Human-readable explanation
- `claim_verdicts` (list): Per-claim verdicts
- `sources` (list): Evidence sources used

**Example:**
```python
result = await engine.analyze_text(
    text="NASA discovered a new moon orbiting Earth.",
    lang="en",
    mode="deep"
)
```

---

## Configuration

### `SpectrueConfig`

```python
from spectrue_core.config import SpectrueConfig

config = SpectrueConfig(
    openai_api_key="sk-...",           # Required
    tavily_api_key="tvly-...",         # Required
    google_fact_check_key="...",       # Optional
    openai_model="gpt-5",               # Model for analysis
    min_confidence_threshold=0.7,      # Minimum confidence
    max_search_depth=3                 # Search iteration limit
)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| **Core API Keys** | | |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `TAVILY_API_KEY` | Tavily search API key | Required |
| `GOOGLE_FACT_CHECK_KEY` | Google Fact Check API key | Optional |
| `DEEPSEEK_API_KEY` | DeepSeek API key (for heavy reasoning) | Optional |
| **Engine Control** | | |
| `SPECTRUE_MAX_CONCURRENT_SEARCHES` | Max parallel searches (PhaseRunner) | `3` |
| `SPECTRUE_LOCALE_PRIMARY` | Default analysis language | `en` |
| `SPECTRUE_LOCALE_FALLBACKS` | Secondary languages (CSV) | `uk` |
| `SPECTRUE_ENGINE_DEBUG` | Enable debug logging | `false` |
| **LLM & Models** | | |
| `OPENAI_CONCURRENCY` | LLM client max concurrency | `6` |
| `OPENAI_TIMEOUT` | Main LLM timeout (seconds) | `60.0` |
| `DEEPSEEK_BASE_URL` | DeepSeek API Base URL | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL_NAMES` | CSV of models routing to DeepSeek | `(empty)` |
| `MODEL_CLAIM_EXTRACTION` | Override model for extraction | `gpt-4o` |
| `MODEL_INLINE_SOURCE_VERIFICATION` | Override model for inline checks | `gpt-5-nano` |
| `MODEL_CLUSTERING_STANCE` | Override model for clustering | `gpt-4o-mini` |
| **Search Configuration** | | |
| `TAVILY_CONCURRENCY` | Max Tavily API concurrency | `5` |
| `SPECTRUE_TAVILY_EXCLUDE_DOMAINS` | Domains to ignore in search (CSV) | `(empty)` |
| `SPECTRUE_TAVILY_INCLUDE_RAW_CONTENT` | Fetch full pages (`true`, `false`, `auto`) | `auto` |
| `SPECTRUE_GOOGLE_CSE_COST` | Cost per Google search (credits) | `0` |
| **Claim Graph (Advanced)** | | |
| `CLAIM_GRAPH_TOP_K` | Number of claims to select | `12` |
| `CLAIM_GRAPH_K_SIM` | Candidate generation neighbors | `10` |
| `CLAIM_GRAPH_STRUCTURAL_ENABLED` | Enable structural boosting | `true` |
| `CLAIM_GRAPH_TENSION_ENABLED` | Enable tension detection | `true` |
| **Content Budget** | | |
| `CONTENT_BUDGET_MAX_DEFAULT_CHARS` | Max chars to process | `120000` |
| `CONTENT_BUDGET_BLOCK_MIN_CHARS` | Min chars for text block | `30` |
| **Feature Flags** | | |
| `FEATURE_RETRIEVAL_STEPS` | Enable fine-grained retrieval steps | `false` |
| `FEATURE_COVERAGE_CHUNKING` | Enable large-text chunking | `false` |
| `FEATURE_INLINE_SOURCE_VERIFICATION` | Enable inline source checking | `true` |
| **Trace & Safety** | | |
| `TRACE_SAFE_PAYLOADS` | Sanitize PII/sensitive data in traces | `false` |
| `TRACE_MAX_HEAD_CHARS` | Truncation limit for trace logs | `120` |

Retrieval pipeline behavior is fixed. There is no experiment-mode flag or
alternate retriever configuration.

---

## Claim Metadata

### `ClaimMetadata`

Metadata for claim-level verification routing.

```python
from spectrue_core.schema.claim_metadata import (
    ClaimMetadata,
    VerificationTarget,
    ClaimRole,
    MetadataConfidence,
    SearchLocalePlan,
    RetrievalPolicy,
    EvidenceChannel
)

metadata = ClaimMetadata(
    verification_target=VerificationTarget.REALITY,
    claim_role=ClaimRole.CORE,
    check_worthiness=0.9,
    search_locale_plan=SearchLocalePlan(primary="en", fallback=["uk"]),
    retrieval_policy=RetrievalPolicy(
        channels_allowed=[EvidenceChannel.AUTHORITATIVE, EvidenceChannel.REPUTABLE_NEWS]
    ),
    metadata_confidence=MetadataConfidence.HIGH
)
```

### `VerificationTarget`

What aspect of the claim to verify.

| Value | Description |
|-------|-------------|
| `REALITY` | Verify factual accuracy (standard) |
| `ATTRIBUTION` | Verify who said/did what |
| `EXISTENCE` | Verify source/document exists |
| `NONE` | Not verifiable (predictions, opinions) |

### `ClaimRole`

Role of claim in document structure.

| Value | Weight | Description |
|-------|--------|-------------|
| `CORE` | 1.0 | Central claim of the article |
| `SUPPORT` | 0.8 | Supporting evidence |
| `ATTRIBUTION` | 0.7 | Quote attribution |
| `AGGREGATED` | 0.6 | Summary from multiple sources |
| `SUBCLAIM` | 0.5 | Subordinate detail |
| `CONTEXT` | 0.0 | Background (no verification) |
| `META` | 0.0 | About the article itself |

### `EvidenceChannel`

Evidence source tiers.

| Value | Description |
|-------|-------------|
| `AUTHORITATIVE` | .gov, .edu, WHO, CDC, journals |
| `REPUTABLE_NEWS` | Reuters, AP, BBC, NYT |
| `LOCAL_MEDIA` | Regional/local news |
| `SOCIAL` | Twitter, Reddit (lead-only) |
| `LOW_RELIABILITY` | Blogs, forums (lead-only) |

### Properties

```python
# Check if claim should skip search
if metadata.should_skip_search:
    print("Claim is not verifiable")

# Check if claim is explain-only (no RGBA impact)
if metadata.is_explain_only:
    print("Claim is context only")

# Get RGBA weight
weight = metadata.role_weight  # 0.0 for CONTEXT, 1.0 for CORE
```

---

## Orchestrator

### `ClaimOrchestrator`

Builds execution plans for claims based on metadata.

```python
from spectrue_core.verification.orchestrator import ClaimOrchestrator
from spectrue_core.verification.execution_plan import BudgetClass

orchestrator = ClaimOrchestrator()

# Build execution plan
plan = orchestrator.build_execution_plan(
    claims=claims,
    budget_class=BudgetClass.STANDARD
)

# Get phases for a specific claim
phases = plan.get_phases("claim_id")
for phase in phases:
    print(f"Phase {phase.phase_id}: locale={phase.locale}, k={phase.max_results}")
```

### `BudgetClass`

Search budget levels.

| Value | Description |
|-------|-------------|
| `MINIMAL` | Quick check (Phase A only, k=3) |
| `STANDARD` | Normal analysis (A+B, k=5) |
| `DEEP` | Thorough analysis (A+B+C+D, k=7) |

### `Phase`

Search phase configuration.

```python
from spectrue_core.verification.execution_plan import Phase, phase_a, phase_b

# Factory functions
phase = phase_a("en")  # Primary locale, authoritative, k=3
phase = phase_b("en")  # +local media, advanced depth, k=5

# Phase properties
print(phase.phase_id)      # "A", "B", "C", "D", "A-light"
print(phase.locale)        # "en"
print(phase.channels)      # [EvidenceChannel.AUTHORITATIVE, ...]
print(phase.search_depth)  # "basic" or "advanced"
print(phase.max_results)   # 3, 5, or 7
```

---

## Phase Runner

### `PhaseRunner`

Executes progressive widening with early exit.

```python
from spectrue_core.verification.phase_runner import PhaseRunner

runner = PhaseRunner(
    search_mgr=search_manager,
    max_concurrent=3  # Parallel search limit
)

# Run all claims with waterfall pattern
evidence = await runner.run_all_claims(claims, execution_plan)

# evidence = {"claim_id": [source1, source2, ...], ...}
```

### `run_claim_phases`

Execute phases for a single claim.

```python
sources = await runner.run_claim_phases(
    claim=claim,
    phases=phases,
    existing_sources=[]  # Optional: already collected sources
)
```

### Trace Events

PhaseRunner emits these trace events:

| Event | Description |
|-------|-------------|
| `phase.started` | Phase execution began |
| `phase.completed` | Phase finished successfully |
| `phase.stopped` | Early exit (sufficiency met) |
| `phase.continue` | Proceeding to next phase |
| `phase.error` | Phase failed (fail-soft) |
| `search.failed` | Search API error |

---

## Sufficiency

### `evidence_sufficiency`

Check if collected evidence is sufficient.

```python
from spectrue_core.verification.sufficiency import (
    evidence_sufficiency,
    SufficiencyStatus
)
from spectrue_core.schema.claim_metadata import VerificationTarget

result = evidence_sufficiency(
    claim_id="c1",
    sources=search_results,
    verification_target=VerificationTarget.REALITY,
    claim_text="The claim text"
)

if result.status == SufficiencyStatus.SUFFICIENT:
    print(f"✓ Rule matched: {result.rule_matched}")
    print(f"  Reason: {result.reason}")
else:
    print(f"✗ Continue searching: {result.reason}")
```

### `SufficiencyResult`

```python
@dataclass
class SufficiencyResult:
    status: SufficiencyStatus  # SUFFICIENT, INSUFFICIENT, SKIP
    reason: str                # Human-readable explanation
    rule_matched: str | None   # "Rule1", "Rule2", "Rule3"
    authoritative_count: int   # Tier A sources found
    reputable_count: int       # Tier B sources found
    origin_count: int          # Origin sources found
```

### Sufficiency Rules

| Rule | Condition | Use Case |
|------|-----------|----------|
| **Rule 1** | 1 authoritative source with quote | Scientific claims, official stats |
| **Rule 2** | 2 independent reputable sources with quotes | News events |
| **Rule 3** | 1 origin source | Attribution/existence claims |

### Helper Functions

```python
from spectrue_core.verification.sufficiency import (
    is_authoritative,
    is_reputable_news,
    is_origin_source,
    get_domain_tier
)

# Check source tier
if is_authoritative("cdc.gov"):
    print("Tier A source")

tier = get_domain_tier("reuters.com")  # Returns "A", "B", "C", etc.
```

---

## RGBA Aggregation

### `aggregate_weighted`

Aggregate RGBA scores with role-based weighting.

```python
from spectrue_core.verification.rgba_aggregation import (
    aggregate_weighted,
    ClaimScore,
    claim_to_score
)

# Create scores from claims
scores = [
    ClaimScore(
        claim_id="c1",
        verified_score=0.95,
        danger_score=0.1,
        style_score=0.5,
        explainability_score=0.8,
        role_weight=1.0,          # CORE
        check_worthiness=0.9,
        evidence_quality=1.0
    ),
    ClaimScore(
        claim_id="c2",
        verified_score=0.5,
        danger_score=0.5,
        style_score=0.5,
        explainability_score=0.5,
        role_weight=0.0,          # CONTEXT (excluded)
        check_worthiness=0.1,
        evidence_quality=0.0
    )
]

# Aggregate
result = aggregate_weighted(scores)

print(f"Verified: {result.verified:.2f}")  # ~0.95 (context excluded)
print(f"Included: {result.included_claims}")  # 1
print(f"Excluded: {result.excluded_claims}")  # 1
```

### `claim_to_score`

Convert claim dict to ClaimScore using metadata.

```python
score = claim_to_score(
    claim=claim,  # dict with "metadata" key
    verified_score=0.95,
    danger_score=0.1,
    style_score=0.5,
    explainability_score=0.8
)

# Automatically extracts role_weight and check_worthiness from metadata
```

---

## Types Reference

### Core TypedDicts

```python
from spectrue_core.verification.evidence_pack import (
    Claim,
    SearchResult,
    EvidencePack
)

# Claim structure
claim: Claim = {
    "id": "c1",
    "text": "Original claim text",
    "normalized_text": "Normalized self-contained claim",
    "type": "core",
    "topic_group": "Politics",
    "importance": 0.9,
    "check_worthiness": 0.9,
    "harm_potential": 3,
    "search_queries": ["query 1", "query 2"],
    "query_candidates": [...],
    "metadata": ClaimMetadata(...),
}

# Search result structure
result: SearchResult = {
    "url": "https://example.com/article",
    "title": "Article Title",
    "content": "Article content...",
    "snippet": "Relevant snippet",
    "stance": "support",
    "quote": "Direct quote from source",
    "relevance_score": 0.85,
    "is_trusted": True,
}
```

---

## Error Handling

The engine uses fail-soft patterns:

```python
try:
    result = await engine.analyze_text(text, lang)
except Exception as e:
    # Engine returns partial results on failure
    if "verified_score" in result:
        # Use partial result with reduced confidence
        pass
```

### Common Exceptions

| Exception | Cause | Handling |
|-----------|-------|----------|
| `TavilyAPIError` | Search API failure | Returns empty results, continues |
| `OpenAIError` | LLM API failure | Returns fallback verdict |
| `ValidationError` | Invalid input | Raises immediately |

---

## Debugging

### Enable Trace

```python
import os
os.environ["SPECTRUE_ENGINE_DEBUG"] = "true"

# Traces are written to data/trace/<trace_id>.jsonl
```

### Trace Events

```python
from spectrue_core.utils.trace import Trace

# Custom trace events
Trace.event("my.event", {"key": "value"})

# View trace file
# data/trace/2024-12-21_22-30-00_abc123.jsonl
```

### Safe Payloads

By default, traces sanitize sensitive data:

```python
# Long strings are truncated (head only, no tail)
# PII is redacted
# Medical dosing is removed
```
