# Spectrue Engine API Reference

This document describes the Python API of Spectrue Engine.

## Table of Contents

- [Engine](#engine)
- [Configuration](#configuration)
- [Pipeline Factory](#pipeline-factory)
- [Claim Metadata](#claim-metadata)
- [Domain Modules](#domain-modules)

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

#### `analyze_text(text, lang, analysis_mode, progress_callback, max_credits, sentences)`

Analyze text for factual accuracy.

**Parameters:**
- `text` (str): The text to analyze
- `lang` (str): UI language code (e.g., "en", "uk", "de")
- `analysis_mode` (str): `"general"`, `"deep"`, or `"deep_v2"`
- `progress_callback` (Callable, optional): Progress callback function
- `max_credits` (int, optional): Budget cap (credits)
- `sentences` (list[str], optional): Pre-segmented sentences (skip segmentation)

**Returns:** `dict` with:
- Standard mode (`analysis_mode="general"`): article-level `rgba`, `verified_score`, `rationale`, `sources`, `claim_verdicts`, plus trace/billing metadata.
- Deep modes (`analysis_mode="deep"`/`"deep_v2"`): `deep_analysis.claim_results[]` with per-claim results and `cost_summary`.

**Example:**
```python
result = await engine.analyze_text(
    text="NASA discovered a new moon orbiting Earth.",
    lang="en",
    analysis_mode="deep_v2"
)
```

---

## Configuration

### `SpectrueConfig`

```python
from spectrue_core.config import SpectrueConfig

config = SpectrueConfig(
    openai_api_key="sk-...",           # Optional
    tavily_api_key="tvly-...",         # Optional
    google_fact_check_key="...",       # Optional
    min_confidence_threshold=0.7,      # Minimum confidence
    max_search_depth=3                 # Search iteration limit
)
```

---

## Pipeline Factory

The recommended way to execute the verification DAG.

### `PipelineFactory`

```python
from spectrue_core.pipeline.factory import PipelineFactory
from spectrue_core.pipeline.mode import PipelineMode

factory = PipelineFactory()

# Build pipeline for requested mode
pipeline = factory.build(PipelineMode.GENERAL_MODE)

# Execute pipeline
ctx = await pipeline.execute(
    text="NASA discovered a new moon...",
    lang="en"
)

print(ctx.verdict)
```

### `PipelineContext`

Holds state across pipeline steps.

| Attribute | Type | Description |
|-----------|------|-------------|
| `claims` | `list[Claim]` | Extracted atomic claims |
| `sources` | `list[SearchResult]` | Collected evidence items |
| `verdict` | `dict` | Final analysis results |
| `extras` | `dict` | Step-specific metadata and trace events |

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

---

## Domain Modules

Core business logic isolated from orchestration.

### `spectrue_core.domain.claims`
- `extraction.py`: Claim extraction and normalization.
- `graph.py`: Claim graph construction and ranking.
- `clustering.py`: Semantic claim clustering.

### `spectrue_core.domain.evidence`
- `clustering.py`: Evidence-to-claim mapping.
- `gating.py`: Semantic relevance and quality gating.
- `deduplication.py`: Content-level SimHash and publisher-level dedupe.

### `spectrue_core.domain.verification`
- `stance/evaluation.py`: Support/Refute/Context classification.
- `verdict/bayesian_update.py`: Log-odds belief updates and G-score computation.

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

# Traces are written to data/trace/<trace_id>.json
```

### Trace Events

```python
from spectrue_core.utils.trace import Trace

# Custom trace events
Trace.event("my.event", {"key": "value"})

# View trace file
# data/trace/2024-12-21_22-30-00_abc123.json
```