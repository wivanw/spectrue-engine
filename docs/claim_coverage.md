# Deterministic Claim Coverage via Anchors

> Structural enforcement of high-recall claim extraction

## Motivation

LLM-only claim extraction may under-extract verifiable facts depending on text density, model behavior, or prompt variance. Spectrue enforces claim coverage structurally through **deterministic anchors**.

## Anchors

Spectrue extracts deterministic anchors from raw text before LLM processing:

| Anchor Type | Pattern | Example |
|------------|---------|---------|
| **Time** | YYYY, YYYY-MM-DD, Q1 2024, January 2025 | `2024-01-15`, `Q4 2023` |
| **Numeric** | Currency, percentages, large numbers | `$5.2 billion`, `15%` |
| **Quote** | Quoted spans with various quote marks | `"We expect growth"` |

Anchors represent facts that **must not disappear** during extraction.

## Hard Guard

The system injects `DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS` into every LLM extraction call:

```python
DEFAULT_CLAIM_EXTRACTION_INSTRUCTIONS = """
You are extracting factual claims for verification.
Do NOT summarize. Do NOT judge truth.
Your task is to enumerate verifiable factual units.

Rules:
- Prefer over-extraction to under-extraction.
- Extract events, measurements, quantities, dates, and quoted statements.
- Each claim must be atomic and independently verifiable.
- Output must strictly follow the provided JSON schema.
"""
```

This prevents extraction failures from missing instructions.

## Coverage Skeleton

The LLM produces a coverage skeleton where each item references one or more anchors via `anchor_refs`. Anchors not used must be explicitly listed in `skipped_anchors` with a structural reason:

- `not_a_fact` - Not a verifiable proposition
- `duplicate_of` - Already covered by another item
- `malformed` - Cannot parse anchor text
- `navigation` - UI/navigation element
- `boilerplate` - Template/footer text

## Validation and Repair

After extraction, Spectrue validates that all anchors are either covered or skipped. If gaps exist, a targeted gap-fill LLM call adds only the minimal items needed.

This guarantees **high recall** without numeric caps or lexical heuristics.

## Guarantees

- ✅ **No top-K or max-claim limits**
- ✅ **No word-based importance heuristics**
- ✅ **Every detected anchor is accounted for**
- ✅ **Instructions always present** (hard guard)

## Usage

```python
from spectrue_core.verification.claims import (
    extract_all_anchors,
    get_anchor_ids,
    anchors_to_prompt_context,
)

# Extract anchors from text
text = "On 2024-01-15, Tesla announced $5.2 billion revenue."
anchors = extract_all_anchors(text)

# Get anchor IDs
ids = get_anchor_ids(anchors)  # {"t1", "n1"}

# Format for LLM prompt
context = anchors_to_prompt_context(anchors)
```

## Trace Events

| Event | Description |
|-------|-------------|
| `claim_extraction.guard.instructions_injected` | Instructions injected into LLM call |
| `claims.coverage.anchors` | Extracted anchors with counts by type |
| `claims.coverage.gaps` | Missing anchor IDs after skeleton extraction |
| `claims.coverage.gapfill` | Gap-fill repair attempt results |

## Related

- [VERIFICATION_METRICS.md](./VERIFICATION_METRICS.md) - RGBA scoring contract
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System components
