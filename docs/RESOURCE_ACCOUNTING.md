# Resource Accounting

This document describes how the Spectrue Engine measures resource consumption during verification runs and how usage is finalized for downstream systems.

## 1. Purpose

The Spectrue Engine provides:

- **Deterministic resource measurement** during verification pipelines
- **Unified usage accounting** across heterogeneous resource types
- **Transparent finalization semantics** for systems requiring discrete settlement

This specification is independent of any business model, pricing policy, or economic consideration. The engine measures and reports; interpretation is left to the consuming application.

---

## 2. Resource Units

### Spectrue Credit (SC)

The **SC** is the engine's universal unit of accounting. It provides a normalized measure for comparing and aggregating resource consumption across different providers and operations.

> SC is a **unit of accounting**, not a financial instrument. The engine does not assign monetary value to SC.

### Search Credit (TC)

**TC** (Tavily Credit) represents consumption of external retrieval services. The engine converts TC to SC for unified accounting:

| Operation | TC Cost |
|-----------|---------|
| Basic Search | 1 TC |
| Advanced Search | 2 TC |
| Extract (per batch) | 1 TC |

**Conversion rate:** `1 TC = 0.5 SC`

### Token-Based Accounting

LLM operations are measured in tokens. The engine tracks:

- **Input tokens** — prompt and context
- **Cached input tokens** — previously processed content (reduced cost)
- **Output tokens** — generated response

Token consumption is converted to SC using model-specific rates.

---

## 3. Cost Sources (Measured Inputs)

The engine measures and aggregates the following resource consumption:

### Search Operations

| Provider | Operation | TC | SC Equivalent |
|----------|-----------|-----|---------------|
| Tavily | Basic Search | 1 | 0.5 |
| Tavily | Advanced Search | 2 | 1.0 |
| Tavily | Extract (batch) | 1 | 0.5 |

### LLM Token Consumption

Reference rates (SC per token):

| Model | Input | Cached Input | Output |
|-------|-------|--------------|--------|
| GPT-5 Nano | 0.000005 | 0.0000005 | 0.00004 |
| GPT-5 Mini | 0.000025 | 0.0000025 | 0.0002 |
| GPT-5.2 | 0.000175 | 0.0000175 | 0.0014 |

> The engine converts external resource usage into a unified SC space for consistent measurement and reporting.

---

## 4. Continuous Accounting

### Core Principle

During a verification run, all resource consumption is measured **continuously** using fractional arithmetic. The engine:

1. Tracks each resource event as a `Decimal` value
2. Accumulates usage without intermediate rounding
3. Produces an `exact_usage_sc` at run completion

### Implementation

```
exact_usage_sc = Σ (resource_event.cost_sc)
```

Where each `cost_sc` is computed with full `Decimal` precision.

### Properties

| Property | Guarantee |
|----------|-----------|
| **Determinism** | Identical inputs produce identical `exact_usage_sc` |
| **Reproducibility** | Results can be independently verified |
| **Precision** | No precision loss during accumulation |
| **Transparency** | Full breakdown available per stage and provider |

### Terminology

- **`exact_usage_sc`** — The precise, unrounded total resource consumption
- **`fractional_usage`** — Any non-integer SC value during measurement

---

## 5. Finalization & Settlement Semantics

### Exact vs. Finalized Values

The engine produces an **exact fractional result**. Systems that require discrete units for settlement may apply a finalization rule.

| Value | Description |
|-------|-------------|
| `exact_usage_sc` | Precise `Decimal` value (e.g., `3.27`) |
| `finalized_usage` | Discrete value after rounding (e.g., `4`) |

### Caller-Applied Finalization

The engine **does not mandate** a specific rounding rule. It exports both exact and finalized values, allowing the caller to choose:

```python
# Engine provides
exact_usage_sc: Decimal  # e.g., Decimal("3.27")

# Caller may apply finalization
finalized = ceil(exact_usage_sc)  # e.g., 4
```

### Typical Settlement Rule

For systems requiring integer SC settlement, the **ceiling function** is commonly applied:

```
finalized_usage = ⌈exact_usage_sc⌉
```

This ensures:
- Fractional usage is always covered
- Exact integer values remain unchanged
- The rule is deterministic and transparent

> **Note:** The finalization rule is a **settlement concern**, not an economic policy. The engine documents the behavior; interpretation is application-specific.

---

## 6. Transparency Guarantees

The engine provides:

| Guarantee | Description |
|-----------|-------------|
| **Full Breakdown** | Usage itemized by stage (`search`, `extract`, `score`) and provider |
| **Exact Values** | Access to `exact_usage_sc` before any finalization |
| **Event History** | Complete list of resource events with timestamps |
| **No Hidden Logic** | All measurement follows documented rules |

### Breakdown Structure

```python
RunCostSummary(
    total_credits=Decimal("3.27"),      # exact_usage_sc
    by_stage_credits={"search": Decimal("1.5"), "llm": Decimal("1.77")},
    by_provider_credits={"tavily": Decimal("1.5"), "openai": Decimal("1.77")},
    events=[...]                         # Full event list
)
```

---

## 7. Numerical Examples

### Example A: Search Only

**Scenario:** 3 Tavily Basic Searches

| Operation | TC | SC |
|-----------|-----|-----|
| Basic Search ×3 | 3 | 1.5 |

```
exact_usage_sc = 1.5 SC
```

---

### Example B: Realistic Mixed Workload (Standard Run)

**Scenario:** 3 Basic Searches + 1 Advanced Search + Scoring (GPT-5.2) + Extraction (Nano)

| Resource | Calculation | SC |
|----------|-------------|-----|
| Search (Basic) | 3 TC × 0.5 | 1.50 |
| Search (Adv) | 2 TC × 0.5 | 1.00 |
| GPT-5.2 Score | 4.6k tokens (mixed) | 1.54 |
| GPT-5 Nano | 12k tokens (extract) | 0.06 |
| GPT-5 Mini | 3k tokens (plan) | 0.08 |

```
exact_usage_sc = 1.50 + 1.00 + 1.54 + 0.06 + 0.08 = 4.18 SC
```

---

### Example C: Finalization Behavior

Demonstrating ceiling finalization on various exact values:

| exact_usage_sc | finalized (ceil) |
|----------------|------------------|
| 19.01 | 20 |
| 20.00 | 20 |
| 20.99 | 21 |
| 0.01 | 1 |

The finalization rule is:
- Deterministic
- Preserves exact integers
- Rounds fractional values to the next integer

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Code Index](./CODE_INDEX.md)
