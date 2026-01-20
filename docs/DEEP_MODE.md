# Deep Mode Architecture (v5)

## Overview

Deep Mode is Spectrue's high-fidelity verification capability. Unlike Standard Mode, which offers a quick "glance" verdict, Deep Mode treats every claim as an independent investigation. The engine currently maps `"deep"` to `"deep_v2"` at runtime.

### v5 Evolution
**v2**: 1:1 Claim-to-Search execution. Expensive, slow (N claims = N searches).
**v5**: Cluster-Centric Execution + EVOI Gating.
- **Clustered Retrieval**: Similar claims share search budgets.
- **EVOI Gating**: Controls optional steps (stance/cluster) based on expected value.
- **Graph-Aware**: Can verify "Load-Bearing Claims" only.

---

## 1. What Deep Mode Enables

- **Granular Verdicts**: "Claim A is true, but Claim B is false."
- **Per-Claim Evidence**: Specific links for specific assertions.
- **No Hallucinated Consensus**: It won't average a lie and a truth into a "Half-Truth".
- **Auditability**: Every claim has its own trace, judge output, and reasoning.

## 2. What It Intentionally Does NOT Do

- **Global Context Smoothing**: It does NOT try to make the article "look good". If 90% of claims are flukes, it reports 90% errors.
- **Fallback Guessing**: If the LLM judge fails, it returns `status: error` with `rgba: null`. It **NEVER** returns `0.5`.
- **Latency Optimization**: It allows itself to be slower (seconds to minutes) to ensure correctness.

---

## 3. Architecture & Data Flow

```mermaid
graph TD
    A[Input Doc] --> B[Extract Claims]
    B --> C[Claim Graph Builder]
    C --> D[Identify Clusters / Load-Bearing Claims]
    D --> E[Cluster-Level Search Planning]
    E --> F[Execution: Search & Extract]
    F --> G[Evidence Attribution & Dedup]
    G --> H[Per-Claim ClaimFrame Assembly]
    H --> I[Deep Judge (LLM)]
    I --> J[Assemble Results]
```

### Key Components

1.  **ClaimClusters**: We group claims by explicit semantic similarity (using embeddings).
    - *Example*: "GDP grew by 2%" and "The economy expanded by 2%" -> **Cluster 1**.
2.  **Shared Budget**: The cluster gets a search budget. Results are pooled.
3.  **Attribution**: The pool is filtered against *individual* claims.
    - *Constraint*: Evidence for "GDP 2%" might not support "Inflation 5%" even if in the same cluster.

---

## 4. EVOI-Based Stopping

**Expected Value of Information (EVOI)** governs the **Search Escalation Ladder**.

For each claim (or cluster), we ask: "Will more search change the verdict?"
- **State A**: 0 sources. -> **High EVOI**. Search.
- **State B**: 1 reputable source (Reuters). -> **Low EVOI**. Stop.
- **State C**: 3 low-repute sources (Blogs). -> **Medium EVOI**. Escalate to authoritative search.
- **State D**: Conflicting reputable sources. -> **Stop** (Verdict is "Conflict").

> **Benefit**: Drastically reduces cost/time for obvious facts, reserves budget for contentious claims.

---

## 5. Execution Scope

Deep Mode (currently `deep_v2`) verifies all extracted claims, subject to a safety guard that caps the maximum number of claims per run.

---

## 6. The Deep Judge Contract

In Deep Mode, the final step is a **Per-Claim Judge** (LLM).

**Input**: `ClaimFrame` (Claim + Specific Evidence + Context).
**Output**: `DeepClaimResult` (RGBA, Verdict, Explanation).

**The Invariant**:
- The Judge output is **final** for that claim's semantic verdict.
- The Engine **does not** post-process or average these scores.
- The Engine **does** validate the format and ensures it adheres to the minimal `ClaimResult` contract (INV-030).

If the Judge outputs:
```json
{ "verdict": "supported", "rgba": [0.9, 1.0, 1.0, 1.0] }
```
...that is exactly what the user sees.

### Failure Handling
If the Judge fails (schema violation, refusal, timeout):
- **Result**: `status: "error"`, `rgba: null`.
- **Trace**: "Judge failed for claim X".
- **User View**: "Unable to verify this specific claim."

---

## 7. Evidence Stats & Counters

Deep Mode v5 exposes rigid counters for downstream consumers:
- `evidence_stats.direct_quotes`: Number of sources with verbatim matches.
- `evidence_stats.publishers_total`: Distinct publishers/domains.
- `confirmation_counts`:
    - `C_precise`: High-relevance, high-trust sources.
    - `C_corr`: Lower-relevance or cluster-based support.

These stats allow the UI/API consumer to display: "Verified by 2 independent sources (Reuters, AP)."
