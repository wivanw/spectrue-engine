# Evidence Semantics in Spectrue
## Scoring vs Corroboration (v1)

**Status:** Design reference. The full two-channel SE/CE model is not fully implemented in the current engine. Treat this as intended behavior and verify against code before relying on these fields.

### Motivation
Spectrue must avoid a policy contradiction:

- We require multiple independent confirmations for strong groundedness (e.g., “≥2 independent sources”).
- We also deduplicate and filter evidence (e.g., duplicates, weak mentions, missing direct quotes).

If we apply strict scoring filters *before* assessing independence/coverage, we collapse the evidence set and make sufficiency rules unattainable.
This leads to misleading outputs such as:
- “Matched candidates: 3” but “Shown evidence: 1”.
- “Need ≥2 sources” but only 1 survives strict scoring.

This document defines a two-channel evidence model that resolves the contradiction.

---

## 1) Core Definitions

### 1.1 Evidence Candidate
A retrieved source (URL + title/snippet/extracted text) that is potentially relevant to one or more claims.

Candidates are not yet evidence; they are inputs to auditing.

### 1.2 Evidence Item
A candidate after extraction (text/snippets/citations available) that can be audited.

### 1.3 Evidence Channels
Spectrue maintains **two parallel evidence channels** per claim:

1) **Scoring Evidence (SE)**
   - High-precision evidence used for truth scoring (G) and conflict detection.
   - Must have clear stance and sufficient specificity.

2) **Corroboration Evidence (CE)**
   - Broader evidence used for assessing *independence*, *coverage*, and *sufficiency*.
   - Can be “mention/context” evidence that is relevant but not decisive.

**Key rule:** CE must never be discarded solely because it is not strong enough for SE.

---

## 2) Channel Membership Rules

### 2.1 Scoring Evidence (SE)
An evidence item belongs to SE if:
- stance ∈ {SUPPORT, REFUTE, MIXED}
- and the stance is based on claim-relevant specifics (date/quantity/entity constraints)
- and extraction confidence meets minimal threshold (configurable)

SE is used for:
- computing per-claim groundedness contributions
- conflict detection (support_mass vs refute_mass)
- forming the “primary evidence shown” set

### 2.2 Corroboration Evidence (CE)
An evidence item belongs to CE if:
- it passes structural relevance checks (deterministic):
  - overlap with structured claim terms (subject_entities + context_entities + retrieval_seed_terms) ≥ 1
  - and (if claim has time anchor) the evidence contains time-like mention OR time metadata
  - and (if claim has numeric anchor) the evidence contains numeric mention
- AND it is not classified as clearly unrelated

CE can include:
- stance ∈ {CONTEXT, MENTION, NEUTRAL}
- weak SUPPORT/REFUTE items that lack decisive specifics
- sources that are relevant but incomplete

CE is used for:
- independence counting via redundancy clustering
- sufficiency checks (“≥2 independent sources” means ≥2 independent clusters in CE, not ≥2 SE items)
- explaining “we found related sources but none decisive”

**Important:** CE is about *coverage and corroboration*, not about a final truth verdict.

---

## 3) Independence Model

### 3.1 Redundancy Clusters
Sources are clustered by redundancy (title/snippet/content overlap) to estimate independent reporting chains.

### 3.2 Independence for Sufficiency
Sufficiency for strong groundedness must be computed over CE clusters:

- `independent_support_clusters`
- `independent_refute_clusters`
- `independent_related_clusters` (relevant but non-decisive)

Two sources that are duplicates count as one cluster.

**Key rule:** “≥2 independent sources” means “≥2 independent clusters”, not “≥2 URLs” and not “≥2 SE items”.

---

## 4) Sufficiency Rules (Non-Contradictory)

### 4.1 Strong Confirmation (example policy)
A claim can be considered strongly supported only if:
- SE provides support evidence, AND
- CE contains at least 2 independent clusters supporting/covering the claim’s truth conditions,
  OR 1 strong authoritative source cluster (policy-dependent, still soft).

This prevents the contradiction:
- dedup is allowed,
- and sufficiency remains meaningful.

### 4.2 NEI / Insufficient Evidence
If SE is empty but CE contains relevant clusters:
- status is not “no evidence”
- status is “insufficient decisive evidence”
- the system must explain:
  - how many corroboration clusters exist,
  - why they are non-decisive (missing date, missing quantity, etc.)

---

## 5) Presentation / Output Requirements

Spectrue outputs must distinguish:
- `matched_candidates_count` (retrieval/matching level)
- `corroboration_sources_count` and `corroboration_clusters_count` (CE)
- `scoring_sources_count` (SE)

A system must not claim “only 1 source found” if multiple relevant corroboration sources exist.

---

## 6) Trace Requirements

For each claim, log:
- candidates_found_count
- CE_items_count, CE_cluster_count
- SE_items_count
- dropped_as_unrelated_count
- reasons for non-decisive CE items (e.g., missing date anchor, generic mention)

This prevents “hidden filtering” and makes policy consistent.

---

## 7) Non-Goals
- CE does not directly determine a truth score.
- CE does not replace SE.
- This model does not mandate hard caps or lexical heuristics.

---

## Summary
Spectrue uses a two-channel evidence model:
- **SE** for truth scoring (high precision),
- **CE** for independence and sufficiency (high coverage).

This resolves the policy contradiction between “require multiple independent sources” and “deduplicate/filter sources”.
