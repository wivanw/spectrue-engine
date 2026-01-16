# Spectrue Engine — Architecture at a Glance

This document provides a high-level overview of the Spectrue Engine architecture.
It is intended for readers who want to understand how the system works conceptually,
without reading the source code.

Spectrue is designed as an analytical pipeline, not as a monolithic model
and not as a verdict-producing system.

---

## 1. High-Level Pipeline

At a conceptual level, Spectrue processes input text through the following stages:

```
Input Text
    ↓
Claim Extraction & Normalization
    ↓
Claim Decomposition (per-claim units)
    ↓
Evidence Discovery (candidates, not confirmations)
    ↓
Claim–Evidence Graph Construction
    ↓
Analytical Evaluation (per-claim)
    ↓
Traceable Output (scores, statuses, evidence, trace)
```

Each stage produces inspectable intermediate artifacts.
No stage assumes that a final truth value exists or can be derived.

---

## 2. Deterministic Control vs Probabilistic Inference

Spectrue separates **control flow** from **inference**:

- The pipeline structure, stage ordering, and state transitions are deterministic.
- Inference components (e.g. language models, search ranking) are probabilistic.

This separation ensures that:
- uncertainty can be localized to specific stages,
- failures are attributable to concrete steps,
- and traces reflect actual system behavior rather than post-hoc explanations.

---

## 3. Claim-Centric Design

The fundamental unit of analysis in Spectrue is a **claim**, not a document.

Key implications:
- Each claim is evaluated independently.
- Evidence is linked to specific claims, not to articles as a whole.
- Article-level summaries (if produced) are derived from per-claim outputs and do not replace them.

This avoids collapsing heterogeneous statements into a single global verdict.

---

## 4. Evidence as Candidates

Evidence in Spectrue is treated as *candidate support*, not confirmation.

For each claim:
- multiple evidence items may be retrieved,
- evidence may partially support, contradict, or be irrelevant,
- absence of evidence is preserved as a signal.

The system does not infer truth from source authority alone.

---

## 5. Claim–Evidence Graph

Relationships in Spectrue are represented using two complementary structures:

- Claim–claim relationships are represented as a graph (ClaimGraph), capturing semantic proximity, dependencies, and relative importance between claims.
- Claim–evidence relationships are represented as attributed collections (evidence packs), where evidence items are linked to claims via identifiers and annotated with relevance, stance, and coverage metadata.

Together, these structures allow conflicting, partial, and missing support to be represented explicitly without collapsing them into a single unified graph.

---

## 6. Uncertainty Propagation

Uncertainty in Spectrue is not added at the end.

It emerges naturally from:
- missing graph connections,
- conflicting evidence nodes,
- unstable claim decompositions,
- failed retrieval or normalization steps.

These conditions result in explicit machine-readable statuses
(e.g. insufficient evidence, conflicting evidence),
rather than implicit low-confidence scores.

---

## 7. Outputs and Trace

Spectrue produces structured outputs per claim, including:
- categorical analytical labels,
- scalar analytical scores,
- explicit uncertainty statuses,
- linked evidence artifacts,
- and an ordered trace of pipeline decisions and state transitions.

The trace is a byproduct of execution, not a generated explanation.

---

## 8. What This Architecture Does Not Do

Spectrue does not:
- determine objective truth,
- enforce a single authoritative interpretation,
- hide uncertainty for usability,
- or collapse analysis into a single article-level verdict.

Its role is to expose analytical structure and limits,
not to resolve them.
