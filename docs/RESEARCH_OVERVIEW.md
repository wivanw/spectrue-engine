# Spectrue Engine — Research Overview

## 1. Problem Space

Automated fact-checking is commonly framed as a classification problem:  
a system is expected to label a claim as *true*, *false*, or *uncertain*.

This framing is fundamentally flawed.

In real-world information environments:
- claims are often **compound**, internally inconsistent, or underspecified;
- available evidence is **fragmentary, delayed, or contradictory**;
- uncertainty is **structural**, not a temporary lack of data.

Treating fact-checking as a verdict-producing task forces AI systems to:
- collapse uncertainty into binary or scalar outputs;
- hide failure modes behind confidence scores;
- produce answers that appear authoritative while being epistemically weak.

Spectrue is designed from the opposite assumption:  
**fact-checking is an analytical process, not a classification outcome.**

---

## 2. Why Fact-Checking Is Not Classification

Most contemporary AI-based fact-checking systems implicitly assume:
- a stable ground truth,
- sufficient retrievable evidence,
- and a single evaluative endpoint.

In practice, none of these assumptions reliably hold.

Key failure modes of verdict-oriented systems include:
- premature truth assignment under incomplete evidence;
- inability to represent unresolved or conflicting sources;
- post-hoc rationalization of LLM outputs as "explanations".

Spectrue does not treat any output as an authoritative "truth label".

Instead, it treats verification as a **multi-stage analytical pipeline** whose outputs are:
- structured claims,
- evidence candidates,
- relational graphs,
- and explicit uncertainty signals.

The system is intentionally conservative about *certainty*: when evidence is insufficient,
it should return explicit insufficiency states rather than a midpoint guess.

---

## 3. Spectrue Methodology (High-Level)

Spectrue operates as a pipeline with deterministic control flow and probabilistic inference components (LLMs for semantic tasks). Orchestration, scoring aggregation, and budget enforcement are deterministic; claim extraction, stance labeling, and summarization are probabilistic.

At a high level, the process consists of:

1. **Claim Intake and Normalization**  
   Incoming text is treated as an information artifact, not a statement to be judged.  
   Claims are normalized without assuming correctness or intent.

2. **Claim Decomposition**  
   Complex or compound statements are decomposed into atomic sub-claims.  
   Decomposition preserves ambiguity where resolution is not possible.

3. **Evidence Discovery**  
   Evidence is retrieved as *candidates*, not confirmations.  
   Absence of evidence is preserved as a signal, not treated as failure.

4. **Graph Construction**  
   Relationships between claims and evidence are represented explicitly.  
   Contradictions, partial support, and gaps coexist within the same structure.

5. **Analytical Output Generation**  
   The system produces:
   - traceable reasoning paths,
   - evidence mappings,
   - uncertainty annotations,
   - and failure indicators.

Spectrue preserves per-claim evaluation as the primary unit of analysis.
In Standard Mode, the system may additionally compute article-level summary scores
as an aggregation over per-claim outputs, while keeping per-claim artifacts available.

---

## 3.1 What Spectrue Produces

Despite its analytical framing, Spectrue produces concrete outputs:

- **Verdicts (per-claim)**: categorical labels (e.g., Supported / Refuted / Mixed / NEI / Unverifiable)
- **Scores (per-claim and global)**: scalar metrics (e.g., RGBA-style dimensions such as Risk, Groundedness, Bias, Explainability)
- **Statuses**: machine-readable uncertainty / failure states (e.g., INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE)
- **Evidence artifacts**: evidence items with metadata (stance/relevance/coverage roles), plus counters (duplicates, independence)
- **Traces**: ordered events documenting decisions, costs, and state transitions across the pipeline

The distinction is not that Spectrue avoids outputs, but that it:
1. Refuses to hide uncertainty behind a confident-looking scalar,
2. Requires explicit statuses alongside scores,
3. Preserves trace and evidence artifacts for inspection.

---

## 3.2 Modes: Standard vs Deep

Spectrue operates in distinct modes with different tradeoffs:

- **Standard Mode**:
  - Produces per-claim outputs and may compute a global aggregation score.
  - Optimized for efficiency and baseline traceability.

- **Deep Mode**:
  - Emphasizes structural reasoning (graphs, clustering, evidence sharing across compatible claims).
  - Maintains per-claim outputs as the primary layer; global aggregation (if present) is secondary.
  - Intended to improve evidence sufficiency and trace clarity, not to increase verbosity.

---

## 4. Uncertainty as a First-Class Signal

In Spectrue, uncertainty is not a numeric confidence score applied at the end of inference.

Instead, uncertainty emerges from:
- missing links in the claim–evidence graph,
- conflicting evidence nodes,
- unstable decompositions,
- or failed retrieval stages.

These conditions are preserved and exposed to the user.

Uncertainty is therefore:
- **structural**, not merely probabilistic noise;
- **interpretable**, not silently averaged away;
- **actionable**, rather than hidden.

This allows downstream users (researchers, journalists, analysts) to reason about *why* a claim cannot be resolved, rather than receiving an opaque "low confidence" result.

Note: Spectrue may still emit scalar scores (e.g., groundedness-like values). Unless explicitly calibrated,
these scores should be treated as *analytical signals*, not literal probabilities of truth.

---

## 5. Traceability and Explainability

Spectrue emphasizes **process traceability**, not post-hoc explanation.

Every analytical stage contributes artifacts that can be inspected:
- intermediate representations,
- evidence candidates,
- graph relationships,
- and degradation points.

The trace is not an explanation generated after the fact,  
but a byproduct of the pipeline itself.

This distinction is critical:
- traces are reconstructions of actual system state transitions.

User-facing explanatory text may still be synthesized (e.g., by an LLM),
but it must be grounded in traceable evidence artifacts and must preserve uncertainty
instead of inventing certainty.

---

## 6. Explicit Non-Goals

Spectrue is intentionally **not** designed to:

- determine objective truth;
- replace human judgment;
- provide authoritative verdicts;
- optimize for persuasive or user-comforting outputs;
- mask uncertainty for usability.

The system is not a chatbot and not a moderation tool.
It does produce scores and verdict-like labels, but these are analytical outputs with explicit uncertainty,
not authoritative determinations.

Its role is to expose analytical structure, not to resolve it.

---

## 7. Research Orientation

Spectrue is developed as an open-source analytical engine intended for:
- research into AI-assisted verification,
- experimentation with uncertainty-aware pipelines,
- and exploration of non-verdict AI system design.

Negative results, unresolved claims, and partial analyses are considered valid and informative outcomes.

From a research perspective, failure cases are signals, not defects.

---

## 8. Summary

Spectrue reframes automated fact-checking as an **explicitly incomplete analytical process**.

By refusing to treat any output as authoritative, and by treating uncertainty as a first-class output, the system challenges prevailing assumptions about how AI should interact with knowledge claims.

Its contribution lies not in determining what is true, but in making visible **why truth determination is often not possible**.
