# Spectrue Engine — Output Contract

This document defines the meaning and interpretation of Spectrue outputs.
It describes what the system produces, what those outputs represent,
and—equally important—what they do **not** represent.

This is a semantic contract, not an implementation guide.

---

## 1. Scope and Principles

Spectrue outputs are **analytical artifacts**, not authoritative judgments.

They are designed to:
- expose evidential structure,
- surface uncertainty and insufficiency,
- support independent human interpretation.

They are explicitly **not** intended to:
- determine objective truth,
- replace expert judgment,
- or provide final answers.

Per-claim outputs are the primary unit. In standard mode, the engine may also emit **article-level aggregates** derived from per-claim results.

---

## 2. Claim as the Unit of Output

Every Spectrue output is associated with a specific **claim**.

A claim is a minimal, checkable statement extracted from input text.
Complex inputs are decomposed into multiple claims, each evaluated independently.

Implications:
- Different claims from the same article may have different outcomes.
- Article-level summaries (if present) are derived aggregations and do not override per-claim outputs.
- No single output represents the "truthfulness" of the entire document.

---

## 3. Verdict Labels (Claim-Level)

Spectrue may emit **verdict-like labels** at the claim level.
These labels are categorical summaries of the analytical state, not truth assertions.

Typical labels include:
- **Verified / Refuted / Ambiguous / Unverified / Partially Verified** (see `VerdictStatus` in code).
- **Supported / Refuted / Conflicted / Insufficient Evidence** (see `VerdictState` in code).
These label sets represent different layers of the verdict model.

### Important:
- A verdict label does **not** imply certainty.
- NEI is a **valid analytical outcome**, not a system failure.
- Labels summarize evidence relationships; they do not replace evidence inspection.

---

## 4. Uncertainty and Status Codes

Uncertainty in Spectrue is represented explicitly through **statuses**, not implicitly through low confidence scores.

Spectrue distinguishes between implemented status codes and conceptual uncertainty categories.

Implemented status codes currently include (RGBA audit layer):
- `INSUFFICIENT_EVIDENCE`
- `CONFLICTING_EVIDENCE`
- `EVIDENCE_MISMATCH`
- `UNVERIFIABLE_BY_NATURE`
- `PIPELINE_ERROR`

In addition, the analytical model recognizes conceptual uncertainty categories such as:
- missing contextual parameters (e.g. time, scope, entity),
- underspecified or ill-defined claims.

These categories may be mapped to explicit status codes in future versions,
but are already reflected in trace data and analytical outcomes.

Statuses indicate **why** a claim cannot be resolved, not merely *that* it cannot.

A claim may have:
- one or more RGBA audit statuses,
- alongside a verdict label or instead of one.

---

## 5. RGBA Analytical Scores

Spectrue may compute scalar analytical scores along multiple dimensions
(e.g. RGBA-style dimensions such as Risk, Groundedness, Bias, Explainability).

### Interpretation rules:

- Scores are **orthogonal analytical signals**, not probabilities of truth.
- Scores are not calibrated to represent real-world likelihood.
- A high score in one dimension does not compensate for low scores in others.

Example:
- A claim may have low risk and high explainability while remaining NEI due to insufficient evidence.

Scores must always be interpreted **in conjunction with**:
- evidence artifacts,
- uncertainty statuses,
- and trace information.

---

## 6. Evidence Artifacts

Evidence is provided as **artifacts**, not as authoritative citations.

Each evidence item may include:
- source metadata,
- relevance indicators,
- stance with respect to the claim,
- coverage or scope notes.

Evidence items:
- may be partial,
- may conflict with each other,
- may fail to directly address the claim.

The presence of evidence does not imply confirmation.

---

## 7. Trace Output

Spectrue produces a **trace** documenting how outputs were generated.

The trace records:
- pipeline stages,
- decisions and state transitions,
- retrieval attempts and failures,
- aggregation steps.

The trace is:
- a byproduct of execution,
- grounded in actual system behavior,
- not a post-hoc explanation narrative.

User-facing explanations may be synthesized on top of trace data,
but they must not invent certainty beyond what the trace supports.

---

## 8. Article-Level Aggregation (Optional)

In some modes, Spectrue may compute **article-level summaries**
by aggregating per-claim outputs.

Key constraints:
- Aggregations are secondary to claim-level results.
- Aggregations do not override per-claim statuses or evidence.
- Aggregation logic must remain transparent and inspectable.

An article-level score is not a verdict on the article's truthfulness.

---

## 9. Common Misinterpretations (Explicitly Disallowed)

The following interpretations are incorrect:

- "Supported" means the claim is objectively true.
- A high score means the claim is probably true.
- NEI means the system failed or is uncertain by mistake.
- An article-level summary replaces claim-level analysis.
- The system decides what users should believe.

Spectrue does not make epistemic decisions on behalf of users.

---

## 10. Summary

Spectrue outputs are designed to make **limits visible**.

They expose:
- what is supported,
- what is contradicted,
- what is missing,
- and why resolution is not possible in many real-world cases.

Correct use of Spectrue requires engaging with its outputs
as analytical inputs—not as final answers.
