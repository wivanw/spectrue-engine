# Spectrue Engine — Failure Modes and Limits

This document describes expected failure modes and analytical limits of the Spectrue Engine.

In Spectrue, unresolved outcomes, uncertainty states, and negative results are not treated as system errors.
They are considered meaningful analytical signals that reflect the structure of real-world information.

---

## 1. Design Principle: Failure as Signal

Spectrue is designed under the assumption that:
- many real-world claims cannot be verified conclusively,
- evidence is often incomplete, delayed, or contradictory,
- and linguistic formulations frequently exceed what available data can support.

As a result, a high proportion of unresolved or partially resolved claims
is an **expected and correct outcome**, not a system deficiency.

---

## 2. Insufficient Evidence

### Description
A claim lacks enough relevant, direct, and specific evidence to support or refute it.

### Typical causes
- No primary or authoritative sources exist.
- Sources discuss the topic generally but not the specific claim.
- Numerical values, dates, or entities are missing.

### Observable signals
- `INSUFFICIENT_EVIDENCE` status
- Evidence items are topically related but non-conclusive.
- Trace indicates retrieval success but evidential insufficiency.

### Example
A claim such as:

> "The Mars Sample Return mission budget was reduced by NASA in Q3 2025."

is well-formed and specifies an entity, timeframe, and action.
However, no authoritative budget documents or official statements
confirm the specific reduction.

---

## 3. Conflicting Evidence

### Description
Available evidence contains mutually incompatible claims.

### Typical causes
- Different sources report divergent figures or conclusions.
- Preliminary reporting contradicts later corrections.
- Official statements conflict with secondary reporting.

### Observable signals
- `CONFLICTING_EVIDENCE` status
- Multiple evidence items with opposing stances.
- No resolution path without external judgment.

### Interpretation
Spectrue does not resolve conflicts by authority or averaging.
Conflicts are preserved and exposed.

---

## 4. Evidence Mismatch

### Description
Evidence is present but does not address the claim as stated.

### Typical causes
- Sources discuss related topics but not the claim's scope.
- Headlines overgeneralize underlying content.
- Claims conflate multiple distinct events or entities.

### Observable signals
- `EVIDENCE_MISMATCH` status
- High topical similarity with low claim alignment.
- Trace shows retrieval success but relevance failure after the escalation ladder is exhausted.

---

## 5. Unverifiable by Nature

### Description
A claim cannot be verified even in principle using external evidence.

### Typical causes
- Future-oriented claims.
- Hypothetical or counterfactual statements.
- Subjective judgments framed as facts.

### Observable signals
- `UNVERIFIABLE_BY_NATURE` status
- Lack of applicable retrieval targets.
- Explicit trace notes indicating epistemic impossibility.

---

## 6. Missing Context (Conceptual Category)

### Description
A claim omits essential contextual parameters required for verification.

### Typical missing parameters
- Time (when?)
- Scope (which instance?)
- Entity resolution (which organization/person?)
- Measurement frame (relative to what?)

### Notes
This category may manifest as `INSUFFICIENT_EVIDENCE`
or as structured trace warnings rather than a dedicated status code.

---

## 7. Underspecified Claims (Conceptual Category)

### Description
A claim is linguistically complete but analytically underdefined.

### Typical patterns
- Vague quantifiers ("significant", "major", "dramatic").
- Implicit baselines.
- Aggregated claims masking heterogeneous facts.

### Example
Claims such as:

> "Congress cut the budget of the mission"

without specifying:
- which mission,
- which fiscal year,
- or which budget line.

### Interpretation
Underspecification is treated as an analytical property of the claim,
not as a retrieval or model failure.

---

## 8. Pipeline Errors

### Description
The analytical pipeline fails to execute a stage reliably.

### Typical causes
- External service failures.
- Parsing or normalization errors.
- Unexpected input formats.

### Observable signals
- `PIPELINE_ERROR` status
- Trace interruption or abort markers.

### Interpretation
Pipeline errors indicate technical failure, not epistemic uncertainty,
and are distinguished clearly from evidential insufficiency.

---

## 9. Aggregation Limits

Even when individual claims are analyzed correctly,
article-level aggregation introduces additional limits:

- Heterogeneous claims may not admit a single summary.
- A small number of strong claims can coexist with many unresolved ones.
- Aggregated scores must not override per-claim statuses.

Spectrue treats aggregation as optional and secondary.

---

## 10. Non-Goals and Explicit Limits

Spectrue does not attempt to:
- infer author intent,
- judge misinformation severity,
- predict impact or harm,
- or arbitrate between competing worldviews.

Such tasks require normative or contextual judgments
outside the scope of analytical verification.

User-facing explanatory text may reference intent or narrative context
as part of synthesized explanations.
Such references are not system-level analytical outputs
and do not represent intent inference as a formal Spectrue function.

---

## 11. Summary

Failure modes in Spectrue are not edge cases.
They represent the dominant structure of real-world information.

By exposing insufficiency, conflict, mismatch, and epistemic limits explicitly,
Spectrue enables users to see **where verification stops**,
rather than pretending that it always succeeds.
