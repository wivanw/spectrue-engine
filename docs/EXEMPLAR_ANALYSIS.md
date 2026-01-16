# Spectrue Engine — Exemplar Analysis

This document provides a concrete example of how to interpret Spectrue outputs.
It is intended to illustrate correct reading of claim-level analysis,
especially in cases where many claims remain unresolved.

The example below is representative of real-world usage and reflects
expected system behavior.

---

## 1. Context

Input material:
- A news-style article discussing NASA missions, congressional budgeting,
  Mars Sample Return, and related political context.

The article contains:
- multiple factual claims,
- numerical assertions,
- causal implications,
- and evaluative language.

Spectrue analyzes the text by decomposing it into **independent claims**.
No global verdict for the article is produced in this mode.

---

## 2. Claim Decomposition

From the input text, Spectrue extracts multiple claims, including:

1. "Congress cut the budget of the mission."
2. "An ambitious Mars sample return mission has reached a bureaucratic end."
3. "Mars rovers provided convincing evidence of past warm and wet periods."
4. "The estimated cost of sample retrieval rose to $11 billion."
5. "Donald Trump expressed interest in acquiring Greenland for national security reasons."

Each claim is evaluated independently.
Their outcomes are not required to be consistent with each other.

---

## 3. Example: Underspecified Claim

### Claim
> "Congress cut the budget of the mission."

### Analysis
This claim lacks essential contextual parameters:
- which mission,
- which fiscal year,
- which budget line,
- and which numerical change.

### Outcome
- Verdict: NEI
- Conceptual category: Underspecified claim
- Typical statuses: `INSUFFICIENT_EVIDENCE` (with trace warnings)

### Interpretation
The system does not fail to retrieve evidence.
Instead, it identifies that the claim itself is not well-formed enough
to be verified.

This outcome reflects a property of the input text, not a system error.

---

## 4. Example: Conflicting Evidence

### Claim
> "The Mars sample return mission has effectively been cancelled."

### Analysis
- Some sources suggest cancellation or restructuring.
- Official NASA program pages still list the mission as active.

### Outcome
- Verdict: Mixed
- Status: `CONFLICTING_EVIDENCE`

### Interpretation
Spectrue preserves conflicting evidence without resolving it by authority.
The reader can inspect the evidence and trace to understand the disagreement.

---

## 5. Example: Insufficient Evidence (Well-Formed Claim)

### Claim
> "The estimated cost of Mars sample retrieval rose to $11 billion."

### Analysis
- The claim is specific and well-formed.
- Retrieved sources discuss budget overruns generally.
- No authoritative document confirms the $11B figure.

### Outcome
- Verdict: NEI
- Status: `INSUFFICIENT_EVIDENCE`

### Interpretation
This is a correct analytical outcome.
Absence of confirmation is preserved explicitly rather than guessed.

---

## 6. Example: Supported Claim

### Claim
> "Donald Trump expressed interest in acquiring Greenland for national security reasons."

### Analysis
- Multiple reputable media sources report the statement.
- Motivations are discussed consistently across sources.

### Outcome
- Verdict: Supported
- Evidence: multiple corroborating items
- Uncertainty: residual (no direct quotation in some sources)

### Interpretation
Even when a claim is supported, Spectrue does not assert objective truth.
It summarizes available evidence and preserves context.

---

## 7. Reading RGBA Scores in Context

In the example report, RGBA-style scores vary across claims.

Key observations:
- High explainability does not imply high certainty.
- Low groundedness often correlates with NEI outcomes.
- Bias scores reflect presentation style, not factual correctness.

Scores must always be interpreted together with:
- verdict labels,
- uncertainty statuses,
- evidence artifacts,
- and trace information.

---

## 8. Why Many NEI Outcomes Are Expected

In this example:
- several claims remain NEI,
- some claims are mixed,
- only a subset is clearly supported.

This distribution is normal for real-world articles that:
- use vague language,
- compress complex events,
- or imply facts without sourcing.

A high NEI ratio indicates **analytical honesty**, not system weakness.

---

## 9. What This Example Demonstrates

This exemplar analysis shows that Spectrue:

- does not collapse heterogeneous claims into a single verdict,
- distinguishes between poorly formed claims and missing evidence,
- preserves conflict instead of resolving it artificially,
- and makes epistemic limits explicit.

The system exposes where verification stops,
rather than pretending that it always succeeds.

---

## 10. Summary

Correct interpretation of Spectrue output requires:
- reading claim-level results independently,
- treating NEI and mixed outcomes as valid analytical states,
- and consulting evidence and trace artifacts.

Spectrue is designed to support scrutiny and understanding,
not to replace judgment or enforce conclusions.
