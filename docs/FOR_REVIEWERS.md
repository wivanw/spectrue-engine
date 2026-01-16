# Spectrue Engine — For Reviewers

This document provides a concise overview of the Spectrue Engine
for reviewers, funders, and evaluators.

Spectrue is an open-source analytical infrastructure for AI-assisted fact-checking.
It is designed as a research-oriented system and should not be interpreted
as a verdict-producing or truth-authoritative tool.

---

## 1. What Spectrue Is

Spectrue is a **claim-centric analytical engine**.

It:
- decomposes input text into individual claims,
- retrieves evidence as candidates rather than confirmations,
- evaluates claim–evidence relationships,
- and exposes uncertainty, insufficiency, and conflict explicitly.

Spectrue treats fact-checking as an **analytical process**, not a classification task.

---

## 2. What Spectrue Is Not

Spectrue does **not**:
- determine objective truth,
- produce authoritative judgments,
- replace human or expert decision-making,
- or hide uncertainty behind confidence scores.

Verdict-like labels and scores are analytical summaries,
not epistemic conclusions.

---

## 3. Why This Matters

Most automated fact-checking systems collapse complex evidence landscapes
into binary or scalar outputs.

This approach:
- hides uncertainty,
- obscures failure modes,
- and creates an illusion of authority.

Spectrue takes the opposite approach:
it makes epistemic limits visible and traceable.

Unresolved outcomes (e.g. "Not Enough Information") are treated
as **correct analytical results**, not as system failures.

---

## 4. How Spectrue Works (High-Level)

At a high level, Spectrue operates as a pipeline:

1. Claim extraction and normalization  
2. Claim decomposition into atomic units  
3. Evidence discovery (as candidates)  
4. Claim–claim and claim–evidence relationship analysis  
5. Claim-level analytical evaluation  
6. Traceable output generation  

Each stage produces inspectable artifacts.
No stage assumes that a final truth value exists.

---

## 5. Outputs and Interpretation

Spectrue produces outputs at the **claim level**, including:
- analytical labels (e.g. Supported, Refuted, Mixed, NEI),
- explicit uncertainty statuses,
- multi-dimensional analytical scores,
- linked evidence artifacts,
- and execution traces.

In some modes, article-level aggregation may be computed as a **secondary summary**.
Such aggregation does not override claim-level results.

---

## 6. Failure Modes Are Expected

A high proportion of unresolved or partially resolved claims
is normal in Spectrue analyses.

Typical reasons include:
- underspecified claims,
- missing contextual parameters,
- insufficient or conflicting evidence,
- or epistemically unverifiable statements.

These outcomes reflect properties of the input text and evidence landscape,
not deficiencies of the system.

---

## 7. Research Orientation

Spectrue is developed as an open-source research infrastructure.

Its goals include:
- studying uncertainty-aware AI pipelines,
- enabling independent scrutiny of automated verification,
- and supporting experimentation with non-authoritative system design.

Negative results and partial analyses are considered valid outputs.

---

## 8. Review Guidance

When evaluating Spectrue, reviewers are encouraged to focus on:
- clarity of analytical structure,
- explicit handling of uncertainty,
- traceability of decisions and failures,
- and alignment with open, inspectable infrastructure principles.

Spectrue should not be evaluated as a "fact-checking oracle",
but as a tool for exposing where verification succeeds and where it stops.

---

## 9. Where to Go Next

For more detail, see:
- `RESEARCH_OVERVIEW.md` — conceptual motivation and framing
- `ARCHITECTURE_AT_A_GLANCE.md` — pipeline overview
- `OUTPUT_CONTRACT.md` — precise meaning of outputs
- `FAILURE_MODES_AND_LIMITS.md` — expected limitations
- `EXEMPLAR_ANALYSIS.md` — concrete example of real output
