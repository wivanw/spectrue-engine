# Spectrue Engine — Documentation Index

This document serves as the primary entry point to Spectrue Engine documentation.

Spectrue is an open-source analytical infrastructure for AI-assisted fact-checking.
It is designed for readers who want to understand the system's purpose, limits,
and architecture without reading the source code first.

---

## 1. What Is Spectrue?

Spectrue is **not** a verdict-producing fact-checker.

It is an analytical engine that:
- decomposes text into individual claims,
- retrieves and evaluates evidence as candidates,
- exposes uncertainty, insufficiency, and conflict explicitly,
- and produces traceable analytical outputs.

Spectrue is designed to support scrutiny and understanding,
not to replace human judgment.

---

## 2. How to Read This Documentation

Different readers may have different goals.
Use the paths below to navigate the documentation efficiently.

---

## 3. For Reviewers, Funders, and Researchers

If you want to understand **what Spectrue is and why it exists**:

1. **`RESEARCH_OVERVIEW.md`**  
   Explains the problem Spectrue addresses, why verdict-based fact-checking fails,
   and how uncertainty-first analysis reframes the task.

2. **`ARCHITECTURE_AT_A_GLANCE.md`**  
   Provides a high-level architectural overview of the pipeline,
   separating deterministic control flow from probabilistic inference.

3. **`FAILURE_MODES_AND_LIMITS.md`**  
   Describes expected failure modes and epistemic limits,
   explaining why unresolved outcomes are correct and meaningful.

4. **`EXEMPLAR_ANALYSIS.md`**  
   Walks through a real example, showing how to interpret claim-level outputs
   and why many claims legitimately remain unresolved.

---

## 4. For Developers and Integrators

If you want to understand **what Spectrue produces and how to use it safely**:

1. **`OUTPUT_CONTRACT.md`**  
   Defines the semantic meaning of all outputs:
   verdict labels, statuses, scores, evidence artifacts, and traces.
   This document is essential before integrating Spectrue into other tools.

2. **Source Code (`/spectrue_core/`)**  
   Implements the analytical pipeline described in the documentation.
   Code follows the documented contracts rather than redefining them.

---

## 5. For Contributors

If you plan to contribute to Spectrue:

- Read **`OUTPUT_CONTRACT.md`** and **`FAILURE_MODES_AND_LIMITS.md`** first.
- Ensure new features preserve explicit uncertainty and traceability.
- Avoid introducing hidden aggregation or implicit verdict logic.

Spectrue treats failure cases as first-class analytical signals.
Contributions should respect this design principle.

---

## 6. What This Documentation Is Not

This documentation does not:
- provide usage tutorials or UI walkthroughs,
- describe deployment or hosting configuration,
- or serve as a marketing overview.

Its purpose is to define **epistemic and architectural contracts**.

---

## 7. Suggested Reading Order (Summary)

- New readers:  
  `RESEARCH_OVERVIEW.md` → `ARCHITECTURE_AT_A_GLANCE.md`

- Reviewers and funders:  
  `RESEARCH_OVERVIEW.md` → `FAILURE_MODES_AND_LIMITS.md` → `EXEMPLAR_ANALYSIS.md`

- Developers and integrators:  
  `OUTPUT_CONTRACT.md` → source code

---

## 8. Status

This documentation reflects the current analytical model and contracts of Spectrue.
As the engine evolves, documents may be extended, but core principles
(non-authoritative outputs, uncertainty-first analysis, traceability)
are considered stable.
