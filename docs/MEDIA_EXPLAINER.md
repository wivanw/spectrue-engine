# Media & Stakeholder Explainer (Spectrue Engine)

This page is for journalists, partners, and non-engineers.

## What Spectrue Engine does

Spectrue Engine takes a piece of text and produces:

- a list of extracted **claims** (things that can be checked),
- an evidence-backed **verdict** per claim,
- and four RGBA metrics:
  - **R** (Risk): potential harm if the text misleads people,
  - **G** (Grounded truth): how well the claims are supported/refuted by evidence,
  - **B** (Honesty of presentation): whether the text is framed misleadingly,
  - **A** (Explainability): how defensible and reproducible the explanation is.

The engine is designed to be **tamper-resistant**: it logs what it did and why.

## How verification works (simplified)

1. **Claim extraction**
   - The text is decomposed into atomic claims.
   - Each claim gets metadata (importance, role, links to other claims).

2. **Budgeted retrieval**
   - To control costs, the engine selects a small set of **verification targets** (typically 1–3) to search for evidence.
   - Deferred claims can share evidence from a related target claim.

3. **Evidence acquisition**
   - The engine searches the web, extracts relevant passages, and normalizes them into evidence items.

4. **Per-claim scoring**
   - Each claim is scored independently based on its evidence set.
   - The engine then aggregates per-claim results into a global output.

5. **Explainability adjustment**
   - Source tiers affect **A** (Explainability) via a deterministic *prior factor*.
   - Tiers do **not** clamp truth scores (G).

## What Spectrue Engine does NOT do

- It does not "decide truth" from source reputation alone.
- It does not hide costs: resource accounting is explicit and traceable.
- It does not require secret heuristics: the important knobs are documented.

## Open Source

Spectrue Engine is the open-source core of Spectrue (AGPLv3). The hosted product adds UI, billing, and operations.
