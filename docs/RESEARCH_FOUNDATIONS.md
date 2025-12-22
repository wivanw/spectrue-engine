# Research Foundations

## Policy-Driven Retrieval

Retrieval quality is a multi-objective tradeoff between precision, recall, and cost. A policy surface makes these tradeoffs explicit and measurable, rather than implicit in ad-hoc branching.

Key research-aligned principles:
- **Precision-first for mainline**: prioritize authoritative and reputable sources for lower-cost verification.
- **Recall-first for deep**: expand channel coverage and hop count for high-stakes or ambiguous claims.
- **Explicit budgets**: cap search hops and results to reduce runaway retrieval while preserving sufficiency checks.
- **Traceable decisions**: emit policy metadata in traces to enable QA review and reproducibility.
