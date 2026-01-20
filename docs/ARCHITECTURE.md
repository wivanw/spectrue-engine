# Spectrue Engine Architecture

Spectrue is an **analytical verification engine**. It decomposes complex texts into atomic claims, retrieves evidence for each, and produces explainable RGBA scores (Risk, Groundedness, Bias, Explainability).

> **Core Philosophy**: Spectrue values correctness over confidence. It prefers to say "I don't know" (uncertainty) rather than hallucinate a verdict. It uses LLMs as semantic allocators and summaries, but relies on deterministic logic for scoring and aggregation.

---

## 1. Project Philosophy

### Analytical, Not Oracular
Spectrue is **not** a chatbot that "knows" the truth. It is a research engine that:
1.  **Extracts** verifiable claims.
2.  **Retrieves** external evidence.
3.  **Measures** the semantic distance between claims and evidence.
4.  **Aggregates** these measurements into scores.

### Uncertainty as a Feature
We explicitly model uncertainty. If evidence is insufficient, contradictory, or off-topic, the system reports this status rather than forcing a probabilistic score.
- **Low Confidence** is a valid and valuable output.
- **Missing Evidence** is distinct from **Refuting Evidence**.

### Deterministic Core, LLM Periphery
- **Deterministic**: Scoring math, aggregation logic, budget enforcement, graph ranking.
- **Probabilistic (LLM)**: Text understanding, claim extraction, query generation, semantic matching, summarization.
**Invariant**: The LLM never "decides" the final score directly in Standard Mode. It provides labeled signals (stance, relevance) which the engine aggregates. (Deep Mode has a specific exception, see `DEEP_MODE.md`).

---

## 2. Claims Model

### Atomic Claims
A **Claim** is the fundamental unit of verification. It must be:
- **Falsifiable**: "Deep learning is popular" (Yes) vs "Deep learning is magic" (No).
- **Atomic**: Contains one assertion. "X happened AND Y happened" is split.
- **Context-Aware**: Carries inherited context (time, location, entities) from its document.

> **Why we never merge claims**: Merging claims obscures verification status. If Claim A is true and Claim B is false, merging them yields a muddy "Partial". We keep them separate to pinpoint misinformation.

### Claim Metadata & Roles
Each claim has a `ClaimRole`:
- `core`: The main points of the text.
- `context`: Background info (not verified, but used for grounding).
- `quote`: Attribution check (did X say Y?).
- `statistic`: Numerical fact check.

### Claim Graph
The **Claim Graph** structures claims by semantic dependency.
- **Purpose**: To identify "Load-Bearing Claims" (if this falls, the argument falls).
- **Capabilities**: PageRank-style centrality, redundancy clustering.
- **Non-Goal**: It does not infer truth propagation (False premise $\nRightarrow$ False conclusion).

---

## 3. Evidence Model

### Decisive vs. Corroborative
- **Decisive**: A high-trust source directly affirming/negating the claim.
- **Corroborative**: Secondary sources, repetition, or indirect validation.

### Evidence Metadata
Every `EvidenceItem` carries structured signals:
- `stance`: `support`, `refute`, `context`, `irrelevant`.
- `relevance`: 0.0 to 1.0 (semantic distance).
- `provider`: Where it came from (Tavily, Google, etc.).
- `checks`: List of passing/failing sanity checks (date match, entity match).

### Insufficient Evidence
If no decisive evidence is found after the **Evidence Acquisition Ladder (EAL)** is exhausted, the RGBA audit status may be `INSUFFICIENT_EVIDENCE`.
- **RGBA audit statuses** use `null` values when evidence is insufficient or off-topic.
- **We distinguish** "No results found" vs "Results found but irrelevant" vs "Results relevant but inconclusive".

---

## 4. Deduplication Semantics

To prevent "illusion of consensus" (10 papers citing the same AP wire), we enforce strict deduplication:

1.  **URL-Level**: Canonical URL normalization.
2.  **Content-Level**: MinHash/SimHash of the precise extracted snippet.
3.  **Publisher-Level**: "Times of London" and "Sunday Times" might be distinct, but multiple URLs from `cnn.com` are grouped.

**Confirmation Counters**:
- `precise_publishers`: Count of distinct domains supporting the claim.
- `corroboration_clusters`: Count of semantic clusters supporting the claim.
> Only `precise_publishers` drives the **Groundedness (G)** score. 100 links from the same blog farm count as 1.

---

## 5. Retrieval Architecture

### Hierarchy
1.  **Orchestrator**: Decides *which* claims need verification (based on check-worthiness).
2.  **Search Manager**: Executes the **Search Escalation Policy**.
3.  **Provider Layer**: Tavily, Google, Archive, etc.

### Claim-Level vs. Cluster-Level
- **Standard Mode**: Global retrieval context. Claims share a pool of evidence.
- **Deep Mode (v5)**: **Cluster-Level Retrieval**. Claims are grouped by semantic similarity. Queries are generated for the *cluster* to save budget, but evidence is attributed back to individual claims.

### Compatibility Checks
Evidence is **never** blindly reused. Even within a cluster, an evidence item must pass the `EvidenceCompatibility` check (verified against the specific claim's entities and predicates) before being attached.

---

## 6. LLM Usage Rules

### Roles
The LLM is an **Annotator**, not a Judge (except in Deep Mode).
- **GOOD**: "Does this text support the claim?" (Stance Classification).
- **BAD**: "Is this claim true based on your training data?" (Hallucination risk).

### Schema Validation & Repair
All LLM outputs are strictly typed (Pydantic).
- **Validation**: JSON Schema enforcement.
- **Repair**: If schema fails, we run a "Repair Loop" (max 1 retry) with error feedback.
- **Fallback**: If repair fails, the step errors out safely. We do not try to parse broken JSON.

### Constraints
- **No Authoritative Verdicts**: The LLM cannot override the absence of evidence.
- **Cost Awareness**: We request `reasoning_effort="low"` for routine tasks and save "high" for complex deep-mode judging.

---

## 7. Embeddings & Text Processing

### No Full-Doc Embeddings
We never embed an entire article as a single vector. It dilutes specific claims.
- **Chunking**: Text is split into semantic sentences/paragraphs.
- **Claim-Centric**: We embed the *Claims*, then search for evidence that matches the Claim vector.

### Safety Nets
- **Token Limits**: Hard caps on input size to prevent cost spikes.
- **Truncation**: Deterministic truncation of middle sections if limit exceeded (preserving head/tail context).

---

## 8. Resource Awareness

### Credit Accounting
Spectrue uses a unified `SpectrueCredit` (SC) currency.
Rates are configured in `spectrue_core/billing/default_pricing.json` and can be overridden at runtime.

### EVOI (Expected Value of Information)
We use a lightweight EVOI model to decide on **Search Escalation**:
- If we already have 2 `authoritative` sources, EVOI of more search is near zero. **STOP**.
- If we have 5 `low-reliability` sources, EVOI is high. **ESCALATE**.

---

## 9. Stability & Extension Guidelines

### Invariants (Do Not Break)
1.  **I0: Single Source of Truth**. Algorithms must be consistent across modes.
2.  **I7: LLM is Auditor (Standard)**. Do not let LLM just guess the score.
3.  **I8: Typed Statuses**. Use explicit statuses (RGBA audit) rather than implicit "unknown" numeric fallbacks.

### Anti-Patterns
- **Heuristic Caps**: "If > 5 sources, score = 1.0". (Wrong, what if they are all blogs?)
- **Silent Failures**: Catching expectations and returning `None` without a trace event.
- **Overconfidence**: Rounding 0.81 to 1.0.

### Adding Signals
To add a new signal (e.g., "Clickbait Score"):
1.  Define it in `EvidenceItem` or `ClaimMetadata`.
2.  Add an extractor (Regex or LLM).
3.  **Log it** in Trace.
4.  **Do not** weight it in RGBA until it is backtested.
