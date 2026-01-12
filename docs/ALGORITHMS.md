# Algorithms (Spectrue Engine)

This document describes the core retrieval, verification, and scoring algorithms 
used by the engine. Each mechanism is classified by its epistemological status.

---

## Classification Legend

| Category | Meaning |
|----------|---------|
| **[A] Formally Grounded** | Based on peer-reviewed academic work |
| **[B] Engineering Heuristic** | Empirically tuned, not derived from theory |
| **[C] Constraint Safeguard** | Hard limit to prevent overconfidence |

---

## Scoring Mechanisms

### Log-Odds Representation [A]

**Status:** Formally Grounded (Probability Theory)

Beliefs are represented in log-odds (logit) space where updates become additive:

```
Posterior(log-odds) = Prior(log-odds) + Evidence(log-odds)
```

**Code location:** `spectrue_core/scoring/belief.py` → `prob_to_log_odds()`, `update_belief()`

**Why log-odds:**
- Additive updates (vs multiplicative in probability space)
- Natural handling of extreme beliefs without numerical underflow
- Standard representation in Bayesian inference

> **Reference:** Good, I.J. (1950). *Probability and the Weighing of Evidence*.
> This is basic probability theory, not a novel algorithm.

---

### BeliefState Data Structure [A]

**Status:** Formally Grounded (Standard Representation)

```python
class BeliefState:
    log_odds: float  # log(P / (1 - P))
    confidence: float  # Meta-certainty [0, 1]
```

**Code location:** `spectrue_core/schema/scoring.py`

---

### Sigmoid Impact Function [B]

**Status:** Engineering Heuristic

Evidence impact uses a sigmoid to dampen weak evidence:

```python
impact = relevance × L_max × sigmoid(strength)

# Parameters (empirically tuned):
k = 10.0      # Steepness
x₀ = 0.5      # Midpoint  
L_max = 2.0   # Maximum log-odds impact
```

**Code location:** `spectrue_core/scoring/belief.py` → `sigmoid_impact()`

> ⚠️ **Epistemological Note:** This is an **engineering decision** inspired by 
> logistic weighting, NOT derived from Bayesian inference or Pearl's framework.
> The parameters were hand-tuned to satisfy noise tolerance requirements (SC-002).
> There are no theoretical guarantees about optimality.

---

### Consensus Cap [C]

**Status:** Constraint-Based Safeguard (Non-Bayesian)

Article credibility is capped by scientific consensus:

```python
if consensus.source_count >= 2:
    posterior = min(posterior, consensus_limit)
```

**Code location:** `spectrue_core/scoring/belief.py` → `apply_consensus_bound()`

> ⚠️ **Epistemological Note:** This is a **hard constraint**, NOT a Bayesian update.
> It implements the principle that a single article cannot exceed scientific consensus.
> This is a safeguard against overconfidence, not probabilistic inference.

---

### Claim Graph Propagation [A]

**Status:** Formally Grounded (Belief Propagation on DAGs)

Beliefs propagate through claim dependencies using single-pass message passing:

```
For each node in topological order:
    message = Σ (parent.belief × edge.weight × sign)
    node.propagated_belief = node.local_belief + message
    
sign = -1 if CONTRADICTS else +1
```

**Code location:** `spectrue_core/graph/propagation.py` → `propagate_belief()`

**Formal basis:**
- Single-pass BP is exact for DAGs (no loopy approximation needed)
- Topological ordering ensures parents are processed before children
- Log-odds additivity preserves mathematical correctness

> **Reference:** Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*.

**Doc-to-Code Matrix (Propagation):**

| Doc Step | Runtime Behavior | Code Location |
|----------|------------------|---------------|
| Topological sort for DAG processing | `ClaimContextGraph.topological_sort()` | `spectrue_core/graph/context.py` |
| Message passing with edge sign/weight | `message = source_log_odds * edge.weight * sign` | `spectrue_core/graph/propagation.py` |
| Update propagated belief | `node.propagated_belief = local + Σ messages` | `spectrue_core/graph/propagation.py` |
| Use in scoring flow | Context graph passed into evidence pipeline | `spectrue_core/verification/pipeline_evidence.py` |

---

### RGBA Belief Dimensions [B]

**Status:** Engineering Decision

Each dimension is tracked independently:

| Dimension | Meaning | Prior Source |
|-----------|---------|--------------|
| R (Danger) | Harm potential | Inverse of tier |
| G (Veracity) | Factual accuracy | Source tier |
| B (Honesty) | Presentation honesty | Neutral (0.5) |
| A (Explainability) | Evidence availability | Neutral (0.5) |

**Code location:** `spectrue_core/scoring/rgba_belief.py`

> ⚠️ **Epistemological Note:** The independence assumption is an **engineering choice**.
> Factual accuracy may correlate with presentation honesty in practice.

---

## Retrieval Mechanisms

### Fixed Retrieval Pipeline [C]

**Status:** Constraint-Based Safeguard (Single deterministic path)

The engine uses one fixed retrieval pipeline with batch-only extraction.
There are no alternate retrievers or experiment branches.

**Data Structures (invariants):**

```python
NormalizedUrl = str

class GlobalUrlRegistry:
    urls: dict[NormalizedUrl, UrlMeta]

class UrlMeta:
    status: Literal["seen", "extracted", "failed"]
    first_seen_stage: int
    seen_by_claims: set[ClaimId]

class ExtractorQueue:
    pending: list[NormalizedUrl]          # status == "seen" only
    extracted: dict[NormalizedUrl, ExtractedContent]

class ClaimBindings:
    eligible: dict[ClaimId, set[NormalizedUrl]]
    audited: dict[ClaimId, set[NormalizedUrl]]
```

**Core invariants:**
- `NormalizedUrl` is the unique key.
- If `status == "extracted"`, extraction never repeats.
- `pending` contains only URLs with `status == "seen"`.
- Extraction only happens in `extract_all_batches()` with batch size 5.

**Pipeline stages:**

```
Stage 0 — Universal Search
urls = tavily.search(universal_query)
register_urls(stage=0, claim_ids=ALL, urls=urls)
extract_all_batches()
bind_after_extract()

Stage 1 — Graph Priority
core_claims = pick_by_graph_centrality(claim_graph)
for claim in core_claims:
  urls = tavily.search(query_for(claim))
  register_urls(stage=1, claim_ids={claim.id}, urls=urls)
extract_all_batches()
bind_after_extract()

Stage 2 — Escalation (algorithmic)
for claim in claims:
  S = compute_sufficiency(claim.metadata)
  if S < S_min:
    urls = tavily.search(query_for(claim))
    register_urls(stage=2, claim_ids={claim.id}, urls=urls)
extract_all_batches()
bind_after_extract()
```

**Sufficiency formula (fixed):**

```
S = (w1 * CE_cluster_count
     + w2 * SE_support_mass
     - w3 * conflict_mass
     - w4 * missing_constraints)
```

There are no additional heuristics or early-stop rules beyond `S < S_min`.

**Trace events (retrieval):**
- `urls_registered(stage, count)`
- `extract_batch_started(batch_size)`
- `extract_batch_finished(success_count)`
- `bind_completed(stage)`

**Code location:** `spectrue_core/verification/retrieval/fixed_pipeline.py`,
`spectrue_core/pipeline/steps/retrieval/web_search.py`

---

### Evidence Acquisition Ladder (EAL) [B]

**Status:** Engineering Heuristic + Bayesian Budget Model

Used for post-search content enrichment (e.g., chunk fetch/inline verification).
The fixed retrieval pipeline uses batch-only extraction and does not use EAL to
control search stages.

Escalation strategy to enrich sources with quotes:
1. Start with snippets from search results
2. Fetch full content for top candidates (Tavily Extract)
3. Extract quotes (semantic or heuristic)
4. Use Bayesian EVOI model to decide when to stop fetching

**Code location:** `spectrue_core/verification/search/search_mgr.py` → `apply_evidence_acquisition_ladder()`

#### Bayesian Budget Allocation

EAL uses Expected Value of Information (EVOI) to decide when to stop:

```
EVOI(next) = value × entropy × posterior_mean × decay × (1 - 0.5 × sufficiency)
Continue if: EVOI(next) ≥ marginal_cost (default: 0.5)
```

**Budget Parameters** (`ExtractBudgetParams` in `spectrue_core/scoring/budget_allocation.py`):
- `marginal_cost_per_extract = 0.5` — EVOI threshold
- `diminishing_returns_decay = 0.85` — each extract worth 85% of previous
- `min_extracts = 2` — always try at least 2
- `max_extracts = 12` — hard ceiling

#### Per-Claim Budget Trackers (Deep Mode)

In deep mode, claims are verified in parallel. Each claim gets an **independent** `GlobalBudgetTracker`:

```python
# Per-claim trackers dictionary
self._claim_budget_trackers: dict[str, GlobalBudgetTracker] = {}

# Get tracker for specific claim
tracker = search_mgr.get_claim_budget_tracker(claim_id)
```

This prevents budget pollution between claims running concurrently via `asyncio.gather()`.

#### Separate Inline vs Claim Budgets

Two independent budget contexts:
- `inline_budget_tracker` — for inline source verification (homepage links, article references)
- Per-claim trackers — for claim evidence acquisition

Inline sources often have lower quote hit-rates, so their Bayesian prior shouldn't pollute claim verification.

---


### Coverage Skeleton Extraction [B]

**Status:** Engineering Heuristic

Two-phase claim extraction to ensure comprehensive coverage:

**Phase 1: Skeleton Extraction**
```
Text → LLM → CoverageSkeleton {
    events: [SkeletonEvent],      # Subject + verb + time/location
    measurements: [SkeletonMeasurement],  # Metric + quantity
    quotes: [SkeletonQuote],      # Speaker + quote_text
    policies: [SkeletonPolicy]    # Subject + action
}
```

**Phase 2: Coverage Validation**
```python
# Regex-based detection (language-agnostic for numbers/dates)
detected_times = extract_time_mentions_count(text)
detected_numbers = extract_number_mentions_count(text)
detected_quotes = detect_quote_spans_count(text)

# Validation with tolerance
ok, reason_codes = validate_skeleton_coverage(
    skeleton, analysis, tolerance=0.5
)
# reason_codes: ["low_time_coverage:1/5", "low_quote_coverage:0/2"]
```

**Phase 3: Skeleton → Claims**
```python
# Per-type converters with validation
results, emitted, dropped = skeleton_to_claims(skeleton)
# dropped claims logged with reason_codes
```

**Code location:** `spectrue_core/agents/skills/coverage_skeleton.py`

> ⚠️ **Epistemological Note:** The tolerance threshold (0.5) and regex patterns are
> **engineering choices** tuned for precision/recall balance. Quote detection pattern
> supports multiple quotation styles (ASCII, curly, guillemets, German).

---

### Claim Verifiability Validation [B]

**Status:** Engineering Heuristic (Deterministic Rules)

Structural validation enforcing verifiable claims without numeric caps:

```python
def validate_core_claim(claim: dict) -> tuple[bool, list[str]]:
    reason_codes = []
    
    # Structural checks
    if not claim.get("claim_text"): 
        reason_codes.append("empty_claim_text")
    if not claim.get("subject_entities"):
        reason_codes.append("missing_subject_entities")
    if len(claim.get("retrieval_seed_terms", [])) < 3:
        reason_codes.append("insufficient_retrieval_seed_terms")
    
    # Falsifiability check
    falsifiability = claim.get("falsifiability", {})
    if not falsifiability.get("is_falsifiable"):
        reason_codes.append("not_falsifiable")
    
    # Time anchor check (with exemptions)
    predicate_type = claim.get("predicate_type", "other")
    if predicate_type not in TIME_ANCHOR_EXEMPT_PREDICATES:
        time_anchor = claim.get("time_anchor", {})
        if time_anchor.get("type") == "unknown":
            reason_codes.append("unknown_time_anchor")
    
    return len(reason_codes) == 0, reason_codes

# Exempt predicates (verifiable without explicit time)
TIME_ANCHOR_EXEMPT_PREDICATES = {"quote", "policy", "ranking", "existence"}
```

**Code location:** `spectrue_core/agents/skills/claims.py`

> ⚠️ **Epistemological Note:** These are **structural rules**, not truth heuristics.
> A claim passing validation is retrievable and scorable, not necessarily true.

---

## Research Grounding

### Directly Used Concepts

| Paper | Concept Used | Code Location |
|-------|--------------|---------------|
| [Pearl (1988)](https://www.sciencedirect.com/book/9780080514895/probabilistic-reasoning-in-intelligent-systems) | Belief Propagation on DAGs | `graph/propagation.py` |
| [ReAct](https://arxiv.org/abs/2210.03629) | Interleaved search-reason loop | `verification/phase_runner.py` |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | Adaptive retrieval stopping | `verification/pipeline_evidence.py` |

### Datasets for Terminology

| Dataset | What We Borrow |
|---------|----------------|
| [FEVER](https://arxiv.org/abs/1803.05355) | SUPPORTED/REFUTED/NEI verdict labels |
| [HoVer](https://arxiv.org/abs/2011.03088) | Multi-hop claim dependency concept |

> **Note:** We do not train on these datasets. We borrow terminology and problem framing.

---

## What Spectrue Is — and Is Not

### What Spectrue IS:

1. **Bayesian-inspired scoring:** Log-odds representation and additive updates follow 
   standard Bayesian inference math.

2. **Belief propagation on claim graphs:** Single-pass BP for DAGs is mathematically 
   sound and follows Pearl's framework.

3. **Engineering system with heuristics:** Practical decisions (sigmoid damping, 
   source priors when enabled, consensus caps) that work empirically but lack theoretical guarantees.

### What Spectrue is NOT:

1. **NOT a true posterior probability:** The final `verified_score` is NOT a calibrated 
   probability that the article is true. It is a credibility score influenced by 
   heuristics and constraints.

2. **NOT modeling full scientific consensus:** The consensus cap is a safeguard, not 
   a proper hierarchical Bayesian model with consensus as hyperprior.

3. **NOT theoretically optimal:** Sigmoid parameters, tier priors, and other 
   engineering choices are empirically tuned, not derived from optimization.

4. **NOT training on fact-checking datasets:** We borrow terminology from FEVER/HoVer 
   but do not train models on these datasets.

---

## Deprecated Algorithm

### TierDominantAggregation

**Status:** DEPRECATED

Replaced by Bayesian Scoring. See `spectrue_core.scoring.belief`.

---

## Test Coverage

- **Unit tests:** `tests/unit/scoring/` — belief math, priors, impact functions
- **Integration tests:** `tests/integration/scoring/` — propagation, consensus bounding

## Tier as a Prior Factor on Explainability (A)

**Tier never clamps truth.** In Spectrue Engine, source tiers are used to adjust **Explainability (A)** only.
They do **not** cap `verdict_score` or `verified_score`.

We treat tier as a calibrated *prior reliability signal* for how interpretable/defensible the explanation is, given the
class of sources involved.

Deterministic rule (per-claim):

- `baseline = TierPrior["B"]`
- `prior = TierPrior[best_tier or "UNKNOWN"]`
- `factor = prior / baseline`
- `A_post = clamp(A_pre * factor, 0..1)`

Trace event:

- `verdict.explainability_tier_factor` logs `pre_A`, `prior`, `baseline`, `factor`, `post_A`, `best_tier`, `claim_id`, and `source`.

## Verification Target Selection (Bayesian EVOI Model)

When a run contains many claims, the engine selects the optimal number of claims to run retrieval against using a **Bayesian Expected Value of Information (EVOI)** model. Non-target claims are **deferred** and inherit verdicts via evidence sharing.

### EVOI-Based Target Count

Instead of hard-coded limits per budget class, the engine computes:

```
k* = argmax_k [ Σ(i=1..k) EVOI_i × decay^i − cost × k ]
```

Where:
- `EVOI_i = value_uncertainty × entropy(prior) × worthiness_i × harm_i × conf_factor_i`
- `entropy(prior)` = Shannon entropy of Bernoulli(prior), max at p=0.5 (maximum uncertainty)
- `decay` = diminishing returns factor (0.85 by default), models that later targets have less marginal value
- `cost` = marginal cost per target (normalized Tavily search cost)

### Claim-Level EVOI Signals

Each claim's Expected Value of Information depends on:

| Signal | Effect |
|--------|--------|
| `check_worthiness` | Higher worthiness → higher EVOI |
| `harm_potential` | Higher harm → higher EVOI (prioritize verifying dangerous claims) |
| `metadata_confidence` | Lower confidence → higher EVOI (uncertain claims benefit more from search) |
| Prior uncertainty | Max EVOI at p=0.5, decreases as certainty increases |

### Budget Class Constraints

Budget classes provide **safety bounds** (floors and ceilings) on top of the Bayesian computation.
These map to the `BudgetClass` enum in `execution_plan.py`:

| Budget Class | Floor | Ceiling | Use Case |
|--------------|-------|---------|----------|
| `minimal`    | 1     | 3       | Low-priority claims, fastest |
| `standard`   | 2     | 5       | Balanced cost/coverage (default) |
| `deep`       | 3     | 30      | High-priority claims, max coverage |

The model stops adding targets when:
1. Marginal EVOI falls below minimum threshold (0.05)
2. Marginal EVOI < marginal cost
3. Ceiling reached

### Trace Events

- `target_selection.bayesian_budget`: logs optimal_k, marginal analysis, params
- `target_selection.completed`: logs chosen targets, deferred claims, and reason codes
- `target_selection.anchor_forced`: logs when anchor claim is force-promoted to targets

### Design Rationale

This is **resource-driven orchestration**, not a truth heuristic: selection only decides *where to spend retrieval budget*.
All selected claims still require evidence to score high. The Bayesian approach naturally:
- Selects more targets for high-harm/high-uncertainty claims
- Selects fewer targets when marginal value drops below cost
- Respects budget constraints as safety nets
