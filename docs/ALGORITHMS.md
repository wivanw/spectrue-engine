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

### Retrieval Loop [A-inspired]

**Status:** Academically Inspired (ReAct Pattern)

The retrieval loop interleaves search and reasoning:

```
for hop in range(max_hops):
    sources = search(query)
    decision = SufficiencyJudge(claim, sources)
    if decision == ENOUGH: break
    query = generate_followup(sources)
```

**Code location:** `spectrue_core/verification/phase_runner.py`

**Inspiration:** [ReAct](https://arxiv.org/abs/2210.03629) paradigm — "think, act, observe" loop.

> **Note:** We use the ReAct *pattern* but not its specific implementation details.
> Our sufficiency judgment is rule-based, not LLM-generated reflection tokens.

---

### SufficiencyJudge [A-inspired]

**Status:** Academically Inspired (Self-RAG Adaptive Retrieval)

Determines when to stop searching:

- **ENOUGH:** Sufficient supporting evidence found
- **NEED_FOLLOWUP:** Quality below threshold, continue searching
- **STOP:** Max hops reached or budget exhausted

**Inspiration:** [Self-RAG](https://arxiv.org/abs/2310.11511) adaptive retrieval tokens.

> **Note:** We implement *analogous* logic but with deterministic rules, not LLM tokens.

---

### Evidence Acquisition Ladder (EAL) [B]

**Status:** Engineering Heuristic

Escalation strategy when evidence is insufficient:
1. Start with snippets
2. Fetch full content for top candidates
3. Extract quotes
4. Re-evaluate sufficiency

**Code location:** `spectrue_core/verification/pipeline_evidence.py`

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

**Status:** DEPRECATED (M104)

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

## Verification Target Selection (Current)

When a run contains many claims, the engine selects up to `max_targets` claims to run retrieval against.
Non-target claims are **deferred** and can inherit verdicts via evidence sharing.

Current selection score (high-level):

- `check_worthiness` (0..1) and `importance` (0..1) from `ClaimMetadata`
- optional graph signals (key-claim membership, structural importance, tension)
- claim role/type boost (`thesis/core` is prioritized)

This is **resource-driven orchestration**, not a truth heuristic: selection only decides *where to spend retrieval budget*.
All selected claims still require evidence to score high.

Trace event:

- `target_selection.completed` logs chosen targets, deferred claims, and reason codes.

## Roadmap Note

The target selection logic is expected to evolve toward a **Bayesian value-of-information ranking** under a fixed budget:
maximize expected uncertainty reduction per cost. This will replace hard boosts over time, while keeping determinism and
traceability.
