# Algorithms (Spectrue Engine)

This document describes the core retrieval, verification, and scoring algorithms used by the engine.

---

## M104: Bayesian Scoring (Current)

The engine uses Bayesian inference for credibility scoring. This replaces arithmetic averaging with probabilistic belief updates.

### Core Principle

**Belief is represented in log-odds space** where Bayesian updates become additive:

```
Posterior(log-odds) = Prior(log-odds) + Σ Evidence(log-odds)
```

This is mathematically grounded in **Pearl (1988)**, where log-odds transforms convert multiplicative Bayes rule into additive updates, enabling efficient message passing on graphical models.

This ensures:
- Strong evidence has bounded impact (via sigmoid saturation)
- Weak/noisy evidence doesn't collapse credibility
- Priors from source tiers influence final scores

### BeliefState

```python
class BeliefState:
    log_odds: float  # Log-odds of truth: log(P / (1 - P))
    confidence: float  # Certainty (0-1)
    
    @property
    def probability(self) -> float:
        return 1 / (1 + exp(-log_odds))  # Logistic function
```

### Probability ↔ Log-Odds Conversion

```
log_odds = log(p / (1 - p))
probability = 1 / (1 + exp(-log_odds))
```

> **Grounding:** This is the standard logit/logistic transform used in Bayesian inference.
> See: Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*, Ch. 2.

### Tier-Based Priors (FR-002)

Source tier determines the starting belief (prior):

| Tier | Veracity Prior | Danger Prior |
|------|----------------|--------------|
| A (Official) | 0.85 | 0.10 |
| A' (Official Social) | 0.75 | 0.15 |
| B (Trusted Media) | 0.70 | 0.20 |
| C (Local Media) | 0.55 | 0.35 |
| D (Social) | 0.35 | 0.50 |

> ⚠️ **Note (Empirical Heuristic):** These prior values are **empirically tuned** based on
> observed accuracy rates of different source types. They are not derived from a specific
> academic paper. Future work should validate these against ground-truth datasets like
> **SciFact** (arXiv:2004.14974) for scientific claims or NewsGuard-style trust ratings.

### Sigmoid Impact Function (FR-005)

Evidence impact is non-linear to saturate weak claims:

```
impact = relevance × L_max × sigmoid(strength)

sigmoid(x) = 1 / (1 + exp(-k × (x - x₀)))
```

Where:
- `strength`: Evidence confidence (0-1)
- `relevance`: Semantic relevance (0-1)
- `k = 10.0`: Steepness (sharp transition at x₀)
- `x₀ = 0.5`: Midpoint
- `L_max = 2.0`: Maximum log-odds impact

This ensures:
- Strong claims (strength > 0.5): Near-maximum impact
- Weak claims (strength < 0.3): Minimal impact (saturation)

> ⚠️ **Note (Engineering Heuristic):** This sigmoid saturation is an **engineering decision**
> to prevent noise accumulation. It is inspired by logistic regression and soft attention
> mechanisms but is not directly derived from a specific fact-checking paper. The parameters
> (k=10, x₀=0.5, L_max=2.0) were tuned to satisfy SC-002 (noise tolerance) requirements.

### Consensus Constraint (FR-006)

Scientific consensus limits maximum posterior:

```
if consensus.source_count >= 2:
    posterior_log_odds = min(posterior_log_odds, prob_to_log_odds(consensus.score))
```

> ⚠️ **Note (Constraint Injection):** This is a **hard constraint**, not a Bayesian update.
> It implements the principle that individual article credibility cannot exceed the broader
> scientific consensus on the topic. This is an engineering safeguard rather than a
> probabilistic inference step. Future versions may replace this with a proper hierarchical
> Bayesian model where consensus acts as a hyperprior.

### RGBA Belief Dimensions (FR-007)

Each RGBA dimension is an **independent** probabilistic belief:

| Dimension | Prior Source | Update Source |
|-----------|--------------|---------------|
| R (Danger) | Inverse of tier | Harm-related evidence |
| G (Veracity) | Source tier | Factual claims |
| B (Honesty) | Neutral (0.5) | Misleading context detection |
| A (Explainability) | Neutral (0.5) | Quote/source availability |

> **Grounding:** Independence of dimensions follows the principle that factual accuracy (G)
> is orthogonal to presentation honesty (B). A true fact can be presented misleadingly.

### Claim Graph Propagation (FR-008)

Beliefs propagate through the ClaimContextGraph using **Belief Propagation** (Pearl, 1988):

```
For each node in topological order:
    message = Σ (source.propagated_belief × edge.weight × sign)
    node.propagated_belief = node.local_belief + message
    
sign = -1 if CONTRADICTS else +1
```

> **Grounding:** This is a simplified single-pass BP for DAGs. Pearl's original algorithm
> supports polytrees with forward/backward passes. Our simplification is valid because:
> 1. We enforce DAG structure (no cycles in claim dependencies)
> 2. Topological ordering ensures correct message flow
>
> The CONTRADICTS → negative sign follows **HoVer** (arXiv:2011.03088) semantics for
> multi-hop claim verification with supporting/contradicting evidence.

---

## Retrieval & Search Algorithms

### SearchPolicy (Main vs Deep)

- **main**: precision-first, strict hop cap.
- **deep**: recall-first, bounded multi-hop loop.
- Policy parameters: `search_depth`, `max_results`, `max_hops`, `channels_allowed`,
  `use_policy_by_channel`, `locale_policy`, `quality_thresholds`, `stop_conditions`.

### Retrieval Loop (Bounded)

- Each hop is policy-shaped (depth, k, channels, locale).
- Follow-ups are generated from evidence snippets, not the original article.
- Stops on ENOUGH, STOP, hop limit, or budget ceiling.

> **Grounding:** This implements the **ReAct** paradigm (arXiv:2210.03629) where reasoning
> and acting are interleaved. The "act" is search; the "reason" is sufficiency judgment.
> Iterative refinement based on previous results aligns with **Self-RAG** (arXiv:2310.11511)
> adaptive retrieval.

### Evidence Acquisition Ladder (EAL)

- If evidence is snippet-only or all-context, escalate:
  1) Fetch raw content for top candidates.
  2) Extract lightweight quote candidates.
  3) Re-evaluate sufficiency.

### SufficiencyJudge

The sufficiency check implements **Self-RAG** style reflection:
- `ISSUP` equivalent: Does evidence support the claim? → ENOUGH
- `ISREL` equivalent: Is evidence relevant? → Filter by threshold
- `ISUSE` equivalent: Is the quality sufficient? → NEED_FOLLOWUP if not

---

## Algorithmic Appendix (Pseudocode)

### Bayesian Scoring (M104)

```text
BayesianScoring(claims, sources, prior):
  current_belief = prior  # From source tier
  
  for claim in claims:
    verdict = score_claim(claim, sources)
    confidence = verdict.confidence
    direction = +1 if "verified" else -1 if "refuted" else 0
    
    impact = sigmoid_impact(confidence, relevance=1.0, direction)
    current_belief = update_belief(current_belief, impact)
  
  consensus = calculate_consensus(sources)
  current_belief = apply_consensus_constraint(current_belief, consensus)
  
  return current_belief.probability
```

### Claim Graph Propagation (Pearl-style BP)

```text
PropagateBeliefs(graph):
  # Single-pass BP for DAGs (valid due to topological ordering)
  for node in topological_sort(graph):
    local = node.local_belief or BeliefState(0.0)
    messages = 0.0
    
    for edge in incoming_edges(node):
      source = edge.source
      sign = -1 if edge.relation == CONTRADICTS else +1
      messages += source.propagated_belief × edge.weight × sign
    
    node.propagated_belief = local.log_odds + messages
```

### SearchPolicy

```text
SearchPolicy(profile, plan):
  phase_list = plan.phases
  phase_list = cap_depth(phase_list, profile.search_depth)
  phase_list = cap_results(phase_list, profile.max_results)
  phase_list = filter_channels(phase_list, profile.channels_allowed)
  phase_list = apply_locale_policy(phase_list, profile.locale_policy)
  phase_list = cap_hops(phase_list, profile.max_hops)
  return phase_list
```

### RetrievalLoop (ReAct-style)

```text
RetrievalLoop(claim, phase_list, policy):
  hops = []
  for hop in range(policy.max_hops):
    if budget_exceeded(): return STOP
    
    # ACT: Execute search
    query = next_query_or_claim_query(claim)
    sources = search(query, phase_list[hop])
    sources = evidence_acquisition_ladder(sources)
    
    # REASON: Evaluate sufficiency
    decision = SufficiencyJudge(claim, sources, policy)
    record_hop(hops, query, decision)
    
    if decision in {ENOUGH, STOP}: break
    
    # PLAN: Generate follow-up
    query = followup_from_snippets(sources)
    if not query: return STOP
    
  return hops
```

### SufficiencyJudge (Self-RAG style)

```text
SufficiencyJudge(claim, sources, policy):
  result = rule_based_sufficiency(claim, sources)
  
  # ISSUP-equivalent: sufficient evidence?
  if result == SUFFICIENT and policy.stop_on_sufficiency: 
    return ENOUGH
  
  # ISUSE-equivalent: quality threshold?
  if below_quality_thresholds(sources, policy): 
    return NEED_FOLLOWUP
  
  if result == INSUFFICIENT: return NEED_FOLLOWUP
  if result == SKIP: return STOP
```

### TemporalFiltering

```text
TemporalFiltering(claim_time, sources):
  for src in sources:
    if src.date outside claim_time.window: mark temporal_mismatch
  if too_many_mismatches: apply temporal_penalty
  return sources
```

### LocaleRouting

```text
LocaleRouting(phase_id, locale_policy, claim_locale):
  if phase_id == "C" and locale_policy.fallback: return locale_policy.fallback[0]
  if locale_policy.primary: return locale_policy.primary
  return claim_locale
```

---

## Deprecated: TierDominantAggregation

> ⚠️ **DEPRECATED (M104)**: This algorithm is replaced by Bayesian Scoring.
> See `spectrue_core.scoring.belief` for the new implementation.

```text
# LEGACY - Do not use
TierDominantAggregation(sources):
  tier = highest_tier_present(sources)
  strength = max_relevance_for_tier(sources, tier)
  score = tier_base(tier) + tier_gain(tier) * strength
  score -= conflict_penalty(sources)
  return clamp(score, 0, tier_ceiling(tier))
```

---

## Research Grounding

### Core Algorithms

| Algorithm | Paper | How We Use It |
|-----------|-------|---------------|
| **Belief Propagation** | Pearl (1988) | Log-odds updates, DAG message passing |
| **ReAct** | arXiv:2210.03629 | Interleaved reasoning + search |
| **Self-RAG** | arXiv:2310.11511 | Adaptive retrieval, reflection tokens |

### Fact Verification Datasets

| Dataset | Paper | Relevance |
|---------|-------|-----------|
| **FEVER** | arXiv:1803.05355 | SUPPORTED/REFUTED/NEI classification |
| **HotpotQA** | arXiv:1809.09600 | Multi-hop reasoning, explainability |
| **HoVer** | arXiv:2011.03088 | Multi-hop claim verification |
| **SciFact** | arXiv:2004.14974 | Scientific claim verification |

### Full References

1. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann.
2. Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.
3. Asai, A. et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv:2310.11511.
4. Thorne, J. et al. (2018). FEVER: A Large-scale Dataset for Fact Extraction and VERification. arXiv:1803.05355.
5. Yang, Z. et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. arXiv:1809.09600.
6. Jiang, Y. et al. (2020). HoVer: A Dataset for Many-Hop Fact Extraction and Claim Verification. arXiv:2011.03088.
7. Wadden, D. et al. (2020). Fact or Fiction: Verifying Scientific Claims. arXiv:2004.14974.

---

## Heuristics & Engineering Decisions

The following are **not derived from academic papers** but are engineering decisions:

| Component | Type | Rationale |
|-----------|------|-----------|
| Tier priors (0.85, 0.70, ...) | Empirical | Tuned for observed source accuracy |
| Sigmoid parameters (k=10, x₀=0.5) | Heuristic | Satisfy noise tolerance requirement |
| L_max = 2.0 | Heuristic | Bound single-evidence impact |
| Consensus bounding | Constraint | Hard cap, not Bayesian |

---

## Test Coverage Note

- **Unit tests**: M92 retrieval loop, M104 Bayesian scoring (18 tests)
- **Integration tests**: M104 belief propagation, consensus bounding, noise tolerance
