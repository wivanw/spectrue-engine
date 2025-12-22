# Algorithms (Spectrue Engine)

This document describes the core retrieval and verification algorithms used by the engine.

## SearchPolicy (Main vs Deep)
- **main**: precision-first, strict hop cap.
- **deep**: recall-first, bounded multi-hop loop.
- Policy parameters: `search_depth`, `max_results`, `max_hops`, `channels_allowed`,
  `use_policy_by_channel`, `locale_policy`, `quality_thresholds`, `stop_conditions`.

## Retrieval Loop (Bounded)
- Each hop is policy-shaped (depth, k, channels, locale).
- Follow-ups are generated from evidence snippets, not the original article.
- Stops on ENOUGH, STOP, hop limit, or budget ceiling.

## Evidence Acquisition Ladder (EAL)
- If evidence is snippet-only or all-context, escalate:
  1) Fetch raw content for top candidates.
  2) Extract lightweight quote candidates.
  3) Re-evaluate sufficiency.

## Algorithmic appendix (pseudocode)

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

```text
RetrievalLoop(claim, phase_list, policy):
  hops = []
  for hop in range(policy.max_hops):
    if budget_exceeded(): return STOP
    query = next_query_or_claim_query(claim)
    sources = search(query, phase_list[hop])
    sources = evidence_acquisition_ladder(sources)
    decision = SufficiencyJudge(claim, sources, policy)
    record_hop(hops, query, decision)
    if decision in {ENOUGH, STOP}: break
    query = followup_from_snippets(sources)
    if not query: return STOP
  return hops
```

```text
SufficiencyJudge(claim, sources, policy):
  result = rule_based_sufficiency(claim, sources)
  if result == SUFFICIENT and policy.stop_on_sufficiency: return ENOUGH
  if below_quality_thresholds(sources, policy): return NEED_FOLLOWUP
  if result == INSUFFICIENT: return NEED_FOLLOWUP
  if result == SKIP: return STOP
```

```text
TierDominantAggregation(sources):
  tier = highest_tier_present(sources)
  strength = max_relevance_for_tier(sources, tier)
  score = tier_base(tier) + tier_gain(tier) * strength
  score -= conflict_penalty(sources)
  return clamp(score, 0, tier_ceiling(tier))
```

```text
TemporalFiltering(claim_time, sources):
  for src in sources:
    if src.date outside claim_time.window: mark temporal_mismatch
  if too_many_mismatches: apply temporal_penalty
  return sources
```

```text
LocaleRouting(phase_id, locale_policy, claim_locale):
  if phase_id == "C" and locale_policy.fallback: return locale_policy.fallback[0]
  if locale_policy.primary: return locale_policy.primary
  return claim_locale
```

## Research grounding
- ReAct: https://arxiv.org/abs/2210.03629
- Self-RAG: https://arxiv.org/abs/2310.11511
- FEVER: https://arxiv.org/abs/1803.05355
- HotpotQA: https://arxiv.org/abs/1809.09600
- HoVer: https://arxiv.org/abs/2011.03088
- SciFact: https://arxiv.org/abs/2004.14974

## Test coverage note
- Unit tests: M92 retrieval loop coverage is in place.
- Integration tests: pending.
