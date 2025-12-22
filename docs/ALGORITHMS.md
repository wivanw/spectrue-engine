# Algorithms

## Retrieval Control Surface (Main vs Deep)

The retrieval system uses a policy-driven control surface to align search cost and quality with the verification goal.

### SearchPolicy
- **Profiles**: `main` (precision/cost-focused) and `deep` (recall/coverage-focused)
- **Parameters**: `search_depth`, `max_results`, `max_hops`, `channels_allowed`, `use_policy_by_channel`, `locale_policy`, and `quality_thresholds`
- **Stop conditions**: early stop on sufficiency and hop budget caps

### Application
- Policy is applied once during orchestration in `spectrue_core/verification/pipeline_search.py`.
- The adapter (`spectrue_core/verification/search_policy_adapter.py`) converts a profile into phase parameters and caps.
- Trace event: `search.policy.applied` emits the profile name, caps, channels, and thresholds for debugging.

### Expected Behavior
- The same claim in `main` vs `deep` yields different phase counts, depths, and result limits.
- No branching on profile names occurs outside the policy adapter and selection step.
