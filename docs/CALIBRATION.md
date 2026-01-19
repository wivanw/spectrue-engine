# Posterior Calibration (alpha, beta)

This document describes how to fit the alpha/beta parameters used by the
claim posterior model:

    l_post = l_prior + alpha * l_llm + beta * l_evidence
    l_final = l_post + W_prior * l_internal_prior

The script performs MAP estimation with Gaussian priors on alpha, beta, and W_prior.

## Dataset format

The calibration script accepts JSONL (one object per line) or a JSON array.
Each record must include a `label` and either precomputed log-odds or raw
signals.

### Option A: Precomputed log-odds

Required fields:
- `label`: 0/1 (or a float in [0, 1])
- `log_odds_prior`
- `log_odds_llm`
- `log_odds_evidence`

Example JSONL:

```json
{"label": 1, "log_odds_prior": 0.405, "log_odds_llm": 1.386, "log_odds_evidence": 2.1}
{"label": 0, "log_odds_prior": -0.405, "log_odds_llm": -0.847, "log_odds_evidence": -1.4}
```

### Option B: Raw signals

Required fields:
- `label`: 0/1 (or a float in [0, 1])
- `llm_verdict_score` (or `p_llm`)
- `best_tier` (or `tier`, optional)
- `evidence_items`: list of evidence objects

Evidence item fields:
- `stance`: support/refute/neutral/context
- `relevance`: 0..1 (defaults to 0.5 if missing)
- `quote_present` (or `quote` / `has_quote`): boolean
- `tier`: optional per-source tier

Example JSONL:

```json
{"label": 1, "llm_verdict_score": 0.82, "best_tier": "B", "evidence_items": [{"stance": "support", "relevance": 0.9, "quote_present": true}]}
{"label": 0, "llm_verdict_score": 0.25, "best_tier": "D", "evidence_items": [{"stance": "refute", "relevance": 0.7, "quote_present": true}]}
```

## Run the calibration script

```bash
python examples/calibrate_claim_posterior.py --input path/to/claims.jsonl
```

Common overrides:

```bash
python examples/calibrate_claim_posterior.py \
  --input path/to/claims.jsonl \
  --prior-alpha-mean 1.0 \
  --prior-beta-mean 1.0 \
  --prior-alpha-std 0.5 \
  --prior-beta-std 0.5 \
  --lr 0.05 \
  --max-steps 1000
```

## Applying the results

Update the fitted values where posterior parameters are configured for
search policy profiles (for example, `SearchPolicyProfile.posterior_alpha`
and `SearchPolicyProfile.posterior_beta`).

If you maintain profile YAMLs or runtime configuration that map into search
policy profiles, apply the same values there to keep behavior consistent.
