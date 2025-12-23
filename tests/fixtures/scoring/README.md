# Scoring Fixtures

These fixtures define evidence packs for deterministic scoring tests.

## Schema

Each JSON file follows this shape:

```json
{
  "claim_id": "c1",
  "claim": {
    "id": "c1",
    "text": "...",
    "verification_target": "reality",
    "type": "atomic",
    "role": "core",
    "check_worthiness": 0.8,
    "metadata_confidence": "high"
  },
  "evidence_pack": {
    "claim_id": "c1",
    "items": [
      {
        "url": "https://example.org",
        "domain": "example.org",
        "channel": "authoritative",
        "tier": "A",
        "stance": "REFUTE",
        "quote": "...",
        "relevance": 0.9,
        "temporal_flag": "in_window"
      }
    ],
    "stats": {
      "domain_diversity": 1,
      "tiers_present": {"A": 1},
      "support_count": 0,
      "refute_count": 1,
      "context_count": 0,
      "outdated_ratio": 0.0
    }
  }
}
```

## Scenarios

- `tierA_refute_vs_many_low_support.json`
- `strong_conflict_same_tier.json`
- `context_only_should_not_verify.json`
- `temporal_mismatch.json`
