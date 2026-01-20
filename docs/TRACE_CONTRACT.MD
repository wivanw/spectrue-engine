# Trace Contract
## Emitted events and minimum fields (current)

This document lists trace events that are **currently emitted** by the engine and their minimum fields.
If an event is not listed here, treat it as optional or debug-only.

---

## 1) Claim extraction / coverage

### 1.1 `claim_extraction.guard.instructions_injected`
**Purpose:** Ensure extraction instructions are always present.
**Fields:**
- `instructions_len` (int)
- `fallback_used` (bool)

### 1.2 `claim_extraction.anchors`
**Purpose:** Record deterministic anchors detected for extraction.
**Fields:**
- `anchor_count` (int)
- `anchor_ids` (list[str])

### 1.3 `claims.coverage.anchors`
**Purpose:** Record anchor counts by type.
**Fields:**
- `counts` `{time:int, number:int, quote:int, total:int}`
- `anchors` list of `{anchor_id, kind, preview}`

### 1.4 `claims.coverage.gaps`
**Purpose:** Record missing anchors after skeleton extraction.
**Fields:**
- `missing_anchor_ids` list[str]
- `missing_count` (int)
- `missing_by_kind` (dict)

### 1.5 `claims.coverage.gapfill.incomplete`
**Purpose:** Record remaining gaps after a gap-fill attempt.
**Fields:**
- `still_missing` list[str]
- `count` (int)

### 1.6 `claims.skeleton.created`
**Purpose:** Coverage skeleton stats.
**Fields:**
- `events`, `measurements`, `quotes`, `policies` (counts)

### 1.7 `claims.skeleton.to_claims`
**Purpose:** Skeleton-to-claims conversion stats.
**Fields:**
- `skeleton_count` (int)
- `claims_emitted` (int)
- `claims_dropped` (int)

### 1.8 `claim_extraction.stats`
**Purpose:** Aggregated extraction stats.
**Fields:**
- `claims_extracted_total` (int)
- `claims_dropped_nonverifiable` (int)
- `claims_emitted_targets` (int)
- `drop_reason_counts` (dict)

### 1.9 `claim_extraction.coverage_summary`
**Purpose:** Final extraction coverage ratio.
**Fields:**
- `total_anchors` (int)
- `claims_extracted` (int)
- `coverage_ratio` (float)

### 1.10 `claims.context.anchored`
**Purpose:** Context inheritance applied to a claim.
**Fields:**
- `claim_id`
- `added_context_entities` list[str]
- `source` enum: `document_pool|neighborhood_pool|none`

---

## 2) Retrieval planning / search

### 2.1 `search.query.variants`
**Purpose:** Query variants prepared for a claim.
**Fields:**
- `claim_id`
- `variants` list of `{query_id, text, strategy}`
- `count` (int)
- `included_context_entities_count` (int)

### 2.2 `search.query.empty_blocked`
**Purpose:** Empty query blocked.
**Fields:**
- `claim_id`
- `reason` (string)
- `stage` (string)

### 2.3 `search.escalation`
**Purpose:** Search escalation pass details.
**Fields:**
- `claim_id`
- `pass_id` (`A|B|C|D`)
- `query_id` (`Q1|Q2|Q3`)
- `reason_codes` list[str]
- `params` `{search_depth, max_results, include_domains_relaxed, include_domains_count}`
- `outcome` `{sources_count, best_relevance, usable_snippets_count, ...}`

### 2.4 `search.sanity`
**Purpose:** Off-topic detection result.
**Fields:**
- `claim_id`
- `pass_id`
- `decision` `pass|off_topic`
- `anchor_terms_count` (int)
- `max_overlap_count` (int)
- `overlap_ratio` (float)
- `reasons` list[str]

### 2.5 `search.summary`
**Purpose:** End-of-retrieval summary.
**Fields:**
- `claim_id`
- `passes_executed` (int)
- `tavily_calls` (int)
- `best_relevance_final` (float)
- `usable_snippets_final` (int)
- `domains_relaxed` (bool)

### 2.6 `target_selection.completed`
**Purpose:** Target selection summary for retrieval.
**Fields:**
- `total_claims` (int)
- `targets_count` (int)
- `deferred_count` (int)
- `evidence_sharing_count` (int)
- `max_targets` (int)
- `budget_class` (string)
- `anchor_claim_id` (string|null)
- `anchor_forced` (bool)

---

## 3) LLM audit (RGBA audit pipeline)

### 3.1 `claim_audit.complete`
**Purpose:** Claim-level audit completion.
**Fields:**
- `count` (int)
- `error_count` (int)

### 3.2 `evidence_audit.complete`
**Purpose:** Evidence-level audit completion.
**Fields:**
- `count` (int)
- `error_count` (int)

---

## 4) Provider fallback

### 4.1 `llm.call.failed`
**Purpose:** Failure detection and classification.
**Fields:**
- `task` (string)
- `provider` (string)
- `failure_kind` (enum)
- `error_type` (string)
- `error_message` (string)

### 4.2 `llm.fallback.used`
**Purpose:** Fallback invocation.
**Fields:**
- `task` (string)
- `primary_provider` (string)
- `fallback_provider` (string)
- `failure_kind` (string)

---

## Compliance
A run is trace-compliant if it emits the events relevant to its executed path.
Missing events indicate a contract drift and should be investigated.
