# Trace Contract
## Required events, fields, and invariant coverage

This document defines required trace events and their minimum field payloads.
Any pipeline variant must emit these events with consistent schemas.

---

## 1) Claim extraction / coverage

### 1.1 `claim_extraction.guard.missing_instructions`
**Purpose:** Proves I1.
**Fields:**
- `task_name` (e.g., `claims.coverage.skeleton` / `claims.coverage.gapfill` / `rgba.audit.*`)
- `injected_default` (bool)
- `provider` (string)
- `instructions_chars_before` (int)

### 1.2 `claims.coverage.anchors`
**Purpose:** Proves I2 baseline.
**Fields:**
- `counts` `{time:int, number:int, quote:int}`
- `anchors` list of `{anchor_id, kind, preview}` (preview <= 40 chars)

### 1.3 `claims.coverage.gaps`
**Purpose:** Proves I2 + drives I3.
**Fields:**
- `missing_anchor_ids` list[str]
- `missing_counts` by kind
- `covered_count` (int)
- `skipped_count` (int)
- `total_count` (int)

### 1.4 `claims.coverage.gapfill`
**Purpose:** Proves I3 one-shot repair path.
**Fields:**
- `missing_before` list[str]
- `added_items_count` (int)
- `still_missing_after` list[str]
- `provider_used` (primary/fallback)

### 1.5 `claims.coverage.final`
**Purpose:** Confirms final coverage state.
**Fields:**
- `total_anchors` (int)
- `covered_anchors` (int)
- `skipped_anchors` (int)
- `missing_anchors` (int)
- `claims_emitted` (int)

### 1.6 `claims.coverage.incomplete_terminal`
**Purpose:** Proves I3 terminal stop before search.
**Fields:**
- `missing_anchor_ids` list[str]
- `missing_counts` by kind
- `message` (short)
- `stop_before_search` (bool, must be true)

### 1.7 `claims.context.anchored`
**Purpose:** Proves I4 context inheritance happened.
**Fields:**
- `claim_id`
- `added_context_entities` list[str]
- `source` enum: `document_pool|neighborhood_pool|none`

---

## 2) Retrieval planning / search

### 2.1 `search.query.variants`
**Purpose:** Proves I5 (non-empty variants) + supports debugging.
**Fields:**
- `claim_id`
- `variants` list of `{query_id, text, strategy}` (text <= 80 chars)
- `included_context_entities_count` (int)

### 2.2 `search.query.empty_blocked`
**Purpose:** Proves I5 enforcement.
**Fields:**
- `claim_id`
- `reason` (string)
- `stage` (e.g., `variant_build` / `select_query`)

### 2.3 `search.escalation`
**Purpose:** Proves I10 and supports cost analysis.
**Fields:**
- `claim_id`
- `pass_id` (`A|B|C|D`)
- `query_id` (`Q1|Q2|Q3`)
- `params` `{search_depth, max_results, include_domains_relaxed, include_domains_count}`
- `outcome` `{sources_count, best_relevance, usable_snippets_count}`
- `reason_codes` list[str]

### 2.4 `search.sanity`
**Purpose:** Proves I6.
**Fields:**
- `claim_id`
- `pass_id`
- `decision` `pass|off_topic`
- `anchor_terms_count` (int)
- `max_overlap_count` (int)
- `overlap_ratio` (float)
- `reasons` list[str]

### 2.5 `search.summary`
**Purpose:** Trace final retrieval outcome per claim.
**Fields:**
- `claim_id`
- `passes_executed` (int)
- `tavily_calls` (int)
- `domains_relaxed` (bool)
- `final_outcome` `{sources_count, best_relevance, usable_snippets_count}`
- `sanity_final` `pass|off_topic`

---

## 3) LLM audit

### 3.1 `rgba.audit.claim.done`
**Purpose:** Proves I7 (LLM audit emitted structured labels).
**Fields:**
- `claim_id`
- `predicate_type`
- `audit_confidence`
- `assertion_strength`
- `risk_facets_count` (int)
- `honesty_facets_count` (int)

### 3.2 `rgba.audit.evidence.done`
**Purpose:** Proves I7 and helps detect drift.
**Fields:**
- `claim_id`
- `evidence_count` (int)
- `stance_distribution` `{support:int, refute:int, unclear:int, unrelated:int}`
- `avg_extraction_confidence` (float)

---

## 4) Independence / aggregation

### 4.1 `rgba.cluster.done`
**Purpose:** Proves independence handling exists.
**Fields:**
- `claim_id`
- `tau` (float)
- `cluster_count` (int)
- `max_cluster_size` (int)

### 4.2 `rgba.aggregate.claim`
**Purpose:** Proves I8 (statuses) + contribution trace.
**Fields:**
- `claim_id`
- `G_status` (enum)
- `B_status` (enum)
- `A_status` (enum)
- `support_mass` (float)
- `refute_mass` (float)
- `conflict` (float)
- `top_contributors` list of `{source_id, cluster_id, delta}` (bounded length)

### 4.3 `rgba.aggregate.run`
**Purpose:** Final summary.
**Fields:**
- `R_status`, `G_status`, `B_status`, `A_status`
- `global_reasons` list[str]
- `claims_count` (int)
- `evidence_count_total` (int)

---

## 5) Provider fallback

### 5.1 `llm.call.failed`
**Purpose:** Confirms failure detection.
**Fields:**
- `task_name`
- `provider`
- `failure_kind` (enum)
- `attempt` (int)

### 5.2 `llm.fallback.used`
**Purpose:** Confirms fallback happened only on failure.
**Fields:**
- `task_name`
- `primary_provider`
- `fallback_provider`
- `failure_kind`

---

## Compliance
A pipeline run is "trace-compliant" if it emits all required events for the executed path.
Missing required events indicates a contract breach and must be treated as a bug.
