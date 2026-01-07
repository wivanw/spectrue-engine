# Changelog

All notable changes to Spectrue Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
- M127: Coverage Skeleton Extraction
  - `coverage_skeleton.py` with dataclasses for skeleton items (event, measurement, quote, policy)
  - Regex-based coverage analyzers (time mentions, number mentions, quote spans)
  - `skeleton_to_claims()` converter with `skeleton_item_id` traceability
  - Coverage validation with tolerance threshold
  - Trace events: `claims.skeleton.created`, `claims.coverage.warning`, `claims.skeleton.to_claims`
  - `COVERAGE_SKELETON_SCHEMA` for Phase-1 LLM extraction
- M126: Search Escalation Policy
  - Multi-pass Tavily search with query variants (Q1/Q2/Q3: anchor_tight, anchor_medium, broad)
  - 4-pass escalation ladder (A→B→C→D) with observable stop conditions
  - Topic selection from structured claim fields (`falsifiable_by`, `time_anchor`)
  - Trace events: `search.topic.selected`, `search.pass.executed`, `search.escalation.summary`
- M125: Claim Verifiability Contract
  - `VERIFIABLE_CORE_CLAIM_SCHEMA` with required fields for search/match/scoring
  - `validate_core_claim()` deterministic validation (no numeric caps, no text heuristics)
  - Extraction stats: `claims_extracted_total`, `claims_dropped_nonverifiable`, `claims_emitted`
  - Time anchor exemptions for quote/policy/ranking predicates
- M124: Search Policy Cost Control
  - Per-claim source category policy enforcement
  - String-to-channel mapping helper with safe fallback
  - Per-claim trace summaries with LLM/search usage
  - Retrieval-planning vs post-evidence enrichment split
- M123: Retrieval Flow Modularization
  - Decomposed retrieval steps: BuildQueriesStep, WebSearchStep, RerankStep, FetchChunksStep
  - Collect-only evidence packaging (no hidden LLM/clustering)
  - StanceAnnotateStep, ClusterEvidenceStep as optional steps
  - Per-claim judging contracts with explicit error payloads
- M122: DAG Mode Separation
  - Standard vs Deep graph separation in pipeline factory
  - EvidenceCollectStep refactor (no judging during collection)
  - JudgeStandardStep with `standard.article_judged` trace
  - Deep mode per-claim results with no fallback RGBA
  - Metering wired to LLM and search clients
- M121: DAG Graph Separation
  - Pipeline contracts: InputDoc, Claims, EvidenceIndex, Judgments
  - EvidenceCollectStep, JudgeStandardStep, JudgeClaimsStep
  - Standard and Deep response contracts
  - Graph-level step validation and metering enforcement
- M120: DAG Architecture Restructure
  - Deterministic topological ordering with execution layers
  - Step execution state tracking (timestamps, errors)
  - Orphan-node detection and cycle error messages
  - DAG execution summary trace events
- M119: Core Logic Modularization
  - Extracted `bayesian_update.py`, `evidence_scoring.py`, `evidence_explainability.py`, `evidence_stance.py`
  - `run_evidence_flow()` with explicit `score_mode` parameter
  - Mode-agnostic steps (`enable_global_scoring`, `process_all_claims`)
  - All mode logic centralized in `factory.py`
- M112: Learned Scoring Calibration
  - `CalibrationRegistry` and `CalibrationModel` for calibrated scoring models
  - Logistic scoring for search relevance and evidence likeness
  - Aggregation policy weights (conflict, temporal, diversity penalties)
  - Feature flags: `SPECTRUE_CALIBRATION_ENABLED` and per-model toggles
- M105: Pipeline Evidence & Billing Fixes
  - Fixed "100% CONTEXT" degradation in Evidence Matrix (added `search_query` to LLM context)
  - Fixed Frontend Cost Display showing "0 credits" (now reads `total_credits` from engine cost summary)
  - Added comprehensive metering for Tavily API calls (search + extracts)
  - Added `cost_summary.attached` trace event for billing observability
  - Fixed ClaimGraph role mapping (thesis->core, background->support)

- M89: Open Source Readiness
  - CI workflow runs an explicit allowlist of core tests (unit + key integrations)
- M80: Claim-Centric Orchestration
  - `ClaimMetadata` schema with verification_target, claim_role, retrieval_policy
  - `ClaimOrchestrator` for building ExecutionPlans
  - `PhaseRunner` for progressive widening with early exit
  - `evidence_sufficiency()` function with 3 sufficiency rules
  - `rgba_aggregation` module for weighted RGBA scoring
  - Feature flag: `FEATURE_CLAIM_ORCHESTRATION`
  - 59 new tests (orchestrator, sufficiency, integration)

### Changed
- **Removed Tier A' (Social Inline Verification)**: LLM cannot reliably verify social account identity without API access
- Docs: Updated module trees (ClaimGraph split + schema serialization helper)
- Horoscopes and predictions now get `verification_target=none` and skip search
- RGBA aggregation uses weighted average (context claims weight=0)
- Improved fail-soft behavior in search pipeline
- Docs: Clarified resource accounting and finalization semantics
  - New document: `docs/RESOURCE_ACCOUNTING.md`
  - Updated ARCHITECTURE.md with cost-aware execution principles
  - Updated README.md design philosophy section

## [0.9.0] - 2024-12-21

### Added
- M78: Satire Awareness
  - `claim_category` classification (FACTUAL/SATIRE/OPINION)
  - `satire_likelihood` routing
  - `SATIRICAL` verdict status
  - EvidencePack split into `scored_sources` and `context_sources`

### Added
- M77: Epistemic Integrity
  - Context isolation in query generation
  - Soft fallback for evidence matrix
  - Salience-driven claim extraction (harm_potential)

### Added
- M76: Semantic Resilience
  - Search Fallback Ladder (news → general)
  - Impact-First claim extraction
  - Markdown-aware text cleaning

## [0.8.0] - 2024-12-21

### Added
- M75: Trace Safe Payloads
  - Head-only truncation for sensitive strings
  - PII redaction in logs
  - SHA256 content hashing for traceability

### Added
- M74: Coverage & Safety Fixes
  - `CoverageSampler` for large articles
  - Claim anchors for source location tracking
  - Topic-aware ClaimGraph
  - Semantic Gating v2 (EXACT/TOPIC/UNRELATED)
  - Medical claim sanitization

## [0.7.0] - 2024-12-21

### Added
- M73: Claim Intelligence Layer
  - Structural prioritization (layer 2)
  - Tension signal detection (layer 3)
  - Evidence-need routing (layer 4)
  - `EvidenceNeedType` enum

## [0.6.0] - 2024-12-21

### Added
- M72: Hybrid ClaimGraph (B+C)
  - Embedding-based candidate generation (B-stage)
  - LLM edge typing (C-stage)
  - PageRank-based claim ranking
  - Quality gates and budget limits

## [0.5.0] - 2024-12-20

### Added
- M70: Schema-First Pipeline
  - `ClaimUnit` with strict Dimension attributes
  - Assertion-level verification
  - `StructuredVerdict` response schema

### Added
- M69: Native LLM Scoring
  - LLM-based aggregation with importance weighting
  - Quote highlighting markers
  - Sentinel values (-1.0) for missing scores

## [0.4.0] - 2024-12-19

### Added
- M68: Evidence Matrix Clustering
  - Strict 1:1 source-to-claim mapping
  - SHA256 cache keys
  - Telemetry for dropped sources

### Added
- M66: Smart Routing & Semantic Gating
  - LLM-driven search method selection
  - Semantic relevance filter

## [0.3.0] - 2024-12-19

### Added
- M64: Topic-Aware Round-Robin
  - 3-pass query selection (coverage, depth, fill)
  - Fuzzy deduplication
  - Gambling safety guards

### Added
- M63: Oracle Hybrid Mode
  - JACKPOT/EVIDENCE/MISS scenarios
  - Smart validator integration

## [0.2.0] - 2024-12-10

### Changed
- Migrated to standalone open-source package
- AGPL v3 license

## [0.1.0] - 2024-11-26

### Added
- Initial release
- Multi-agent verification architecture
- Waterfall search strategy
- RGBA scoring system
- Tavily integration
- Google Fact Check API integration
