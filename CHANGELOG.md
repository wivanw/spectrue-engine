# Changelog

All notable changes to Spectrue Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
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
