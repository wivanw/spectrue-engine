"""Unit tests for prompt caching (prefix cache) convention.

Prompts must follow: STATIC_PREFIX + DELIMITER + DYNAMIC_CONTENT
so that OpenAI can cache the prefix. Delimiter is typically \"--- DATA ---\" or \"--- ARTICLE ---\".
"""


from spectrue_core.adapters.llm.claims_prompts import (
    build_core_extraction_prompt,
    build_claim_strategist_prompt,
    build_retrieval_planning_prompt,
    build_claim_schema_prompt,
    build_skeleton_extraction_prompt,
)
from spectrue_core.adapters.llm.scoring_contract import (
    build_score_evidence_prompt,
    build_score_evidence_structured_prompt,
    build_single_claim_scoring_prompt,
    build_stance_matrix_prompt,
)
from spectrue_core.adapters.llm.claim_judge_prompts import (
    _build_claim_judge_data_block,
    build_claim_judge_prompt,
)
from spectrue_core.adapters.llm.evidence_summarizer_prompts import build_evidence_summarizer_prompt
from spectrue_core.adapters.llm.audit_prompts import (
    build_claim_audit_prompt,
    build_evidence_audit_prompt,
)
from spectrue_core.adapters.llm.clustering_contract import build_evidence_matrix_prompt
from spectrue_core.domain.claims.frame import (
    ClaimFrame,
    EvidenceItemFrame,
    ContextExcerpt,
    ContextMeta,
    EvidenceStats,
)


# Delimiters used in the codebase for cache-friendly prompts
CACHE_DELIMITERS = ("--- DATA ---", "--- ARTICLE ---", "--- INPUT ---")


def _assert_delimiter_then_dynamic(prompt: str, delimiter: str, dynamic_contains: str) -> None:
    """Assert prompt has delimiter and dynamic content after it."""
    assert delimiter in prompt, f"Prompt must contain delimiter {delimiter!r}"
    idx = prompt.index(delimiter)
    after = prompt[idx + len(delimiter) :].strip()
    assert after, "Content after delimiter must be non-empty"
    assert dynamic_contains in after, f"Dynamic content after delimiter must contain {dynamic_contains!r}"


class TestClaimsPromptsCacheFormat:
    """Claim extraction and related prompts."""

    def test_core_extraction_has_article_delimiter_and_dynamic(self):
        article = "The president announced a new policy in January 2025."
        prompt = build_core_extraction_prompt(text_excerpt=article)
        _assert_delimiter_then_dynamic(prompt, "--- ARTICLE ---", article)

    def test_claim_strategist_has_article_delimiter_and_dynamic(self):
        article = "Short excerpt for strategist."
        prompt = build_claim_strategist_prompt(article)
        _assert_delimiter_then_dynamic(prompt, "--- ARTICLE ---", article)

    def test_retrieval_planning_has_input_delimiter_and_dynamic(self):
        claim = "Tesla stock rose in Q4 2024."
        context = "Article context here."
        prompt = build_retrieval_planning_prompt(
            claim_text=claim, article_context_sm=context, lang_name="en"
        )
        _assert_delimiter_then_dynamic(prompt, "--- INPUT ---", claim)
        assert context in prompt

    def test_claim_schema_has_article_delimiter_and_dynamic(self):
        article = "Schema parse excerpt."
        prompt = build_claim_schema_prompt(text_excerpt=article)
        _assert_delimiter_then_dynamic(prompt, "--- ARTICLE ---", article)

    def test_skeleton_extraction_has_article_delimiter_and_dynamic(self):
        article = "Skeleton extraction text."
        prompt = build_skeleton_extraction_prompt(text_excerpt=article)
        _assert_delimiter_then_dynamic(prompt, "--- ARTICLE ---", article)


class TestScoringPromptsCacheFormat:
    """Scoring and stance prompts."""

    def test_score_evidence_prompt_has_data_delimiter_and_dynamic(self):
        prompt = build_score_evidence_prompt(
            safe_original_fact="Original fact.",
            claims_info=[{"id": "c1", "text": "Claim one."}],
            sources_by_claim={"c1": [{"url": "https://example.com", "snippet": "Snippet"}]},
        )
        _assert_delimiter_then_dynamic(prompt, "--- DATA ---", "original_fact")
        assert "Claim one" in prompt or "c1" in prompt

    def test_score_evidence_structured_prompt_has_data_delimiter(self):
        prompt = build_score_evidence_structured_prompt(
            claims_data=[{"claim_id": "c1"}],
            evidence_by_assertion={"c1": []},
        )
        _assert_delimiter_then_dynamic(prompt, "--- DATA ---", "claims_with_assertions")

    def test_single_claim_scoring_prompt_has_data_delimiter(self):
        prompt = build_single_claim_scoring_prompt(
            claim_info={"id": "c1", "text": "A claim."},
            evidence=[],
        )
        _assert_delimiter_then_dynamic(prompt, "--- DATA ---", "claim")

    def test_stance_matrix_prompt_has_data_delimiter(self):
        prompt = build_stance_matrix_prompt(
            claims_lite=[{"id": "c1"}],
            sources_lite=[{"url": "https://x.com"}],
        )
        _assert_delimiter_then_dynamic(prompt, "--- DATA ---", "claims")


class TestClaimJudgeCacheFormat:
    """Claim judge: static prefix + --- DATA --- + data block."""

    def test_claim_judge_data_block_contains_claim_id_and_sections(self):
        frame = ClaimFrame(
            claim_id="c1",
            claim_text="Test claim text.",
            claim_language="en",
            context_excerpt=ContextExcerpt(text="Context here.", span_start=0, span_end=13),
            context_meta=ContextMeta(document_id="doc1"),
            evidence_items=(),
            evidence_stats=EvidenceStats(),
        )
        block = _build_claim_judge_data_block(
            frame, "No evidence.", "No summary.", "Stats: 0", "  (none)"
        )
        assert "c1" in block
        assert "Test claim text" in block
        assert "## CLAIM TO JUDGE" in block
        assert "## ORIGINAL CONTEXT" in block
        assert "## EVIDENCE ITEMS" in block

    def test_claim_judge_prompt_contains_data_delimiter(self):
        # build_claim_judge_prompt needs a full frame and locale; we only check that
        # when we have static content from YAML, the assembled prompt has the delimiter.
        # If YAML is missing we get fallback which also uses --- DATA ---
        frame = ClaimFrame(
            claim_id="c1",
            claim_text="Claim.",
            claim_language="en",
            context_excerpt=ContextExcerpt(text="Ctx", span_start=0, span_end=3),
            context_meta=ContextMeta(document_id="doc1"),
            evidence_items=(),
            evidence_stats=EvidenceStats(),
        )
        prompt = build_claim_judge_prompt(frame, None, ui_locale="en")
        assert "--- DATA ---" in prompt
        assert "c1" in prompt


class TestEvidenceSummarizerCacheFormat:
    """Evidence summarizer prompt."""

    def test_evidence_summarizer_prompt_has_input_delimiter_and_dynamic(self):
        frame = ClaimFrame(
            claim_id="cs1",
            claim_text="Summarizer claim.",
            claim_language="en",
            context_excerpt=ContextExcerpt(text="Context.", span_start=0, span_end=8),
            context_meta=ContextMeta(document_id="doc1"),
            evidence_items=(),
            evidence_stats=EvidenceStats(),
        )
        prompt = build_evidence_summarizer_prompt(frame)
        _assert_delimiter_then_dynamic(prompt, "--- INPUT ---", "cs1")
        assert "Summarizer claim" in prompt


class TestAuditPromptsCacheFormat:
    """Audit prompts."""

    def test_claim_audit_prompt_has_data_delimiter(self):
        frame = ClaimFrame(
            claim_id="ca1",
            claim_text="Audit claim.",
            claim_language="en",
            context_excerpt=ContextExcerpt(text="Excerpt", span_start=0, span_end=7),
            context_meta=ContextMeta(document_id="doc1"),
            evidence_items=(),
            evidence_stats=EvidenceStats(total_sources=1, context_sources=1),
        )
        prompt = build_claim_audit_prompt(frame)
        _assert_delimiter_then_dynamic(prompt, "--- DATA ---", "ca1")

    def test_evidence_audit_prompt_has_data_delimiter(self):
        frame = ClaimFrame(
            claim_id="c1",
            claim_text="Claim.",
            claim_language="en",
            context_excerpt=ContextExcerpt(text="C", span_start=0, span_end=1),
            context_meta=ContextMeta(document_id="doc1"),
            evidence_items=(),
            evidence_stats=EvidenceStats(),
        )
        ev = EvidenceItemFrame(
            evidence_id="e1",
            claim_id="c1",
            source_id="s1",
            url="https://example.com",
            title="Title",
            snippet="Snippet",
            quote=None,
            source_tier=None,
            stance=None,
            relevance=None,
        )
        prompt = build_evidence_audit_prompt(frame, ev)
        _assert_delimiter_then_dynamic(prompt, "--- DATA ---", "e1")


class TestClusteringCacheFormat:
    """Clustering / evidence matrix prompt."""

    def test_evidence_matrix_prompt_has_data_delimiter(self):
        prompt = build_evidence_matrix_prompt(
            claims_lite=[{"id": "c1"}],
            sources_lite=[{"url": "https://a.com"}],
        )
        _assert_delimiter_then_dynamic(prompt, "--- DATA ---", "claims")
