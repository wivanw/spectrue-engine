from unittest.mock import MagicMock
from spectrue_core.adapters.llm.article_cleaner import ArticleCleanerSkill

def test_evidence_cleaner_boilerplate_stripping():
    # Mock LLMClient so it doesn't fail on AsyncOpenAI init missing key
    dummy_llm = MagicMock()
    cleaner = ArticleCleanerSkill(llm_client=dummy_llm)
    raw = "Share this article! Subscribe to our newsletter.\nActual substantive evidence that matters. Quote goes here."
    cleaned, metadata = cleaner.clean_evidence_item(raw)
    
    assert "Share this" not in cleaned
    assert "Subscribe to our newsletter" not in cleaned
    assert "Actual substantive evidence" in cleaned
    assert not metadata["is_boilerplate"]
    assert metadata["retention_ratio"] > 0.4

def test_evidence_cleaner_insufficient_retention():
    dummy_llm = MagicMock()
    cleaner = ArticleCleanerSkill(llm_client=dummy_llm)
    raw = "Read more at our website. Subscribe below."
    cleaned, metadata = cleaner.clean_evidence_item(raw)
    
    assert metadata["is_boilerplate"] is True
    assert metadata["retention_ratio"] < 0.5
