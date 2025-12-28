from spectrue_core.analysis.content_budgeter import ContentBudgeter, ContentBudgetConfig


def test_small_plain_text_passes_through():
    text = "Short note about content budgeting.\nStill concise."
    cfg = ContentBudgetConfig()
    budgeter = ContentBudgeter(cfg)

    result = budgeter.trim(text)

    assert result.trimmed_text == text
    assert result.selection_meta == []
    assert result.trimmed_len == result.raw_len
