from spectrue_core.analysis.text_analyzer import TextAnalyzer


def test_parse_html_applies_budgeter():
    analyzer = TextAnalyzer()
    noisy = "<p>" + ("noise123 " * 8000) + "</p>"
    article = "<h1>Article</h1><p>" + ("meaningful sentence " * 120) + "</p>"
    html = f"<html><body>{noisy}{article}{noisy}</body></html>"

    parsed = analyzer.parse_html(html, language="en")

    assert parsed.raw_len and parsed.cleaned_len
    assert parsed.cleaned_len <= parsed.raw_len
    assert parsed.selection_meta is not None
    assert parsed.blocks_stats is not None
    assert "meaningful sentence" in parsed.text
