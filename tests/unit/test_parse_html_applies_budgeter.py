# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.analysis.text_analyzer import TextAnalyzer


def test_parse_html_applies_budgeter():
    analyzer = TextAnalyzer()
    # The budgeter only engages above max_clean_text_chars_default, measured on
    # what trafilatura extracts — not on raw HTML size. The filler must be unique:
    # this used to repeat "noise123 " in two identical blocks, and trafilatura
    # 2.2.0 dedupes those, halving the extracted text to 74k and dropping it under
    # the 120k threshold. The budgeter then never ran and selection_meta was None.
    # Unique tokens extract identically on 2.0.0 and 2.2.0 (269k chars).
    filler = " ".join(f"filler{i} word{i} token{i}" for i in range(9000))
    article = "<h1>Article</h1><p>" + ("meaningful sentence " * 120) + "</p>"
    html = f"<html><body><p>{filler}</p>{article}</body></html>"

    parsed = analyzer.parse_html(html, language="en")

    assert parsed.raw_len and parsed.cleaned_len
    assert parsed.cleaned_len <= parsed.raw_len
    assert parsed.selection_meta is not None
    assert parsed.blocks_stats is not None
    assert "meaningful sentence" in parsed.text
