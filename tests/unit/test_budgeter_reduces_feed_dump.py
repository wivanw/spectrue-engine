# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.analysis.content_budgeter import ContentBudgeter, ContentBudgetConfig


def _build_feed_dump() -> str:
    nav_block = "NAVIGATION 12345 :: " + ("menu " * 20)
    repeated_nav = "\n\n".join(nav_block for _ in range(1200))
    unique_article = "Unique article body starts here. " + ("signal content " * 80)
    footer = "FOOTER 999 :: " + ("links " * 30)
    filler = "\n\n".join("FILLER " + ("x" * 120) for _ in range(600))
    return "\n\n".join([repeated_nav, unique_article, filler, repeated_nav, footer])


def test_budgeter_reduces_feed_dump():
    text = _build_feed_dump()
    cfg = ContentBudgetConfig()
    budgeter = ContentBudgeter(cfg)

    result = budgeter.trim(text)

    assert result.raw_len > 300_000
    assert result.trimmed_len < result.raw_len
    assert result.trimmed_len <= cfg.max_clean_text_chars_huge_input
    assert "Unique article body starts here" in result.trimmed_text
    assert result.blocks_stats["selected_count"] < result.blocks_stats["block_count"]
    assert result.selection_meta
