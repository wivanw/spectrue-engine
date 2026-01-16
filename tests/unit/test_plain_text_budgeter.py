# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.analysis.content_budgeter import ContentBudgeter, ContentBudgetConfig


def test_plain_text_budgeter_trims_and_preserves_order():
    blocks = [f"Block {i} " + ("content " * 40) for i in range(20)]
    text = "\n\n".join(blocks * 30)

    cfg = ContentBudgetConfig()
    budgeter = ContentBudgeter(cfg)
    result = budgeter.trim(text)

    assert result.trimmed_len < result.raw_len
    assert result.trimmed_len <= cfg.max_clean_text_chars_huge_input
    trimmed = result.trimmed_text
    assert trimmed.index("Block 0") < trimmed.index("Block 5")
    assert trimmed.index("Block 5") < trimmed.index("Block 10")
