# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from spectrue_core.analysis.content_budgeter import ContentBudgeter, ContentBudgetConfig


def test_small_plain_text_passes_through():
    text = "Short note about content budgeting.\nStill concise."
    cfg = ContentBudgetConfig()
    budgeter = ContentBudgeter(cfg)

    result = budgeter.trim(text)

    assert result.trimmed_text == text
    assert result.selection_meta == []
    assert result.trimmed_len == result.raw_len
