# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""M147 T001: fact-checker benchmark helpers (no network, no dataset)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "fact_check_benchmark", Path(__file__).resolve().parents[2] / "scripts" / "fact_check_benchmark.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_fact_checker_pages_are_recognised():
    assert bench.is_fact_check("https://www.politifact.com/factchecks/2020/x")
    assert bench.is_fact_check("https://factcheck.afp.com/x")
    assert bench.is_fact_check("https://www.usatoday.com/story/news/factcheck/2020/a")   # mixed outlet, fact-check path
    assert bench.is_fact_check("https://www.reuters.com/article/uk-factcheck-x")
    assert not bench.is_fact_check("https://en.wikipedia.org/wiki/Nigeria")
    assert not bench.is_fact_check("https://www.reuters.com/world/africa/nigeria-census")


def test_sample_is_balanced_deduplicated_and_self_contained(tmp_path):
    rows = ([{"claim": f"Claim {i} is true.", "label": "Supported", "claim_types": []} for i in range(5)]
            + [{"claim": "Claim 0 is true", "label": "Supported", "claim_types": []},            # duplicate text
               {"claim": "We built the bridge.", "label": "Supported", "claim_types": []},       # first person
               {"claim": "X said Y.", "label": "Refuted", "claim_types": ["Quote Verification"]}]
            + [{"claim": f"Claim {i} is false.", "label": "Refuted", "claim_types": []} for i in range(5)]
            + [{"claim": "Unclear one.", "label": "Not Enough Evidence", "claim_types": []}])
    src = tmp_path / "dev.json"; src.write_text(json.dumps(rows))
    out = tmp_path / "claims.json"
    bench.sample(src, 6, 1, out)
    claims = json.loads(out.read_text())
    assert sorted(c["truth"] for c in claims) == [0, 0, 0, 1, 1, 1]
    texts = [c["text"].lower().rstrip(".") for c in claims]
    assert len(texts) == len(set(texts)) and not any("we built" in t or "said" in t for t in texts)


def test_summary_metrics_and_spread():
    a = {"c1": {"truth": 1, "g": 0.9, "usd": 0.05, "credits": 6}, "c2": {"truth": 0, "g": 0.2, "usd": 0.04, "credits": 5},
         "c3": {"truth": 0, "g": None, "usd": 0.03, "credits": 4}}
    b = {"c1": {"truth": 1, "g": 0.7, "usd": 0.05, "credits": 6}, "c2": {"truth": 0, "g": 0.6, "usd": 0.04, "credits": 5},
         "c3": {"truth": 0, "g": 0.1, "usd": 0.03, "credits": 4}}
    s = bench.summarize([a, b])
    assert s["runs"][0]["resolved"] == 2 and s["runs"][0]["brier"] == round((0.01 + 0.04) / 2, 4)
    assert s["runs"][0]["accuracy"] == 1.0
    assert s["spread"] == {"n": 2, "median": 0.3, "mean": 0.3, "max": 0.4, "verdict_flips": 1}
