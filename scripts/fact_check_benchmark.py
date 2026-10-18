#!/usr/bin/env python3
# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fact-checker benchmark: how well G (the probability that a claim is true) matches fact-checker verdicts.

Fact-checker verdicts are the answer key, so the run must not see them:
- the oracle step is disabled;
- fact-checker domains and fact-check paths are excluded from retrieval (search and its cache).

The dataset is NOT part of the repository. AVeriTeC (fever.ai, CC BY-NC 4.0) is read from a local path:

    python scripts/fact_check_benchmark.py sample --averitec ~/data/averitec/dev.json --n 60 --out bench/claims.json
    python scripts/fact_check_benchmark.py run --claims bench/claims.json --out bench/run_a
    python scripts/fact_check_benchmark.py report bench/run_a bench/run_b

`run` writes `<out>/results.jsonl` (one row per claim: G, credits, USD, sources) and `<out>/traces/<id>.json`.
`report` prints Brier, log-loss, accuracy, a reliability table, cost, and the run-to-run spread when given 2+ runs.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import random
import re
import shutil
import statistics
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

# Dedicated fact-checking sites (AVeriTeC dev fact_checking_article hosts + Google Fact Check publishers + IFCN
# members). Mixed news outlets are not listed; their fact-check sections are caught by FACT_CHECK_PATH.
FACT_CHECK_DOMAINS: tuple[str, ...] = (
    "politifact.com", "leadstories.com", "fullfact.org", "factcheck.org", "snopes.com", "checkyourfact.com",
    "factcheck.afp.com", "factuel.afp.com", "africacheck.org", "dubawa.org", "polygraph.info", "healthfeedback.org",
    "climatefeedback.org", "sciencefeedback.co", "truthorfiction.com", "factcheck.thedispatch.com", "covid19facts.ca",
    "stopfake.org", "voxukraine.org", "provereno.media", "correctiv.org", "dpa-factchecking.com", "mimikama.org",
    "maldita.es", "newtral.es", "verificat.cat", "chequeado.com", "colombiacheck.com", "aosfatos.org",
    "lupa.uol.com.br", "mygopen.com", "tfc-taiwan.org.tw", "factcheckcenter.jp", "infact.press", "misbar.com",
    "teyit.org", "demagog.cz", "demagog.org.pl", "faktisk.no", "factcheckni.org", "ghanafact.com", "zimfact.org",
    "icirnigeria.org", "verafiles.org", "newsmeter.in", "altnews.in", "newsmobile.in", "boomlive.in",
    "vishvasnews.com", "factly.in", "newschecker.in", "fit.thequint.com", "logicallyfacts.com", "poynter.org",
)
FACT_CHECK_PATH = re.compile(r"(fact-?check|hoax-alert)", re.I)
TAVILY_EXCLUDE_CAP = 32  # TavilyClient sends at most 32 exclude domains; the result filter covers the rest


def is_fact_check(url: str) -> bool:
    p = urllib.parse.urlparse(url or "")
    host = p.netloc.lower().removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in FACT_CHECK_DOMAINS) or bool(FACT_CHECK_PATH.search(p.path))


# ---------------------------------------------------------------- sample

def sample(averitec: Path, n: int, seed: int, out: Path) -> None:
    """Balanced Supported/Refuted claims, deduplicated by text, self-contained (no first-person speaker)."""
    data = json.loads(averitec.read_text(encoding="utf-8"))
    first_person = {"i", "we", "our", "my", "us"}
    seen: set[str] = set()
    pools: dict[str, list[dict]] = {"Supported": [], "Refuted": []}
    for i, x in enumerate(data):
        text = " ".join(str(x.get("claim") or "").split())
        key = text.lower().rstrip(".")
        if x.get("label") not in pools or not text or key in seen or len(text) > 220:
            continue
        if "Quote Verification" in (x.get("claim_types") or []):
            continue
        if first_person & set(re.findall(r"[a-z']+", text.lower())):
            continue
        seen.add(key)
        pools[x["label"]].append({"id": f"avt_{i:03d}", "text": text, "label": x["label"],
                                  "truth": 1 if x["label"] == "Supported" else 0, "date": x.get("claim_date")})
    rng = random.Random(seed)
    half = n // 2
    if min(len(v) for v in pools.values()) < half:
        raise SystemExit(f"not enough claims for n={n}: {({k: len(v) for k, v in pools.items()})}")
    claims = rng.sample(pools["Supported"], half) + rng.sample(pools["Refuted"], n - half)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(claims, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(claims)} claims -> {out}")


# ---------------------------------------------------------------- run

def _isolate_answer_key() -> None:
    from spectrue_core.pipeline.steps.oracle_flow import OracleFlowStep
    from spectrue_core.tools.web_search_tool import WebSearchTool
    from spectrue_core.utils.trace import Trace

    async def oracle_off(self, ctx):
        return ctx.set_extra("oracle_hit", False)

    original = WebSearchTool._search_internal

    async def search_without_fact_checkers(self, **kw):
        kw["exclude_domains"] = sorted(set((kw.get("exclude_domains") or []) + list(FACT_CHECK_DOMAINS[:TAVILY_EXCLUDE_CAP])))
        context, sources = await original(self, **kw)
        kept = [s for s in sources or [] if not is_fact_check(str(s.get("url") or s.get("link") or ""))]
        if len(kept) != len(sources or []):
            Trace.event("benchmark.fact_check_filtered", {"removed": len(sources) - len(kept)})
        return context, kept

    OracleFlowStep.run = oracle_off
    WebSearchTool._search_internal = search_without_fact_checkers


def _load_env(path: Path) -> None:
    import os
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _g(result: dict) -> float | None:
    v = result.get("veracity_signed")
    return float(v) if isinstance(v, (int, float)) and 0.0 <= v <= 1.0 else None


async def _run(claims: list[dict], out: Path, mode: str) -> None:
    from spectrue_core.config import SpectrueConfig
    from spectrue_core.engine import SpectrueEngine
    from spectrue_core.utils.trace import _trace_dir

    _isolate_answer_key()
    engine = SpectrueEngine(SpectrueConfig())
    (out / "traces").mkdir(parents=True, exist_ok=True)
    done = set()
    results = out / "results.jsonl"
    if results.exists():
        done = {json.loads(line)["id"] for line in results.read_text(encoding="utf-8").splitlines() if line.strip()}
    with results.open("a", encoding="utf-8") as fh:
        for c in claims:
            if c["id"] in done:
                continue
            t0 = time.monotonic()
            try:
                r, err = await engine.analyze_text(c["text"], lang=c.get("lang", "en"), analysis_mode=mode), None
            except Exception as exc:  # recorded per claim; the batch continues
                r, err = {}, f"{type(exc).__name__}: {exc}"
            for f in glob.glob(str(_trace_dir() / "*.json")):  # the engine keeps only the latest trace
                shutil.copy(f, out / "traces" / f"{c['id']}.json")
            cs = r.get("cost_summary") or {}
            urls = [s.get("url") for s in (r.get("sources") or []) if isinstance(s, dict)]
            row = {"id": c["id"], "truth": c["truth"], "text": c["text"], "g": _g(r), "error": err,
                   "credits": r.get("credits"), "usd": cs.get("total_usd"), "sec": round(time.monotonic() - t0, 1),
                   "sources": urls, "fact_check_leaks": sum(1 for u in urls if is_fact_check(u or ""))}
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            fh.flush()
            print(c["id"], c["truth"], row["g"], row["credits"], row["sec"], err or "", flush=True)


# ---------------------------------------------------------------- report

def _load(run: Path) -> dict[str, dict]:
    rows = [json.loads(x) for x in (run / "results.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["id"]: r for r in rows}


def summarize(runs: list[dict[str, dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"runs": []}
    for rows in runs:
        res = [(r["truth"], r["g"]) for r in rows.values() if r["g"] is not None]
        k = len(res)
        brier = sum((p - t) ** 2 for t, p in res) / k if k else None
        logloss = (-sum(t * math.log(max(p, 1e-3)) + (1 - t) * math.log(max(1 - p, 1e-3)) for t, p in res) / k) if k else None
        usd = [r["usd"] for r in rows.values() if isinstance(r.get("usd"), (int, float))]
        cr = [r["credits"] for r in rows.values() if isinstance(r.get("credits"), (int, float))]
        bins = []
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)):
            b = [(t, p) for t, p in res if lo <= p < hi]
            if b:
                bins.append({"bin": f"{lo:.1f}-{min(hi, 1):.1f}", "n": len(b), "mean_g": round(sum(p for _, p in b) / len(b), 3),
                             "frac_true": round(sum(t for t, _ in b) / len(b), 3)})
        out["runs"].append({
            "n": len(rows), "resolved": k, "brier": brier and round(brier, 4), "logloss": logloss and round(logloss, 3),
            "accuracy": round(sum((p >= .5) == (t == 1) for t, p in res) / k, 3) if k else None,
            "usd_per_check": round(statistics.mean(usd), 4) if usd else None,
            "credits_per_check": round(statistics.mean(cr), 2) if cr else None,
            "errors": sum(1 for r in rows.values() if r.get("error")),
            "fact_check_leaks": sum(r.get("fact_check_leaks", 0) for r in rows.values()), "reliability": bins,
        })
    if len(runs) >= 2:
        ids = set(runs[0]).intersection(*runs[1:])
        spreads, flips = [], 0
        for i in ids:
            gs = [r[i]["g"] for r in runs if r[i]["g"] is not None]
            if len(gs) == len(runs):
                spreads.append(max(gs) - min(gs))
                flips += len({g >= .5 for g in gs}) > 1
        if spreads:
            out["spread"] = {"n": len(spreads), "median": round(statistics.median(spreads), 3),
                             "mean": round(statistics.mean(spreads), 3), "max": round(max(spreads), 3), "verdict_flips": flips}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--averitec", type=Path, required=True)
    s.add_argument("--n", type=int, default=60); s.add_argument("--seed", type=int, default=147)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run"); r.add_argument("--claims", type=Path, required=True); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--mode", default="general", choices=["general", "deep_v2"])
    r.add_argument("--env", type=Path, help="dotenv file with provider keys (values already in the environment win)")
    p = sub.add_parser("report"); p.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        sample(a.averitec, a.n, a.seed, a.out)
    elif a.cmd == "run":
        if a.env:
            _load_env(a.env)
        asyncio.run(_run(json.loads(a.claims.read_text(encoding="utf-8")), a.out, a.mode))
    else:
        print(json.dumps(summarize([_load(x) for x in a.runs]), indent=1))


if __name__ == "__main__":
    sys.exit(main())
