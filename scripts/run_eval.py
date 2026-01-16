#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from spectrue_core.config import SpectrueConfig
from spectrue_core.engine import SpectrueEngine
from spectrue_core.pipeline.mode import AnalysisMode


@dataclass
class Case:
    case_id: str
    lang: str
    text: str


def load_cases(path: Path) -> list[Case]:
    """
    Accepts JSONL where each line is:
      {"id": "...", "lang": "uk", "text": "..."}
    """
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cases.append(
            Case(
                case_id=str(obj.get("id") or obj.get("case_id") or f"case_{len(cases)+1}"),
                lang=str(obj.get("lang") or "en"),
                text=str(obj.get("text") or ""),
            )
        )
    return cases


def _extract_claim_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Best-effort extraction of per-claim results from heterogeneous outputs.
    We do NOT assume a fixed schema to keep this harness resilient.
    """
    for key in ("claim_results", "claims", "results"):
        v = result.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    # Some outputs nest under "audit" or "report"
    audit = result.get("audit")
    if isinstance(audit, dict):
        for key in ("claim_results", "claims"):
            v = audit.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def _claim_id(ci: dict[str, Any], idx: int) -> str:
    return str(ci.get("id") or ci.get("claim_id") or f"c{idx+1}")


def _claim_text(ci: dict[str, Any]) -> str:
    for k in ("claim", "text", "claim_text", "statement"):
        v = ci.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _claim_verdict(ci: dict[str, Any]) -> str:
    for k in ("verdict", "stance", "label", "final_label"):
        v = ci.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    # Some structures nest verdict under "claim_verdict"
    cv = ci.get("claim_verdict")
    if isinstance(cv, dict):
        for k in ("verdict", "stance", "label"):
            v = cv.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().upper()
    return "UNKNOWN"


def _claim_rgba(ci: dict[str, Any]) -> list[float] | None:
    v = ci.get("rgba")
    if isinstance(v, list) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    return None


def _claim_confirm_counts(ci: dict[str, Any]) -> dict[str, Any] | None:
    v = ci.get("confirmation_counts")
    if isinstance(v, dict):
        return v
    return None


def fingerprint_result(result: dict[str, Any]) -> tuple:
    """
    Deterministic fingerprint of a run for stability checks.
    Uses (claim_text, verdict) pairs sorted lexicographically.
    """
    items = _extract_claim_items(result)
    pairs: list[tuple[str, str]] = []
    for i, ci in enumerate(items):
        if not isinstance(ci, dict):
            continue
        t = _claim_text(ci)
        v = _claim_verdict(ci)
        if not t:
            t = _claim_id(ci, i)
        pairs.append((t, v))
    return tuple(sorted(pairs))


def summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    items = _extract_claim_items(result)
    total = 0
    nei = 0
    a_vals: list[float] = []
    c_total_vals: list[float] = []
    for i, ci in enumerate(items):
        if not isinstance(ci, dict):
            continue
        total += 1
        v = _claim_verdict(ci)
        if v in {"NEI", "UNKNOWN", "NOT_ENOUGH_INFO"}:
            nei += 1
        rgba = _claim_rgba(ci)
        if rgba and len(rgba) == 4:
            a_vals.append(float(rgba[3]))
        cc = _claim_confirm_counts(ci)
        if cc and "C_total" in cc:
            try:
                c_total_vals.append(float(cc["C_total"]))
            except Exception:
                pass
    return {
        "claims": total,
        "nei_rate": (nei / total) if total else 0.0,
        "A_avg": (sum(a_vals) / len(a_vals)) if a_vals else None,
        "C_total_avg": (sum(c_total_vals) / len(c_total_vals)) if c_total_vals else None,
    }


async def run_case(engine: SpectrueEngine, case: Case, mode: AnalysisMode, max_credits: int | None) -> dict[str, Any]:
    return await engine.analyze_text(
        text=case.text,
        lang=case.lang,
        analysis_mode=mode,
        max_credits=max_credits,
    )


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SpectrueConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        openai_model=os.environ.get("SPECTRUE_OPENAI_MODEL", "gpt-4o"),
    )
    engine = SpectrueEngine(cfg)

    modes = [AnalysisMode.GENERAL, AnalysisMode.DEEP_V2] if args.both else [AnalysisMode(args.mode)]
    repeats = int(args.repeats)
    max_credits = int(args.max_credits) if args.max_credits is not None else None

    runs_path = out_dir / "eval_runs.jsonl"
    summary_path = out_dir / "eval_summary.json"

    summaries: dict[str, Any] = {"generated_at": datetime.utcnow().isoformat() + "Z", "cases": len(cases), "modes": [m.value for m in modes], "repeats": repeats, "results": {}}

    with runs_path.open("w", encoding="utf-8") as f:
        for mode in modes:
            mode_key = mode.value
            mode_stats = []
            stability_hits = 0
            stability_total = 0

            for case in cases:
                fingerprints = []
                per_run = []

                for r in range(repeats):
                    res = await run_case(engine, case, mode, max_credits=max_credits)
                    fp = fingerprint_result(res)
                    fingerprints.append(fp)
                    s = summarize_run(res)
                    per_run.append(s)

                    f.write(json.dumps({
                        "mode": mode_key,
                        "case_id": case.case_id,
                        "repeat": r,
                        "summary": s,
                    }, ensure_ascii=False) + "\n")

                # stability: all repeats identical
                stability_total += 1
                if all(fp == fingerprints[0] for fp in fingerprints[1:]):
                    stability_hits += 1

                # aggregate per-case (avg over repeats)
                def avg_field(name: str) -> float | None:
                    vals = [x[name] for x in per_run if x.get(name) is not None]
                    if not vals:
                        return None
                    return float(sum(vals) / len(vals))

                mode_stats.append({
                    "case_id": case.case_id,
                    "claims_avg": avg_field("claims"),
                    "nei_rate_avg": avg_field("nei_rate"),
                    "A_avg": avg_field("A_avg"),
                    "C_total_avg": avg_field("C_total_avg"),
                    "stable": all(fp == fingerprints[0] for fp in fingerprints[1:]),
                })

            # mode aggregate
            def avg_over_cases(name: str) -> float | None:
                vals = [x[name] for x in mode_stats if x.get(name) is not None]
                if not vals:
                    return None
                return float(sum(vals) / len(vals))

            summaries["results"][mode_key] = {
                "cases": len(mode_stats),
                "stability_rate": (stability_hits / stability_total) if stability_total else 0.0,
                "claims_avg": avg_over_cases("claims_avg"),
                "nei_rate_avg": avg_over_cases("nei_rate_avg"),
                "A_avg": avg_over_cases("A_avg"),
                "C_total_avg": avg_over_cases("C_total_avg"),
                "per_case": mode_stats,
            }

    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {runs_path}")
    print(f"Wrote: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spectrue evaluation harness (stability + summary metrics)")
    p.add_argument("--cases", required=True, help="Path to JSONL cases file")
    p.add_argument("--out-dir", default="eval_out", help="Output directory")
    p.add_argument("--repeats", type=int, default=2, help="Repeats per case per mode")
    p.add_argument("--max-credits", type=int, default=None, help="Max credits per run (optional)")
    p.add_argument("--both", action="store_true", help="Run both GENERAL and DEEP_V2")
    p.add_argument("--mode", default="deep_v2", help="Single mode to run if not --both")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
