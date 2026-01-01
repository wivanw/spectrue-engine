"""
Fit posterior calibration parameters for claim scoring.

This script estimates alpha/beta for the log-odds posterior model using
MAP (Gaussian priors) on labeled claim data.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from spectrue_core.scoring.claim_posterior import (
    PosteriorParams,
    EvidenceItem,
    compute_claim_posterior,
)


@dataclass(frozen=True)
class DataRow:
    label: float
    log_odds_prior: float
    log_odds_llm: float
    log_odds_evidence: float


@dataclass(frozen=True)
class FitResult:
    alpha: float
    beta: float
    loss: float
    steps: int


def _logit(p: float, eps: float = 1e-9) -> float:
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of records")
        return payload

    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _parse_label(record: dict, *, index: int) -> float:
    for key in ("label", "y", "truth", "is_true"):
        if key in record:
            value = record[key]
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            label = float(value)
            if not 0.0 <= label <= 1.0:
                raise ValueError(f"Label out of range at record {index}")
            return label
    raise ValueError(f"Missing label at record {index}")


def _coerce_float(value: object, *, name: str, index: int) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} at record {index}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Non-finite {name} at record {index}")
    return out


def _extract_log_odds(record: dict, *, index: int) -> tuple[float, float, float]:
    keys = (
        ("log_odds_prior", "l_prior"),
        ("log_odds_llm", "l_llm"),
        ("log_odds_evidence", "l_evidence"),
    )
    values: list[float] = []
    for candidates in keys:
        found = None
        for key in candidates:
            if key in record:
                found = record[key]
                break
        if found is None:
            return ()
        values.append(_coerce_float(found, name=candidates[0], index=index))
    return values[0], values[1], values[2]


def _extract_from_raw(
    record: dict,
    *,
    index: int,
    params: PosteriorParams,
) -> tuple[float, float, float]:
    llm_score = record.get("llm_verdict_score")
    if llm_score is None:
        llm_score = record.get("p_llm")
    if llm_score is None:
        llm_score = record.get("llm_score")

    best_tier = record.get("best_tier")
    if best_tier is None:
        best_tier = record.get("tier")

    evidence_items = record.get("evidence_items")
    if evidence_items is None:
        evidence_items = record.get("evidence")

    if llm_score is None or evidence_items is None:
        raise ValueError(f"Missing raw fields at record {index}")

    items: list[EvidenceItem] = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        stance = item.get("stance") or item.get("verdict") or ""
        tier = item.get("tier")
        relevance = item.get("relevance")
        if not isinstance(relevance, (int, float)):
            relevance = 0.5
        quote_present = bool(
            item.get("quote_present")
            or item.get("quote")
            or item.get("has_quote")
        )
        items.append(
            EvidenceItem(
                stance=str(stance),
                tier=tier,
                relevance=float(relevance),
                quote_present=quote_present,
            )
        )

    if "p_prior" in record and "p_llm" in record and "log_odds_evidence" in record:
        l_prior = _logit(_coerce_float(record["p_prior"], name="p_prior", index=index))
        l_llm = _logit(_coerce_float(record["p_llm"], name="p_llm", index=index))
        l_evidence = _coerce_float(record["log_odds_evidence"], name="log_odds_evidence", index=index)
        return l_prior, l_llm, l_evidence

    result = compute_claim_posterior(
        llm_verdict_score=float(llm_score),
        best_tier=str(best_tier) if best_tier is not None else None,
        evidence_items=items,
        params=params,
    )
    return result.log_odds_prior, result.log_odds_llm, result.log_odds_evidence


def load_dataset(path: Path, *, params: PosteriorParams) -> list[DataRow]:
    records = _read_records(path)
    data: list[DataRow] = []
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {idx} is not an object")
        label = _parse_label(record, index=idx)
        extracted = _extract_log_odds(record, index=idx)
        if extracted:
            l_prior, l_llm, l_evidence = extracted
        else:
            l_prior, l_llm, l_evidence = _extract_from_raw(
                record,
                index=idx,
                params=params,
            )
        data.append(
            DataRow(
                label=label,
                log_odds_prior=l_prior,
                log_odds_llm=l_llm,
                log_odds_evidence=l_evidence,
            )
        )
    if not data:
        raise ValueError("Dataset is empty")
    return data


def compute_loss_and_grad(
    data: Iterable[DataRow],
    *,
    alpha: float,
    beta: float,
    prior_alpha_mean: float,
    prior_beta_mean: float,
    prior_alpha_std: float,
    prior_beta_std: float,
) -> tuple[float, float, float]:
    nll = 0.0
    grad_alpha = 0.0
    grad_beta = 0.0
    count = 0

    for row in data:
        count += 1
        z = row.log_odds_prior + alpha * row.log_odds_llm + beta * row.log_odds_evidence
        p = _sigmoid(z)
        p = max(1e-9, min(1.0 - 1e-9, p))

        nll -= row.label * math.log(p) + (1.0 - row.label) * math.log(1.0 - p)
        diff = p - row.label
        grad_alpha += diff * row.log_odds_llm
        grad_beta += diff * row.log_odds_evidence

    if count == 0:
        raise ValueError("Dataset is empty")

    var_alpha = prior_alpha_std ** 2
    var_beta = prior_beta_std ** 2
    if var_alpha <= 0 or var_beta <= 0:
        raise ValueError("Prior std must be positive")

    prior_penalty = (
        ((alpha - prior_alpha_mean) ** 2) / (2.0 * var_alpha)
        + ((beta - prior_beta_mean) ** 2) / (2.0 * var_beta)
    )
    grad_alpha += (alpha - prior_alpha_mean) / var_alpha
    grad_beta += (beta - prior_beta_mean) / var_beta

    scale = float(count)
    loss = (nll + prior_penalty) / scale
    grad_alpha /= scale
    grad_beta /= scale
    return loss, grad_alpha, grad_beta


def fit_map(
    data: list[DataRow],
    *,
    init_alpha: float,
    init_beta: float,
    prior_alpha_mean: float,
    prior_beta_mean: float,
    prior_alpha_std: float,
    prior_beta_std: float,
    lr: float,
    max_steps: int,
    tol: float,
    log_interval: int,
) -> FitResult:
    alpha = init_alpha
    beta = init_beta
    prev_loss = None

    for step in range(1, max_steps + 1):
        loss, grad_alpha, grad_beta = compute_loss_and_grad(
            data,
            alpha=alpha,
            beta=beta,
            prior_alpha_mean=prior_alpha_mean,
            prior_beta_mean=prior_beta_mean,
            prior_alpha_std=prior_alpha_std,
            prior_beta_std=prior_beta_std,
        )

        alpha -= lr * grad_alpha
        beta -= lr * grad_beta

        if log_interval and step % log_interval == 0:
            print(f"step={step} loss={loss:.6f} alpha={alpha:.4f} beta={beta:.4f}")

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            return FitResult(alpha=alpha, beta=beta, loss=loss, steps=step)
        prev_loss = loss

    return FitResult(alpha=alpha, beta=beta, loss=loss, steps=max_steps)


def evaluate_metrics(data: list[DataRow], *, alpha: float, beta: float) -> dict:
    total = len(data)
    if total == 0:
        raise ValueError("Dataset is empty")

    log_loss = 0.0
    brier = 0.0
    correct = 0

    for row in data:
        z = row.log_odds_prior + alpha * row.log_odds_llm + beta * row.log_odds_evidence
        p = _sigmoid(z)
        p = max(1e-9, min(1.0 - 1e-9, p))
        log_loss -= row.label * math.log(p) + (1.0 - row.label) * math.log(1.0 - p)
        brier += (p - row.label) ** 2
        if (p >= 0.5) == (row.label >= 0.5):
            correct += 1

    return {
        "count": total,
        "log_loss": log_loss / total,
        "brier": brier / total,
        "accuracy": correct / total,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit posterior alpha/beta using MAP estimation",
    )
    parser.add_argument("--input", required=True, help="Path to JSON/JSONL dataset")
    parser.add_argument("--output", help="Write fitted parameters to JSON")
    parser.add_argument("--init-alpha", type=float, help="Initial alpha value")
    parser.add_argument("--init-beta", type=float, help="Initial beta value")
    parser.add_argument("--prior-alpha-mean", type=float, default=1.0)
    parser.add_argument("--prior-beta-mean", type=float, default=1.0)
    parser.add_argument("--prior-alpha-std", type=float, default=0.5)
    parser.add_argument("--prior-beta-std", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    params = PosteriorParams()
    dataset = load_dataset(Path(args.input), params=params)

    init_alpha = args.init_alpha if args.init_alpha is not None else args.prior_alpha_mean
    init_beta = args.init_beta if args.init_beta is not None else args.prior_beta_mean

    result = fit_map(
        dataset,
        init_alpha=init_alpha,
        init_beta=init_beta,
        prior_alpha_mean=args.prior_alpha_mean,
        prior_beta_mean=args.prior_beta_mean,
        prior_alpha_std=args.prior_alpha_std,
        prior_beta_std=args.prior_beta_std,
        lr=args.lr,
        max_steps=args.max_steps,
        tol=args.tol,
        log_interval=args.log_interval,
    )

    metrics = evaluate_metrics(dataset, alpha=result.alpha, beta=result.beta)
    output = {
        "alpha": result.alpha,
        "beta": result.beta,
        "loss": result.loss,
        "steps": result.steps,
        **metrics,
    }

    print(json.dumps(output, indent=2, sort_keys=True))

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
