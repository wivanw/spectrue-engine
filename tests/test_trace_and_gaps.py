import json
from uuid import uuid4
from pathlib import Path

import pytest

from spectrue_core.utils import trace as trace_mod


def test_trace_keeps_tail_snippet_and_hash(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECTRUE_ENV", "local")

    def _tmp_trace_dir() -> Path:
        p = tmp_path / "trace"
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(trace_mod, "_trace_dir", _tmp_trace_dir)

    trace_id = f"test-{uuid4()}"
    long_text = "start-" + ("x" * 4500) + "-TAIL_MARKER"

    trace_mod.Trace.start(trace_id)
    trace_mod.Trace.event("llm.prompt", {"prompt": long_text})
    trace_mod.Trace.stop()

    trace_file = _tmp_trace_dir() / f"{trace_id}.jsonl"
    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    prompt_rec = next(r for r in records if r["event"] == "llm.prompt")
    prompt_payload = prompt_rec["data"]["prompt"]

    if isinstance(prompt_payload, str):
        assert "TAIL_MARKER" in prompt_payload
    else:
        assert prompt_payload.get("len") == len(long_text)
        assert "TAIL_MARKER" in prompt_payload.get("tail", "")
        assert prompt_payload.get("sha256")


@pytest.mark.parametrize(
    "lang,label",
    [
        ("uk", "прогалини"),
        ("ru", "пробелы"),
        ("en", "gaps"),
    ],
)
def test_gap_text_is_domain_neutral(lang, label):
    pytest.skip("Method _augment_rationale_with_gaps removed in T164 (LLM-based scoring).")
    # verifier = FactVerifierComposite.__new__(FactVerifierComposite)
    # rationale = "Базове пояснення"
    # augmented = verifier._augment_rationale_with_gaps(
    #     rationale, lang=lang, missing=["by_whom"], max_tier=3
    # )

    # banned = ["гурт", "організатори", "майданчик", "влада", "venue", "organizers", "police", "league"]
    # lower_augmented = augmented.lower()

    # assert label in lower_augmented
    # for word in banned:
    #     assert word not in lower_augmented
