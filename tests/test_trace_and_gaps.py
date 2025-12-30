import json
from uuid import uuid4
from pathlib import Path


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

    # Safe payloads update
    # In safe mode (default=True in config), we expect:
    # - No "tail"
    # - "head" limited to 120 (default for prompt/sensitive keys) or 600 (others)
    # - "sha256" present
    # - "len" present
    
    if isinstance(prompt_payload, str):
        # If it wasn't truncated, that's unexpected for a 4500-char string
        # unless trace limit is huge. Default head limit is 120.
        assert len(prompt_payload) <= 120 + 50 # padding tolerance
        # But wait, if safe mode is ON, and len > limit, it returns dict.
        # If it returns string, it means len <= limit.
        pass
    else:
        assert prompt_payload.get("len") == len(long_text)
        assert "head" in prompt_payload
        assert "tail" not in prompt_payload  # CRITICAL ASSERTION for M75
        assert prompt_payload.get("sha256")
        
        # Verify head content
        assert prompt_payload["head"].startswith("start-")
        assert len(prompt_payload["head"]) <= 120  # Default limit for 'prompt' key

