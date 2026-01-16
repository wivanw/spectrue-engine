
import pytest
from spectrue_core.agents.skills.scoring_contract import (
    build_stance_matrix_instructions,
    build_stance_matrix_prompt,
    STANCE_PASS_SINGLE
)

class TestScoringContractRegression:
    def test_build_stance_matrix_instructions_no_crash(self):
        """
        Regression test for 'Invalid format specifier' crash.
        Ensures that the f-string in build_stance_matrix_instructions
        correctly handles the JSON example (with escaped braces).
        """
        try:
            instructions = build_stance_matrix_instructions(
                num_sources=5,
                pass_type=STANCE_PASS_SINGLE
            )
            assert "You are an Evidence Analyst" in instructions
            assert '"matrix": [' in instructions
            assert "{{" not in instructions  # Should be rendered as single brace
            assert "}}" not in instructions
        except Exception as e:
            pytest.fail(f"build_stance_matrix_instructions raised exception: {e}")

    def test_build_stance_matrix_prompt_no_crash(self):
        """Ensure prompt builder also works."""
        claims = [{"id": "c1", "text": "Claim 1"}]
        sources = [{"index": 0, "text": "Source 1"}]
        try:
            prompt = build_stance_matrix_prompt(
                claims_lite=claims,
                sources_lite=sources
            )
            assert "Claim 1" in prompt
            assert "Source 1" in prompt
        except Exception as e:
            pytest.fail(f"build_stance_matrix_prompt raised exception: {e}")
