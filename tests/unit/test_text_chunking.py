# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors

from spectrue_core.utils.text_chunking import CoverageSampler

class TestCoverageSampler:
    
    def test_chunking_preserves_content(self):
        """Total merged content length (accounting for overlaps/gaps) should match."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        sampler = CoverageSampler()
        chunks = sampler.chunk(text, max_chunk_chars=10) # Small chunk size to force split
        
        # Verify coverage
        assert len(chunks) >= 2
        # Text reconstruction
        reconstructed = "".join(c.text for c in chunks)
        assert reconstructed == text
        
        # Verify offsets
        for c in chunks:
            assert  text[c.char_start:c.char_end] == c.text

    def test_huge_block_splitting(self):
        """A block larger than max_chunk_chars must be forcibly split."""
        text = "A" * 100
        sampler = CoverageSampler()
        chunks = sampler.chunk(text, max_chunk_chars=40)
        
        assert len(chunks) == 3 # 40 + 40 + 20
        assert chunks[0].length == 40
        assert chunks[1].length == 40
        assert chunks[2].length == 20
        
        assert chunks[0].char_start == 0
        assert chunks[1].char_start == 40
        assert chunks[2].char_start == 80

    def test_merge_introduces_separators(self):
        sampler = CoverageSampler()
        cleaned = ["Part A", "Part B"]
        merged = sampler.merge(cleaned)
        assert "--- [SECTION BREAK] ---" in merged
        assert merged.startswith("Part A")
        assert merged.endswith("Part B")
