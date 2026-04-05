from spectrue_core.use_cases.verification.scoring.freshness_signal import (
    parse_freshness_signal,
    calculate_freshness_adjustment,
)
from dataclasses import dataclass

@dataclass
class MockEvidence:
    url: str
    snippet: str = ""

def test_parse_freshness_parsed():
    item = MockEvidence(url="https://example.com/2023/news", snippet="some text")
    record = parse_freshness_signal("c1", item, current_year=2024)
    assert record.parse_status == "parsed"
    assert record.parsed_year == 2023
    assert record.recency_score == 0.8
    assert record.contribution_applied is True

def test_parse_freshness_unsupported():
    item = MockEvidence(url="https://example.com/news", snippet="no dates here")
    record = parse_freshness_signal("c1", item, current_year=2024)
    assert record.parse_status == "unsupported"
    assert record.parsed_year is None
    assert record.recency_score is None

def test_parse_freshness_snippet_date():
    item = MockEvidence(url="https://example.com/news", snippet="published in 2021")
    record = parse_freshness_signal("c1", item, current_year=2024)
    assert record.parse_status == "parsed"
    assert record.parsed_year == 2021

def test_calculate_freshness_adjustment_penalty():
    # All items are from 5+ years ago -> avg recency 0.0
    items = [
        MockEvidence(url="https://example.com/2018/news"),
        MockEvidence(url="https://example.com/2017/news")
    ]
    adj = calculate_freshness_adjustment("c1", items, current_year=2024)
    assert adj == -0.2

def test_calculate_freshness_adjustment_no_penalty():
    # Mix of old and new -> avg recency > 0.2
    items = [
        MockEvidence(url="https://example.com/2018/news"),
        MockEvidence(url="https://example.com/2023/news")
    ]
    adj = calculate_freshness_adjustment("c1", items, current_year=2024)
    assert adj == 0.0

def test_calculate_freshness_unsupported_no_penalty():
    items = [
        MockEvidence(url="https://example.com/news")
    ]
    adj = calculate_freshness_adjustment("c1", items, current_year=2024)
    assert adj == 0.0
