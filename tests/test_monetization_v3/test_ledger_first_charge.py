import pytest
from unittest.mock import MagicMock
from spectrue_core.monetization.adapters.firestore import FirestoreBillingStore

from spectrue_core.monetization.ledger import LedgerEntry

@pytest.fixture
def store():
    db = MagicMock()
    return FirestoreBillingStore(db)

def test_imports():
    assert LedgerEntry is not None

