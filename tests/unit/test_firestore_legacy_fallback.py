# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Unit tests for legacy fallback in FirestoreBillingStore."""

import sys
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

# Mock firestore at the system level BEFORE any imports
mock_firestore = MagicMock()
sys.modules["firebase_admin"] = MagicMock()
sys.modules["firebase_admin.firestore"] = mock_firestore

from spectrue_core.monetization.adapters.firestore import FirestoreBillingStore
from spectrue_core.monetization.config import MonetizationConfig
from spectrue_core.monetization.types import MoneySC

class TestFirestoreLegacyFallback:
    def test_read_user_wallet_legacy_fallback(self):
        # Setup mock db
        mock_db = MagicMock()
        mock_user_doc = MagicMock()
        mock_snapshot = MagicMock()
        
        # Scenario: balance_sc is missing, credits exists
        mock_snapshot.exists = True
        mock_snapshot.to_dict.return_value = {
            "credits": 100,
            "available_sc": 50,
        }
        mock_user_doc.get.return_value = mock_snapshot
        mock_db.collection.return_value.document.return_value = mock_user_doc
        
        config = MonetizationConfig()
        store = FirestoreBillingStore(mock_db, config=config)
        
        # Execute
        wallet = store.read_user_wallet("user1")
        
        # Verify
        assert wallet.balance_sc.value == Decimal("100")
        assert wallet.available_sc.value == Decimal("50")

    def test_read_user_wallet_v3_priority(self):
        # Setup mock db
        mock_db = MagicMock()
        mock_user_doc = MagicMock()
        mock_snapshot = MagicMock()
        
        # Scenario: both balance_sc and credits exist, balance_sc should win
        mock_snapshot.exists = True
        mock_snapshot.to_dict.return_value = {
            "balance_sc": 200,
            "credits": 100,
            "available_sc": 50,
        }
        mock_user_doc.get.return_value = mock_snapshot
        mock_db.collection.return_value.document.return_value = mock_user_doc
        
        config = MonetizationConfig()
        store = FirestoreBillingStore(mock_db, config=config)
        
        # Execute
        wallet = store.read_user_wallet("user1")
        
        # Verify
        assert wallet.balance_sc.value == Decimal("200")

    def test_user_balance_from_snapshot_legacy_fallback(self):
        # This tests the method used by reserve_run/settle_run
        mock_db = MagicMock()
        mock_snapshot = MagicMock()
        
        mock_snapshot.exists = True
        mock_snapshot.id = "user1"
        mock_snapshot.to_dict.return_value = {
            "credits": 300,
        }
        
        config = MonetizationConfig()
        store = FirestoreBillingStore(mock_db, config=config)
        
        # Execute
        ub = store._user_balance_from_snapshot(mock_snapshot)
        
        # Verify
        assert ub.balance_sc == Decimal("300")
        assert ub.legacy_credits == Decimal("300")
