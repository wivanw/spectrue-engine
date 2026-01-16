# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Spectrue Contributors
"""DAG pipeline metadata constants."""

DAG_EXECUTION_STATE_KEY = "dag_execution_state"
DAG_EXECUTION_SUMMARY_KEY = "dag_execution_summary"

DAG_STEP_STATUS_PENDING = "pending"
DAG_STEP_STATUS_RUNNING = "running"
DAG_STEP_STATUS_SUCCEEDED = "succeeded"
DAG_STEP_STATUS_FAILED = "failed"
DAG_STEP_STATUS_SKIPPED = "skipped"
