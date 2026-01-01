# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Spectrue Core Engine
====================

The open-source AI fact-checking core.
"""

__version__ = "1.2.0"

# Versioning for saved checks (SaaS-friendly reproducibility).
# When changing prompts/strategy, bump these strings.
PROMPT_VERSION = "fc_agent_v3"
SEARCH_STRATEGY_VERSION = "waterfall_v2"