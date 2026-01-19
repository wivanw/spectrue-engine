from spectrue_core.llm.model_registry import ModelID
# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Model constants for Spectrue Engine.

Default model IDs for OpenRouter. These can be overridden via ENV variables.
See: https://openrouter.ai/models for available models.
"""

# DeepSeek constant
MODEL_DEEPSEEK_CHAT = ModelID.MID
MODEL_DEEPSEEK_REASONER = "deepseek-reasoner"

# Default models for pipeline steps (Override via ENV)
# Switched to DeepSeek-V3 (deepseek-chat) for cost/performance balance
DEFAULT_MODEL_CLAIM_EXTRACTION = ModelID.MID
DEFAULT_MODEL_INLINE_SOURCE_VERIFICATION = ModelID.NANO
DEFAULT_MODEL_CLUSTERING_STANCE = ModelID.NANO

# OpenAI models (used for other skills)
DEFAULT_MODEL_OPENAI_NANO = ModelID.NANO
