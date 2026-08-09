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

# DeepSeek constants (V4 generation; the deepseek-chat / deepseek-reasoner
# aliases were retired 2026-07-24)
MODEL_DEEPSEEK_PRO = ModelID.MID          # 1.6T/49B — used for the MID tier
MODEL_DEEPSEEK_FLASH = "deepseek-v4-flash"  # 284B/13B — cheaper, weaker on reasoning

# Default models for pipeline steps (Override via ENV)
# MID = DeepSeek-V4 Pro: keeps V3-class reasoning quality at ~1/11 of PRO's input cost
DEFAULT_MODEL_CLAIM_EXTRACTION = ModelID.MID
DEFAULT_MODEL_INLINE_SOURCE_VERIFICATION = ModelID.NANO
DEFAULT_MODEL_CLUSTERING_STANCE = ModelID.NANO

# OpenAI models (used for other skills)
DEFAULT_MODEL_OPENAI_NANO = ModelID.NANO
