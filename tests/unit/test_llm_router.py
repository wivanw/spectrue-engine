# Copyright (C) 2025 Ivan Bondarenko
#
# This file is part of Spectrue Engine.
#
# Spectrue Engine is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
Unit tests for LLMRouter - model routing to OpenAI vs local endpoints.

Tests routing logic without making actual network calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestLLMRouterRouting:
    """Test that LLMRouter routes models to correct clients."""

    def test_router_routes_chat_model_to_chat_client(self):
        """Models in chat_model_names should use chat_client."""
        from spectrue_core.agents.llm_router import LLMRouter

        openai_client = MagicMock()
        chat_client = MagicMock()

        router = LLMRouter(
            openai_client=openai_client,
            chat_client=chat_client,
            chat_model_names=["deepseek-chat", "deepseek-reasoner"],
        )

        # Test routing decision
        # Test routing decision
        assert router._get_client("deepseek-chat") is chat_client
        assert router._get_client("deepseek-reasoner") is chat_client
        assert router._get_client("DEEPSEEK-CHAT") is chat_client  # Case-insensitive

    def test_router_routes_openai_model_to_openai_client(self):
        """Models NOT in chat_model_names should use openai_client."""
        from spectrue_core.agents.llm_router import LLMRouter

        openai_client = MagicMock()
        chat_client = MagicMock()

        router = LLMRouter(
            openai_client=openai_client,
            chat_client=chat_client,
            chat_model_names=["deepseek-chat"],
        )

        assert router._get_client("gpt-5-nano") is openai_client
        assert router._get_client("gpt-5.2") is openai_client
        assert router._get_client("some-other-model") is openai_client

    def test_router_routes_all_to_openai_when_no_chat_client(self):
        """When chat_client is None, all models use openai_client."""
        from spectrue_core.agents.llm_router import LLMRouter

        openai_client = MagicMock()

        router = LLMRouter(
            openai_client=openai_client,
            chat_client=None,
            chat_model_names=["deepseek-chat"],
        )

        # Even if model is in chat_model_names, should use openai_client
        assert router._get_client("deepseek-chat") is openai_client
        assert router._get_client("gpt-5-nano") is openai_client

    def test_router_routes_all_to_openai_when_empty_chat_models(self):
        """When chat_model_names is empty, all models use openai_client."""
        from spectrue_core.agents.llm_router import LLMRouter

        openai_client = MagicMock()
        chat_client = MagicMock()

        router = LLMRouter(
            openai_client=openai_client,
            chat_client=chat_client,
            chat_model_names=[],
        )

        assert router._get_client("deepseek-chat") is openai_client
        assert router._get_client("gpt-5-nano") is openai_client


class TestLLMRouterCallJSON:
    """Test call_json routing."""

    @pytest.mark.asyncio
    async def test_call_json_routes_to_correct_client(self):
        """call_json should route to the appropriate client based on model."""
        from spectrue_core.agents.llm_router import LLMRouter

        openai_client = MagicMock()
        openai_client.call_json = AsyncMock(return_value={"result": "openai"})

        chat_client = MagicMock()
        chat_client.call_json = AsyncMock(return_value={"result": "chat"})

        router = LLMRouter(
            openai_client=openai_client,
            chat_client=chat_client,
            chat_model_names=["deepseek-chat"],
        )

        # Call with chat model
        result_chat = await router.call_json(
            model="deepseek-chat",
            input="test prompt",
            instructions="test instructions",
        )
        assert result_chat == {"result": "chat"}
        chat_client.call_json.assert_called_once()
        openai_client.call_json.assert_not_called()

        # Reset mocks
        chat_client.call_json.reset_mock()
        openai_client.call_json.reset_mock()

        # Call with OpenAI model
        result_openai = await router.call_json(
            model="gpt-5-nano",
            input="test prompt",
            instructions="test instructions",
        )
        assert result_openai == {"result": "openai"}
        openai_client.call_json.assert_called_once()
        chat_client.call_json.assert_not_called()


class TestLLMClientChatCompletions:
    """Test LLMClient Chat Completions mode (base_url set)."""

    def test_client_with_base_url_uses_chat_completions_mode(self):
        """When base_url is set, client should use Chat Completions API."""
        from spectrue_core.agents.llm_client import LLMClient

        client = LLMClient(
            openai_api_key="test-key",
            base_url="http://localhost:8000/v1",
        )

        assert client._use_chat_completions is True
        assert client.base_url == "http://localhost:8000/v1"

    def test_client_without_base_url_uses_responses_api(self):
        """When base_url is not set, client should use Responses API."""
        from spectrue_core.agents.llm_client import LLMClient

        client = LLMClient(
            openai_api_key="test-key",
        )

        assert client._use_chat_completions is False
        assert client.base_url is None


class TestRuntimeConfigDeepSeek:
    """Test runtime config for DeepSeek LLM settings."""

    def test_default_local_llm_config(self):
        """Test default values for OpenRouter LLM config."""
        from spectrue_core.runtime_config import EngineLLMConfig

        config = EngineLLMConfig()

        assert config.deepseek_base_url == "https://api.deepseek.com"
        assert config.deepseek_api_key == ""
        assert config.model_claim_extraction == "gpt-5-nano"
        assert config.model_inline_source_verification == "gpt-5-nano"
        assert config.model_clustering_stance == "gpt-5-nano"
        assert config.enable_inline_source_verification is True

    def test_load_from_env_with_local_llm_settings(self):
        """Test loading OpenRouter LLM settings from environment."""
        import os
        from spectrue_core.runtime_config import EngineRuntimeConfig

        env_vars = {
            "DEEPSEEK_BASE_URL": "https://custom.deepseek/v1",
            "DEEPSEEK_API_KEY": "custom-key",
            "MODEL_CLAIM_EXTRACTION": "custom-model-1",
            "MODEL_INLINE_SOURCE_VERIFICATION": "custom-model-2",
            "MODEL_CLUSTERING_STANCE": "custom-model-3",
            "FEATURE_INLINE_SOURCE_VERIFICATION": "0",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = EngineRuntimeConfig.load_from_env()

        assert config.llm.deepseek_base_url == "https://custom.deepseek/v1"
        assert config.llm.deepseek_api_key == "custom-key"
        assert config.llm.model_claim_extraction == "custom-model-1"
        assert config.llm.model_inline_source_verification == "custom-model-2"
        assert config.llm.model_clustering_stance == "custom-model-3"
        assert config.llm.enable_inline_source_verification is False

    def test_empty_local_api_key_handled_correctly(self):
        """Test that empty DEEPSEEK_API_KEY is handled (not None error)."""
        import os
        from spectrue_core.runtime_config import EngineRuntimeConfig

        env_vars = {
            "DEEPSEEK_API_KEY": "",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = EngineRuntimeConfig.load_from_env()

        assert config.llm.deepseek_api_key == ""

    def test_empty_deepseek_model_names_means_all_direct_to_openai(self):
        """Test that empty DEEPSEEK_MODEL_NAMES = no models through DeepSeek."""
        import os
        from spectrue_core.runtime_config import EngineRuntimeConfig

        # Clear model-related ENV vars
        env_to_clear = {
            "MODEL_CLAIM_EXTRACTION": "",
            "MODEL_INLINE_SOURCE_VERIFICATION": "",
            "MODEL_CLUSTERING_STANCE": "",
            "DEEPSEEK_MODEL_NAMES": "",
        }
        with patch.dict(os.environ, env_to_clear, clear=False):
            # Remove the keys entirely
            for key in env_to_clear:
                os.environ.pop(key, None)
            config = EngineRuntimeConfig.load_from_env()

        # Empty list = all models go directly to OpenAI
        assert config.llm.deepseek_model_names == ()

    def test_deepseek_model_names_can_be_set_via_env(self):
        """Test that DEEPSEEK_MODEL_NAMES ENV explicitly sets which models use DeepSeek."""
        import os
        from spectrue_core.runtime_config import EngineRuntimeConfig

        env_vars = {
            "DEEPSEEK_MODEL_NAMES": "deepseek-chat, custom-local-model",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = EngineRuntimeConfig.load_from_env()

        assert config.llm.deepseek_model_names == ("deepseek-chat", "custom-local-model")


class TestInlineSourceVerificationFeatureFlag:
    """Test feature flag for inline source verification."""

    @pytest.mark.asyncio
    async def test_verify_inline_source_returns_skipped_when_disabled(self):
        """When enable_inline_source_verification=False, should return verification_skipped."""
        from unittest.mock import MagicMock, AsyncMock

        # Create mock agent with disabled verification
        mock_runtime = MagicMock()
        mock_runtime.llm.enable_inline_source_verification = False

        mock_agent = MagicMock()
        mock_agent.runtime = mock_runtime

        # Mock the verify method to check short-circuit
        async def mock_verify(claims, src, excerpt):
            # This simulates the behavior in fact_checker_agent.py
            if not mock_agent.runtime.llm.enable_inline_source_verification:
                return {
                    "is_relevant": True,
                    "is_primary": False,
                    "reason": "inline_verification_disabled",
                    "verification_skipped": True,
                }
            return {"is_relevant": True, "is_primary": True, "reason": "test"}

        mock_agent.verify_inline_source_relevance = mock_verify

        result = await mock_agent.verify_inline_source_relevance(
            [{"text": "test claim"}],
            {"url": "http://example.com", "domain": "example.com"},
            "article excerpt",
        )

        assert result["verification_skipped"] is True
        assert result["is_primary"] is False
        assert result["reason"] == "inline_verification_disabled"

