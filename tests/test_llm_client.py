"""Unit tests for LLM client module: factory, config, JSON extraction."""

import pytest
from unittest.mock import patch, MagicMock

from flowplan_g2p.llm_client import (
    BaseLLMClient,
    LLMConfig,
    OpenAIClient,
    AnthropicClient,
    BedrockClient,
    create_llm_client,
)


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.model_id is None
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.2
        assert cfg.top_p == 0.9
        assert cfg.api_key is None
        assert cfg.aws_region == "us-east-1"
        assert cfg.retry_attempts == 3
        assert cfg.retry_delay == 1.0

    def test_to_dict_excludes_none(self):
        cfg = LLMConfig()  # model_id=None, api_key=None
        d = cfg.to_dict()
        assert "model_id" not in d
        assert "api_key" not in d
        # Non-None fields present
        assert d["provider"] == "openai"
        assert d["max_tokens"] == 4096

    def test_to_dict_includes_all_when_set(self):
        cfg = LLMConfig(provider="anthropic", model_id="claude-sonnet-4-20250514", api_key="sk-test")
        d = cfg.to_dict()
        assert d["model_id"] == "claude-sonnet-4-20250514"
        assert d["api_key"] == "sk-test"


# ---------------------------------------------------------------------------
# create_llm_client factory
# ---------------------------------------------------------------------------

class TestCreateLLMClient:
    @patch("flowplan_g2p.llm_client.openai", create=True)
    def test_openai_returns_correct_type(self, mock_openai, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # Mock the openai module so import succeeds inside OpenAIClient.__init__
        import sys
        mock_mod = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", mock_mod)
        cfg = LLMConfig(provider="openai")
        client = create_llm_client(cfg)
        assert isinstance(client, OpenAIClient)
        assert isinstance(client, BaseLLMClient)

    @patch("flowplan_g2p.llm_client.anthropic", create=True)
    def test_anthropic_returns_correct_type(self, mock_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        import sys
        mock_mod = MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", mock_mod)
        cfg = LLMConfig(provider="anthropic")
        client = create_llm_client(cfg)
        assert isinstance(client, AnthropicClient)
        assert isinstance(client, BaseLLMClient)

    def test_bedrock_returns_correct_type(self, monkeypatch):
        import sys
        mock_boto3 = MagicMock()
        monkeypatch.setitem(sys.modules, "boto3", mock_boto3)
        cfg = LLMConfig(provider="bedrock")
        client = create_llm_client(cfg)
        assert isinstance(client, BedrockClient)
        assert isinstance(client, BaseLLMClient)

    def test_unsupported_provider_raises(self):
        cfg = LLMConfig(provider="unsupported_xyz")
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_llm_client(cfg)

    def test_unsupported_provider_lists_supported(self):
        cfg = LLMConfig(provider="bad")
        with pytest.raises(ValueError, match="openai.*anthropic.*bedrock"):
            create_llm_client(cfg)


# ---------------------------------------------------------------------------
# Missing API key errors
# ---------------------------------------------------------------------------

class TestMissingAPIKey:
    def test_openai_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        import sys
        mock_mod = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", mock_mod)
        cfg = LLMConfig(provider="openai", api_key=None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIClient(cfg)

    def test_anthropic_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        import sys
        mock_mod = MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", mock_mod)
        cfg = LLMConfig(provider="anthropic", api_key=None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicClient(cfg)


# ---------------------------------------------------------------------------
# BaseLLMClient._extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_fenced_json_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = BaseLLMClient._extract_json(text)
        assert result == {"key": "value"}

    def test_raw_json(self):
        text = '{"a": 1, "b": 2}'
        result = BaseLLMClient._extract_json(text)
        assert result == {"a": 1, "b": 2}

    def test_embedded_json(self):
        text = 'Some preamble text {"nested": true} and trailing text'
        result = BaseLLMClient._extract_json(text)
        assert result == {"nested": True}

    def test_no_json_raises(self):
        text = "No JSON here at all"
        with pytest.raises(RuntimeError, match="No valid JSON"):
            BaseLLMClient._extract_json(text)

    def test_complex_nested_json(self):
        text = '```json\n{"outer": {"inner": [1, 2, 3]}}\n```'
        result = BaseLLMClient._extract_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}
