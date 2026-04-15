"""Unit tests for FlowPlanG2P pipeline initialization and validation."""

import pytest
from unittest.mock import patch, MagicMock

from flowplan_g2p.llm_client import LLMConfig, BaseLLMClient
from flowplan_g2p.pipeline import FlowPlanG2P


def _mock_create_llm_client(config):
    """Return a MagicMock that quacks like BaseLLMClient."""
    mock_client = MagicMock(spec=BaseLLMClient)
    mock_client.config = config
    mock_client.validate_connection.return_value = True
    mock_client.get_model_info.return_value = {
        "provider": config.provider,
        "model_id": config.model_id,
    }
    return mock_client


class TestPipelineInit:
    @patch("flowplan_g2p.pipeline.create_llm_client", side_effect=_mock_create_llm_client)
    def test_init_with_default_config(self, mock_factory):
        pipeline = FlowPlanG2P()
        mock_factory.assert_called_once()
        assert pipeline.llm_config.provider == "openai"
        assert pipeline.llm_client is not None

    @patch("flowplan_g2p.pipeline.create_llm_client", side_effect=_mock_create_llm_client)
    def test_init_with_custom_provider(self, mock_factory):
        cfg = LLMConfig(provider="anthropic", api_key="test-key")
        pipeline = FlowPlanG2P(llm_config=cfg)
        assert pipeline.llm_config.provider == "anthropic"
        mock_factory.assert_called_once_with(cfg)

    @patch("flowplan_g2p.pipeline.create_llm_client", side_effect=_mock_create_llm_client)
    def test_init_with_bedrock_provider(self, mock_factory):
        cfg = LLMConfig(provider="bedrock", aws_region="us-west-2")
        pipeline = FlowPlanG2P(llm_config=cfg)
        assert pipeline.llm_config.aws_region == "us-west-2"

    @patch("flowplan_g2p.pipeline.create_llm_client", side_effect=_mock_create_llm_client)
    def test_stages_initialized(self, mock_factory):
        pipeline = FlowPlanG2P()
        assert pipeline.concept_induction is not None
        assert pipeline.paragraph_planning is not None
        assert pipeline.graph_generation is not None


class TestPipelineValidateSetup:
    @patch("flowplan_g2p.pipeline.create_llm_client", side_effect=_mock_create_llm_client)
    def test_validate_returns_true_on_success(self, mock_factory):
        pipeline = FlowPlanG2P()
        pipeline.llm_client.validate_connection.return_value = True
        assert pipeline.validate_setup() is True

    @patch("flowplan_g2p.pipeline.create_llm_client", side_effect=_mock_create_llm_client)
    def test_validate_returns_false_on_failure(self, mock_factory):
        pipeline = FlowPlanG2P()
        pipeline.llm_client.validate_connection.return_value = False
        assert pipeline.validate_setup() is False

    @patch("flowplan_g2p.pipeline.create_llm_client", side_effect=_mock_create_llm_client)
    def test_validate_returns_false_on_exception(self, mock_factory):
        pipeline = FlowPlanG2P()
        pipeline.llm_client.validate_connection.side_effect = Exception("network error")
        assert pipeline.validate_setup() is False
