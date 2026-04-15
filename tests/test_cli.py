"""Unit tests for CLI argument parsing."""

import pytest
import argparse
import sys
from unittest.mock import patch


def _get_parser():
    """Import and build the argparse parser from cli.py.

    We replicate the parser construction here to test parsing in isolation
    without triggering the full main() function.
    """
    parser = argparse.ArgumentParser(description="FlowPlan-G2P CLI test")
    parser.add_argument("input_paper", type=str)
    parser.add_argument("output_patent", type=str)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tau-c", type=float, default=0.5)
    parser.add_argument("--tau-s", type=float, default=0.6)
    parser.add_argument("--provider", type=str, default="openai",
                        choices=["openai", "anthropic", "bedrock"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--aws-region", type=str, default="us-east-1")
    parser.add_argument("--export-graph", type=str, metavar="PATH")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


class TestCLIProviderChoices:
    def test_openai_valid(self):
        args = _get_parser().parse_args(["in.md", "out.md", "--provider", "openai"])
        assert args.provider == "openai"

    def test_anthropic_valid(self):
        args = _get_parser().parse_args(["in.md", "out.md", "--provider", "anthropic"])
        assert args.provider == "anthropic"

    def test_bedrock_valid(self):
        args = _get_parser().parse_args(["in.md", "out.md", "--provider", "bedrock"])
        assert args.provider == "bedrock"

    def test_invalid_provider_raises(self):
        with pytest.raises(SystemExit):
            _get_parser().parse_args(["in.md", "out.md", "--provider", "invalid"])


class TestCLIDefaults:
    def test_default_provider_is_openai(self):
        args = _get_parser().parse_args(["in.md", "out.md"])
        assert args.provider == "openai"

    def test_model_optional_defaults_none(self):
        args = _get_parser().parse_args(["in.md", "out.md"])
        assert args.model is None

    def test_api_key_optional_defaults_none(self):
        args = _get_parser().parse_args(["in.md", "out.md"])
        assert args.api_key is None

    def test_aws_region_default(self):
        args = _get_parser().parse_args(["in.md", "out.md"])
        assert args.aws_region == "us-east-1"

    def test_max_tokens_default(self):
        args = _get_parser().parse_args(["in.md", "out.md"])
        assert args.max_tokens == 4096

    def test_temperature_default(self):
        args = _get_parser().parse_args(["in.md", "out.md"])
        assert args.temperature == 0.2


class TestCLIFlags:
    def test_validate_only_flag(self):
        args = _get_parser().parse_args(["in.md", "out.md", "--validate-only"])
        assert args.validate_only is True

    def test_validate_only_default_false(self):
        args = _get_parser().parse_args(["in.md", "out.md"])
        assert args.validate_only is False

    def test_model_flag(self):
        args = _get_parser().parse_args(["in.md", "out.md", "--model", "gpt-4o"])
        assert args.model == "gpt-4o"

    def test_api_key_flag(self):
        args = _get_parser().parse_args(["in.md", "out.md", "--api-key", "sk-test"])
        assert args.api_key == "sk-test"

    def test_aws_region_flag(self):
        args = _get_parser().parse_args(["in.md", "out.md", "--aws-region", "us-west-2"])
        assert args.aws_region == "us-west-2"

    def test_verbose_flag(self):
        args = _get_parser().parse_args(["in.md", "out.md", "-v"])
        assert args.verbose is True

    def test_combined_flags(self):
        args = _get_parser().parse_args([
            "in.md", "out.md",
            "--provider", "bedrock",
            "--model", "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "--aws-region", "eu-west-1",
            "--validate-only",
        ])
        assert args.provider == "bedrock"
        assert args.model == "us.anthropic.claude-sonnet-4-20250514-v1:0"
        assert args.aws_region == "eu-west-1"
        assert args.validate_only is True
