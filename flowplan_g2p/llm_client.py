"""
Multi-provider LLM client abstraction for FlowPlan-G2P.
Supports OpenAI, Anthropic (direct API), and AWS Bedrock providers.
"""

import abc
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from dataclasses import dataclass, fields


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    provider: str = "openai"
    model_id: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    api_key: Optional[str] = None
    aws_region: str = "us-east-1"
    retry_attempts: int = 3
    retry_delay: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary, excluding None fields."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM provider clients.

    Subclasses must implement ``generate()``. All other methods
    (structured output, batching, retry, JSON extraction, connection
    validation, model info) are shared.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a text completion for *prompt*."""
        ...

    # ------------------------------------------------------------------
    # Shared public methods
    # ------------------------------------------------------------------

    def generate_structured(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response and parse it as JSON."""
        text = self.generate(prompt, system_prompt)
        return self._extract_json(text)

    def batch_generate(
        self, prompts: List[str], system_prompt: Optional[str] = None
    ) -> List[str]:
        """Generate responses for multiple prompts sequentially."""
        responses: List[str] = []
        for prompt in prompts:
            try:
                responses.append(self.generate(prompt, system_prompt))
            except Exception as exc:
                responses.append(f"ERROR: {exc}")
        return responses

    def validate_connection(self) -> bool:
        """Send a test prompt and check for ``CONNECTION_OK`` in the reply."""
        try:
            response = self.generate("Respond with exactly: 'CONNECTION_OK'")
            return "CONNECTION_OK" in response
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Return a dictionary describing the current configuration."""
        return {
            "provider": self.config.provider,
            "model_id": self.config.model_id,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _retry_with_backoff(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call *fn* with exponential back-off, honouring retry config.

        Authentication errors (HTTP 401 / 403) are **not** retried.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.config.retry_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_error = exc
                # Do not retry auth errors
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                if status in (401, 403):
                    raise
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        raise RuntimeError(
            f"Failed after {self.config.retry_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract the first JSON object from *text*.

        Handles:
        - ````` ```json ... ``` ````` fenced blocks
        - Raw JSON
        - JSON embedded in surrounding prose (brace-matching)
        """
        # 1. Fenced code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                return json.loads(text[start:end].strip())

        # 2. Brace matching for the first complete JSON object
        if "{" in text:
            start_idx = text.index("{")
            depth = 0
            for i, ch in enumerate(text[start_idx:], start_idx):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start_idx : i + 1])

        raise RuntimeError(
            f"No valid JSON found in response: {text[:200]}..."
        )


class OpenAIClient(BaseLLMClient):
    """LLM client for the OpenAI API (``openai`` package)."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        config = config or LLMConfig(provider="openai")
        super().__init__(config)

        try:
            import openai  # noqa: F811 – lazy import
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIClient. "
                "Install it with: pip install openai"
            )

        # Resolve API key
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment "
                "variable or pass api_key to LLMConfig."
            )

        # Default model
        if config.model_id is None:
            config.model_id = "gpt-4o"

        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text via the OpenAI chat completions API."""

        def _call() -> str:
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            return response.choices[0].message.content

        return self._retry_with_backoff(_call)


class AnthropicClient(BaseLLMClient):
    """LLM client for the Anthropic API (``anthropic`` package)."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        config = config or LLMConfig(provider="anthropic")
        super().__init__(config)

        try:
            import anthropic  # noqa: F811 – lazy import
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with: pip install anthropic"
            )

        # Resolve API key
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set the ANTHROPIC_API_KEY "
                "environment variable or pass api_key to LLMConfig."
            )

        # Default model
        if config.model_id is None:
            config.model_id = "claude-sonnet-4-20250514"

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text via the Anthropic messages API."""

        def _call() -> str:
            kwargs: Dict[str, Any] = {
                "model": self.config.model_id,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)
            return response.content[0].text

        return self._retry_with_backoff(_call)


class BedrockClient(BaseLLMClient):
    """LLM client for AWS Bedrock (``boto3`` package)."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        config = config or LLMConfig(provider="bedrock")
        super().__init__(config)

        try:
            import boto3  # noqa: F811 – lazy import
        except ImportError:
            raise ImportError(
                "The 'boto3' package is required for BedrockClient. "
                "Install it with: pip install flowplan-g2p[bedrock]"
            )

        # Default model
        if config.model_id is None:
            config.model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=config.aws_region,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text via AWS Bedrock invoke_model (Anthropic format)."""

        def _call() -> str:
            body: Dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                body["system"] = system_prompt

            response = self.bedrock_client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(body),
            )
            response_body = json.loads(response["body"].read())

            if "content" in response_body and len(response_body["content"]) > 0:
                return response_body["content"][0]["text"]
            raise RuntimeError("Unexpected response format from Bedrock API")

        return self._retry_with_backoff(_call)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Instantiate the appropriate LLM client for *config.provider*.

    Raises ``ValueError`` for unsupported provider strings.
    """
    if config.provider == "openai":
        return OpenAIClient(config)
    elif config.provider == "anthropic":
        return AnthropicClient(config)
    elif config.provider == "bedrock":
        return BedrockClient(config)
    else:
        raise ValueError(
            f"Unsupported provider: '{config.provider}'. "
            f"Supported providers: openai, anthropic, bedrock"
        )
