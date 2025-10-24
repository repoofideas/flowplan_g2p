"""
AWS Bedrock Llama integration for FlowPlan-G2P.
Handles communication with Llama 4 via AWS Bedrock service.
"""

import boto3
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    max_gen_len: int = 2048
    temperature: float = 0.2
    # top_p: float = 0.9
    retry_attempts: int = 3
    retry_delay: float = 1.0


class DeepSeekClient:
    """
    AWS Bedrock client for DeepSeek integration.
    Handles API calls, retries, and error management.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, region_name: str = "us-east-1"):
        """
        Initialize Bedrock client.
        
        Args:
            config: LLM configuration settings
            region_name: AWS region for Bedrock service
        """
        self.config = config or LLMConfig()
        self.region_name = region_name
        
        # Initialize Bedrock runtime client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")
    
    def _prepare_request_body(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Prepare the request body for DeepSeek API call.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            JSON string for API request
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        body = {
            "messages": messages,
            "max_tokens": self.config.max_gen_len,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
            
        return json.dumps(body)
    
    def _make_request(self, request_body: str) -> Dict[str, Any]:
        """
        Make API request to Bedrock with retry logic.
        
        Args:
            request_body: JSON request body
            
        Returns:
            API response dictionary
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=self.config.model_id,
                    body=request_body
                )
                
                response_body = json.loads(response['body'].read())
                return response_body
                
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                continue
        
        raise RuntimeError(f"Failed to get response after {self.config.retry_attempts} attempts: {last_error}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using DeepSeek model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        request_body = self._prepare_request_body(prompt, system_prompt)
        response = self._make_request(request_body)
        
        # Extract text from DeepSeek response
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        else:
            raise RuntimeError("Unexpected response format from Bedrock API")
    
    def generate_structured(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate structured response (expecting JSON) using DeepSeek model.
        
        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response as dictionary
        """
        response_text = self.generate(prompt, system_prompt)
        
        # Try to extract JSON from response
        try:
            # Look for JSON content between code blocks or raw JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_content = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                # Try to find the first complete JSON object
                start_idx = response_text.find("{")
                brace_count = 0
                end_idx = start_idx
                
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_content = response_text[start_idx:end_idx]
            else:
                raise ValueError("No JSON content found in response")
            
            return json.loads(json_content)
            
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse JSON from response: {e}. Response: {response_text[:200]}...")
    
    def batch_generate(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """
        Generate responses for multiple prompts sequentially.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (applied to all)
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, system_prompt)
                responses.append(response)
            except Exception as e:
                # Continue with other prompts even if one fails
                responses.append(f"ERROR: {str(e)}")
        
        return responses
    
    def validate_connection(self) -> bool:
        """
        Test connection to Bedrock service.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_prompt = "Respond with exactly: 'CONNECTION_OK'"
            response = self.generate(test_prompt)
            return "CONNECTION_OK" in response
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            "model_id": self.config.model_id,
            "max_gen_len": self.config.max_gen_len,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "region": self.region_name
        }


class AnthropicClient:
    """
    AWS Bedrock client for Anthropic Claude integration.
    Handles API calls, retries, and error management.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, region_name: str = "us-east-1"):
        """
        Initialize Bedrock client.
        
        Args:
            config: LLM configuration settings
            region_name: AWS region for Bedrock service
        """
        self.config = config or LLMConfig()
        self.region_name = region_name
        
        # Initialize Bedrock runtime client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")
    
    def _prepare_request_body(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Prepare the request body for Anthropic API call.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            JSON string for API request
        """
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_gen_len,
            "messages": messages
        }
        
        if system_prompt:
            body["system"] = system_prompt
            
        return json.dumps(body)
    
    def _make_request(self, request_body: str) -> Dict[str, Any]:
        """
        Make API request to Bedrock with retry logic.
        
        Args:
            request_body: JSON request body
            
        Returns:
            API response dictionary
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=self.config.model_id,
                    body=request_body
                )
                
                response_body = json.loads(response['body'].read())
                return response_body
                
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                continue
        
        raise RuntimeError(f"Failed to get response after {self.config.retry_attempts} attempts: {last_error}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Anthropic model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        request_body = self._prepare_request_body(prompt, system_prompt)
        response = self._make_request(request_body)
        
        # Extract text from Anthropic response
        if "content" in response and len(response["content"]) > 0:
            return response["content"][0]["text"]
        else:
            raise RuntimeError("Unexpected response format from Bedrock API")
    
    def generate_structured(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate structured response (expecting JSON) using Anthropic model.
        
        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response as dictionary
        """
        response_text = self.generate(prompt, system_prompt)
        
        # Try to extract JSON from response
        try:
            # Look for JSON content between code blocks or raw JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_content = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                # Try to find the first complete JSON object
                start_idx = response_text.find("{")
                brace_count = 0
                end_idx = start_idx
                
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_content = response_text[start_idx:end_idx]
            else:
                raise ValueError("No JSON content found in response")
            
            return json.loads(json_content)
            
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse JSON from response: {e}. Response: {response_text[:200]}...")
    
    def batch_generate(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """
        Generate responses for multiple prompts sequentially.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (applied to all)
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, system_prompt)
                responses.append(response)
            except Exception as e:
                # Continue with other prompts even if one fails
                responses.append(f"ERROR: {str(e)}")
        
        return responses
    
    def validate_connection(self) -> bool:
        """
        Test connection to Bedrock service.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_prompt = "Respond with exactly: 'CONNECTION_OK'"
            response = self.generate(test_prompt)
            return "CONNECTION_OK" in response
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            "model_id": self.config.model_id,
            "max_gen_len": self.config.max_gen_len,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "region": self.region_name
        }


class BedrockLlamaClient:
    """
    AWS Bedrock client for Llama 4 integration.
    Handles API calls, retries, and error management.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, region_name: str = "us-west-2"):
        """
        Initialize Bedrock client.
        
        Args:
            config: LLM configuration settings
            region_name: AWS region for Bedrock service
        """
        self.config = config or LLMConfig()
        self.region_name = region_name
        
        # Initialize Bedrock runtime client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")
    
    def _prepare_request_body(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Prepare the request body for Llama API call.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            JSON string for API request
        """
        # Format prompt using Llama's instruction template
        if system_prompt:
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        body = {
            "prompt": formatted_prompt,
            "max_gen_len": self.config.max_gen_len,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
            
        return json.dumps(body)
    
    def _make_request(self, request_body: str) -> Dict[str, Any]:
        """
        Make API request to Bedrock with retry logic.
        
        Args:
            request_body: JSON request body
            
        Returns:
            API response dictionary
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.bedrock_client.invoke_model(
                    modelId=self.config.model_id,
                    body=request_body
                )
                
                response_body = json.loads(response['body'].read())
                return response_body
                
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                continue
        
        raise RuntimeError(f"Failed to get response after {self.config.retry_attempts} attempts: {last_error}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate text using Llama model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
        """
        request_body = self._prepare_request_body(prompt, system_prompt)
        response = self._make_request(request_body)
        
        # Extract text from Llama response
        if "generation" in response:
            return response["generation"]
        else:
            raise RuntimeError("Unexpected response format from Bedrock API")
    
    def generate_structured(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate structured response (expecting JSON) using Llama model.
        
        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response as dictionary
        """
        response_text = self.generate(prompt, system_prompt)
        
        # Try to extract JSON from response
        try:
            # Look for JSON content between code blocks or raw JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_content = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                # Try to find the first complete JSON object
                start_idx = response_text.find("{")
                brace_count = 0
                end_idx = start_idx
                
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_content = response_text[start_idx:end_idx]
            else:
                raise ValueError("No JSON content found in response")
            
            return json.loads(json_content)
            
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse JSON from response: {e}. Response: {response_text[:200]}...")
    
    def batch_generate(self, prompts: List[str], system_prompt: Optional[str] = None) -> List[str]:
        """
        Generate responses for multiple prompts sequentially.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt (applied to all)
            
        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, system_prompt)
                responses.append(response)
            except Exception as e:
                # Continue with other prompts even if one fails
                responses.append(f"ERROR: {str(e)}")
        
        return responses
    
    def validate_connection(self) -> bool:
        """
        Test connection to Bedrock service.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_prompt = "Respond with exactly: 'CONNECTION_OK'"
            response = self.generate(test_prompt)
            return "CONNECTION_OK" in response
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary with model configuration details
        """
        return {
            "model_id": self.config.model_id,
            "max_gen_len": self.config.max_gen_len,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "region": self.region_name
        }