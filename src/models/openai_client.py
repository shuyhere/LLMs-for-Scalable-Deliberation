import os
import time
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI, OpenAIError

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except ImportError:
    pass

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Enhanced OpenAI client with GPT-5 reasoning control and robust error handling."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_backoff_base_sec: float = 1.0,
        reasoning_effort: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff_base_sec = retry_backoff_base_sec
        self.reasoning_effort = reasoning_effort
        
        self._setup_client(api_key, base_url, extra_headers)
    
    def _setup_client(self, api_key: Optional[str] = None, base_url: Optional[str] = None, extra_headers: Optional[Dict[str, str]] = None):
        """Setup OpenAI client with credentials resolution."""
        # Resolve credentials from multiple sources
        resolved_api_key = (
            api_key
            or os.getenv("openai_key")  # Prefer lowercase from .env
            or os.getenv("OPENAI_API_KEY")  # Standard uppercase
        )
        resolved_base_url = (
            base_url
            or os.getenv("openai_base_url")
            or os.getenv("OPENAI_BASE_URL")
        )
        
        if not resolved_api_key:
            raise ValueError("Missing OpenAI API key. Provide api_key or set OPENAI_API_KEY/openai_key.")
        
        # Initialize client
        client_kwargs = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
            
        self.client = OpenAI(**client_kwargs)
    
    def _is_gpt5_model(self) -> bool:
        """Check if model is GPT-5 family for reasoning control."""
        try:
            name = str(self.model).lower()
            return ("gpt-5" in name) or ("gpt5" in name)
        except Exception:
            return False
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Make chat completion request with enhanced features."""
        effective_temperature = temperature if temperature is not None else self.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                create_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": effective_temperature,
                }
                
                # Add optional parameters
                if self.top_p is not None:
                    create_kwargs["top_p"] = self.top_p
                if effective_max_tokens is not None:
                    create_kwargs["max_tokens"] = effective_max_tokens
                
                # Add GPT-5 reasoning effort
                if self.reasoning_effort and self._is_gpt5_model():
                    create_kwargs["reasoning_effort"] = self.reasoning_effort
                
                response = self.client.chat.completions.create(**create_kwargs)
                
                # Validate response
                if not response or not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("Invalid API response: missing choices")
                if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                    raise ValueError("Invalid API response: missing message content")
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_err = e
                logger.error(f"OpenAI attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    sleep_time = self.retry_backoff_base_sec * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
        
        if last_err:
            raise last_err
        return ""
    
    def response(self, instructions: str, input_text: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """OpenAI Responses API for GPT-5 models, fallback to chat completion for others."""
        if self._is_gpt5_model():
            return self._make_responses_call(instructions, input_text, temperature, max_tokens)
        else:
            # Fallback to chat completion
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text}
            ]
            return self.chat_completion(messages, temperature, max_tokens)
    
    def _make_responses_call(self, instructions: str, input_text: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Make call to OpenAI Responses API for GPT-5 models."""
        effective_temperature = temperature if temperature is not None else self.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        extra: Dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning"] = {"effort": self.reasoning_effort}

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "temperature": effective_temperature,
        }
        if self.top_p is not None:
            create_kwargs["top_p"] = self.top_p
        if effective_max_tokens is not None:
            create_kwargs["max_output_tokens"] = effective_max_tokens

        create_kwargs.update(extra)

        resp = self.client.responses.create(**create_kwargs)

        # Extract text from response
        text = getattr(resp, "output_text", None) or getattr(resp, "output", None)
        if isinstance(text, str):
            return text

        # Fallback: aggregate from content parts
        if hasattr(resp, "content"):
            pieces: List[str] = []
            try:
                for part in resp.content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str):
                        pieces.append(t)
            except Exception:
                pass
            if pieces:
                return "".join(pieces)

        return ""


if __name__ == "__main__":
    # Test OpenAI client
    try:
        client = OpenAIClient("gpt-3.5-turbo")
        print("OpenAI client initialized successfully")
        
        # Test with GPT-5 if available
        gpt5_client = OpenAIClient("gpt-5-mini", reasoning_effort="medium")
        print("GPT-5 client with reasoning initialized")
    except Exception as e:
        print(f"OpenAI client test failed: {e}")
