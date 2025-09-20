import os
import time
import logging
from typing import Dict, List, Optional, Union

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)


class ZhizengzengClient:
    """Client for Zhizengzeng API with multi-key rotation and robust error handling."""
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        api_keys: Optional[Union[str, List[str]]] = None,
        base_url: str = "https://api.zhizengzeng.com/v1",
        max_retries: int = 3,
        retry_backoff_base_sec: float = 1.0,
        rate_limit_delay: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_backoff_base_sec = retry_backoff_base_sec
        self.rate_limit_delay = rate_limit_delay
        
        # Rate limiting
        self._last_request_time = 0
        self._api_key_index = 0
        
        self._setup_api_keys(api_keys)
        self._setup_client()
    
    def _setup_api_keys(self, api_keys: Optional[Union[str, List[str]]] = None):
        """Setup API keys from multiple sources."""
        if api_keys:
            if isinstance(api_keys, str):
                self.api_keys = [api_keys]
            else:
                self.api_keys = api_keys
        else:
            # Try to get from environment
            env_keys = os.getenv("ZHI_API_KEY", "")
            if env_keys:
                self.api_keys = [key.strip() for key in env_keys.split(",") if key.strip()]
            else:
                self.api_keys = []
        
        if not self.api_keys:
            raise ValueError("No Zhizengzeng API keys available. Provide api_keys or set ZHI_API_KEY.")
    
    def _setup_client(self):
        """Setup OpenAI client with current API key."""
        if not self.api_keys:
            raise ValueError("No API keys available")
        
        current_key = self.api_keys[self._api_key_index]
        self.client = OpenAI(api_key=current_key, base_url=self.base_url)
        logger.info(f"Using Zhizengzeng API key: {current_key[:8]}...")
    
    def _rotate_api_key(self):
        """Rotate to next available API key."""
        if len(self.api_keys) > 1:
            self._api_key_index = (self._api_key_index + 1) % len(self.api_keys)
            self._setup_client()
            logger.info("Rotated to next API key")
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate and format messages for Zhizengzeng API."""
        validated_messages = []
        for msg in messages:
            # Ensure valid roles
            if msg["role"] not in ["system", "user", "assistant"]:
                msg = {"role": "user", "content": msg.get("content", "")}
            # Ensure content is not None
            if "content" not in msg or msg["content"] is None:
                msg["content"] = ""
            validated_messages.append(msg)
        return validated_messages
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Make chat completion request with multi-key rotation."""
        effective_temperature = temperature if temperature is not None else self.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Validate messages
        validated_messages = self._validate_messages(messages)
        
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._enforce_rate_limit()
                
                create_kwargs = {
                    "model": self.model,
                    "messages": validated_messages,
                    "temperature": effective_temperature,
                }
                
                # Add optional parameters
                if self.top_p is not None:
                    create_kwargs["top_p"] = self.top_p
                if effective_max_tokens is not None:
                    # Try max_tokens first, fallback to max_completion_tokens
                    try:
                        create_kwargs["max_tokens"] = effective_max_tokens
                        response = self.client.chat.completions.create(**create_kwargs)
                    except Exception:
                        create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens")
                        response = self.client.chat.completions.create(**create_kwargs)
                else:
                    response = self.client.chat.completions.create(**create_kwargs)
                
                # Validate response
                logger.debug(f"API Response type: {type(response)}")
                logger.debug(f"API Response attributes: {dir(response) if response else 'None'}")
                if response:
                    logger.debug(f"API Response: {response}")
                
                if not response or not hasattr(response, 'choices') or not response.choices:
                    logger.error(f"Invalid API response: response={response}, has_choices={hasattr(response, 'choices') if response else False}, choices={getattr(response, 'choices', None) if response else None}")
                    raise ValueError("Invalid API response: missing choices")
                if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                    logger.error(f"Invalid API response: message={getattr(response.choices[0], 'message', None) if response.choices else None}")
                    raise ValueError("Invalid API response: missing message content")
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_err = e
                logger.error(f"Zhizengzeng attempt {attempt} failed: {str(e)}")
                
                # Rotate API key on failure (except for last attempt)
                if attempt < self.max_retries:
                    try:
                        self._rotate_api_key()
                    except Exception as key_err:
                        logger.error(f"Failed to rotate API key: {key_err}")
                    
                    # Exponential backoff
                    sleep_time = self.retry_backoff_base_sec * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
        
        if last_err:
            raise last_err
        return ""
    
    def get_response(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Simple prompt-response interface."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, temperature, max_tokens)
    
    def get_chat_response(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Chat with system prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.chat_completion(messages, temperature, max_tokens)
