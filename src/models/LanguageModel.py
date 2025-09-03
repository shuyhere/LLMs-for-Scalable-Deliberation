import os
import logging
from typing import Dict, List, Optional, Union

try:
    from .openai_client import OpenAIClient
    from .zhizengzeng_client import ZhizengzengClient
except ImportError:
    # Fallback for direct execution
    from openai_client import OpenAIClient
    from zhizengzeng_client import ZhizengzengClient

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZHI_API_KEY = os.getenv("ZHI_API_KEY", "").split(",") if os.getenv("ZHI_API_KEY") else []

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageModel:
    """Unified language model router supporting OpenAI and Zhizengzeng providers.
    
    This class acts as a router that delegates requests to appropriate provider clients
    based on the model name. It provides a clean, unified interface while keeping
    provider-specific logic separated.
    
    Supported Providers:
    - OpenAI: GPT models with enhanced features (GPT-5 reasoning control)
    - Zhizengzeng: Qwen, Llama models with multi-key rotation
    
    Features:
    - Automatic provider detection from model name
    - Unified API across all providers
    - Enhanced error handling and retry mechanisms
    - Backward compatibility with existing code
    
    Example:
        # OpenAI usage
        model = LanguageModel("gpt-4")
        response = model.get_response("Hello, world!")
        
        # Advanced OpenAI with reasoning
        model = LanguageModel("gpt-5-mini", reasoning_effort="high")
        response = model.chat_completion("You are helpful", "Explain quantum computing")
        
        # Zhizengzeng usage
        model = LanguageModel("qwen-plus")
        response = model.get_response("你好，世界！")
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        api_key: Optional[Union[str, List[str]]] = None,
        base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_backoff_base_sec: float = 1.0,
        reasoning_effort: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_backoff_base_sec = retry_backoff_base_sec
        self.reasoning_effort = reasoning_effort
        
        # Determine provider and setup client
        self.provider = self._detect_provider(model_name)
        self.client = self._create_client(
            api_key=api_key,
            base_url=base_url,
            request_timeout=request_timeout,
            extra_headers=extra_headers,
            **kwargs
        )
        
        logger.info(f"LanguageModel initialized with {self.provider} provider for model: {model_name}")
    
    def _detect_provider(self, model_name: str) -> str:
        """Detect provider based on model name."""
        model_lower = model_name.lower()
        
        if self._is_zhizengzeng_model(model_lower):
            return "zhizengzeng"
        else:
            return "openai"
    
    def _is_openai_model(self, model_name: str) -> bool:
        """Check if model is OpenAI."""
        return any(keyword in model_name for keyword in [
            "gpt", "o1", "gpt-5", "gpt5"
        ]) or (model_name.count("o") > 0 and any(c.isdigit() for c in model_name[model_name.index("o")+1:]))
    
    def _is_zhizengzeng_model(self, model_name: str) -> bool:
        """Check if model is Zhizengzeng."""
        return any(keyword in model_name for keyword in [
            "qwen", "llama", "deepseek", "gemini"
        ])
    
    def _create_client(self, **kwargs):
        """Create appropriate client based on provider."""
        if self.provider == "openai":
            return self._create_openai_client(**kwargs)
        elif self.provider == "zhizengzeng":
            return self._create_zhizengzeng_client(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _create_openai_client(self, api_key=None, base_url=None, extra_headers=None, **kwargs):
        """Create OpenAI client."""
        # Use provided key or fallback to config/env
        resolved_key = api_key or OPENAI_API_KEY
        
        return OpenAIClient(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            api_key=resolved_key,
            base_url=base_url,
            extra_headers=extra_headers,
            max_retries=self.max_retries,
            retry_backoff_base_sec=self.retry_backoff_base_sec,
            reasoning_effort=self.reasoning_effort,
        )
    
    def _create_zhizengzeng_client(self, api_key=None, base_url=None, **kwargs):
        """Create Zhizengzeng client."""
        # Use provided keys or fallback to config/env
        resolved_keys = api_key or ZHI_API_KEY
        resolved_base_url = base_url or "https://api.zhizengzeng.com/v1/chat/completions"
        
        return ZhizengzengClient(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            api_keys=resolved_keys,
            base_url=resolved_base_url,
            max_retries=self.max_retries,
            retry_backoff_base_sec=self.retry_backoff_base_sec,
        )
    
    # Public API methods - unified interface
    
    def get_response(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Get a response from the language model with a single prompt."""
        try:
            if self.provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                return self.client.chat_completion(messages, temperature, max_tokens)
            elif self.provider == "zhizengzeng":
                return self.client.get_response(prompt, temperature, max_tokens)
        except Exception as e:
            logger.error(f"get_response failed: {e}")
            return None
    
    def get_chat_response(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Get a response from the language model using chat format."""
        try:
            return self.client.chat_completion(messages, temperature, max_tokens)
        except Exception as e:
            logger.error(f"get_chat_response failed: {e}")
            return None
    
    def chat_completion(self, system_prompt: str, input_text: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Chat completion with system prompt and user input."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        return self.get_chat_response(messages, temperature, max_tokens)
    
    def response(self, instructions: str, input_text: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Response method compatible with openai_api_models.py interface."""
        try:
            if self.provider == "openai":
                # Use OpenAI's response method (supports GPT-5 Responses API)
                return self.client.response(instructions, input_text, temperature, max_tokens)
            else:
                # Fallback to chat completion for other providers
                return self.chat_completion(instructions, input_text, temperature, max_tokens)
        except Exception as e:
            logger.error(f"response failed: {e}")
            return None
    
    # Utility methods
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider and model."""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "client_type": type(self.client).__name__
        }
    
    def is_openai(self) -> bool:
        """Check if using OpenAI provider."""
        return self.provider == "openai"
    
    def is_zhizengzeng(self) -> bool:
        """Check if using Zhizengzeng provider."""
        return self.provider == "zhizengzeng"
    
    def supports_reasoning(self) -> bool:
        """Check if current model supports reasoning control."""
        return self.provider == "openai" and hasattr(self.client, '_is_gpt5_model') and self.client._is_gpt5_model()


# Backward compatibility functions
def create_openai_client(model: str, **kwargs) -> LanguageModel:
    """Create OpenAI-compatible client (backward compatibility)."""
    return LanguageModel(model, **kwargs)

