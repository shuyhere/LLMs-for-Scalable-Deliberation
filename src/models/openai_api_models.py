import os
import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

from openai import OpenAI

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


class OpenAIClientBase:
    """Reusable parent class for OpenAI-style Chat Completions API calls.

    Features:
    - Uses the official OpenAI Python v1 client (OpenAI class)
    - Adds optional reasoning control automatically for GPT-5 model names
    - Retries transient errors with exponential backoff
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        retry_backoff_base_sec: float = 1.0,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff_base_sec = retry_backoff_base_sec
        self.reasoning_effort = reasoning_effort
        
        # Resolve credentials and endpoint from env if not provided explicitly
        # Prefer lowercased keys in user's .env, then fall back to standard uppercase
        resolved_api_key = (
            api_key
            or os.getenv("openai_key")
            or os.getenv("OPENAI_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.getenv("openai_base_url")
            or os.getenv("OPENAI_BASE_URL")
        )

        if not resolved_api_key:
            raise ValueError("Missing API key. Provide api_key or set OPENAI_API_KEY/openai_key.")

        # Instantiate client
        if resolved_base_url:
            self.client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key, default_headers=extra_headers)
        else:
            self.client = OpenAI(api_key=resolved_api_key, default_headers=extra_headers)

    def _is_gpt5_model(self) -> bool:
        try:
            name = str(self.model).lower()
            return ("gpt-5" in name) or ("gpt5" in name)
        except Exception:
            return False
    def response(self, instructions: str, input: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                extra: Dict[str, Any] = {}
                if self.reasoning_effort and self._is_gpt5_model():
                    # Only GPT-5 family supports reasoning control on Responses API
                    extra["reasoning"] = {"effort": self.reasoning_effort}

                create_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "instructions": instructions,
                    "input": input,
                    "temperature": self.temperature,
                }
                if self.top_p is not None:
                    create_kwargs["top_p"] = self.top_p
                if self.max_tokens is not None:
                    create_kwargs["max_output_tokens"] = self.max_tokens

                create_kwargs.update(extra)

                resp = self.client.responses.create(**create_kwargs)

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
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_sec = self.retry_backoff_base_sec * (2 ** (attempt - 1))
                time.sleep(sleep_sec)
        if last_err:
            raise last_err
        return ""
    def chat_completion(self, system_prompt: str, input: str) -> str:
        create_kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": input}],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        
        # Only add reasoning_effort for GPT-5 models
        if self.reasoning_effort and self._is_gpt5_model():
            create_kwargs["reasoning_effort"] = self.reasoning_effort
            
        response = self.client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content
    
if __name__ == "__main__":
    client = OpenAIClientBase(model="gpt-5-mini", reasoning_effort="high")
    # print(client.response(instructions="You are a helpful assistant not having any knowledge about the world.", input="Write a one-sentence bedtime story about a unicorn."))
    print(client.chat_completion(system_prompt="You are a helpful assistant not having any knowledge about the world.", input="Write a one-sentence bedtime story about a unicorn."))
    