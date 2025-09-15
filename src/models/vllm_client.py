#!/usr/bin/env python3
"""
vLLM Client for local model inference
Compatible with LanguageModel interface for running locally trained models
"""

import os
import json
import time
import signal
import subprocess
import logging
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from openai import OpenAI
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}. Please install torch and openai.") from e

logger = logging.getLogger(__name__)


class VLLMClient:
    """vLLM client for local model inference with LanguageModel compatibility."""
    
    def __init__(
        self,
        model_name: str,
        model_path: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        max_retries: int = 3,
        retry_backoff_base_sec: float = 1.0,
        vllm_api_port: int = 8001,
        tensor_parallel_size: Optional[int] = None,
        gpu_memory_utilization: float = 0.75,
        max_model_len: Optional[int] = None,
        quantization: bool = False,
        load_lora: bool = False,
        lora_path: Optional[str] = None,
        trust_remote_code: bool = True,
        verbose: bool = True,
        **_kwargs
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens or 2048
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_backoff_base_sec = retry_backoff_base_sec
        self.vllm_api_port = vllm_api_port
        self.tensor_parallel_size = tensor_parallel_size or torch.cuda.device_count()
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.load_lora = load_lora
        self.lora_path = lora_path
        self.trust_remote_code = trust_remote_code
        self.verbose = verbose
        
        # Server process
        self.server = None
        self.client = None
        self._is_initialized = False
        
        logger.info("VLLMClient initialized for model: %s at %s", model_name, model_path)
    
    def _get_model_max_len(self) -> int:
        """Get model max length from config.json."""
        config_path = os.path.join(self.model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            model_max_len = model_config.get('max_position_embeddings', 8192)
            # Cap the max length to save CUDA memory
            model_max_len = min(model_max_len, 8192)
            return model_max_len
        return 8192
    
    def _build_server_command(self) -> List[str]:
        """Build vLLM server command."""
        max_model_len = str(self.max_model_len or self._get_model_max_len())
        
        server_command = [
            'vllm',
            'serve',
            self.model_path,
            '--served-model-name', self.model_name,
            '--task', 'generate',
            '--port', str(self.vllm_api_port),
            '--api-key', 'anonymous123',
            '--max-model-len', max_model_len,
            '--tensor-parallel-size', str(self.tensor_parallel_size),
            '--gpu-memory-utilization', str(self.gpu_memory_utilization),
            '--disable-log-requests',
            '--trust-remote-code' if self.trust_remote_code else '',
        ]
        
        # Remove empty strings
        server_command = [arg for arg in server_command if arg]
        
        # Enable reasoning for Qwen3 models
        if 'qwen3' in self.model_name.lower():
            server_command.extend([
                '--reasoning-parser', 'deepseek_r1',
            ])
        
        # Large models configuration
        large_models = [
            'qwen2.5-32b', 'qwen3-32b', 'qwen3-30b-a3b', 'qwen3-30b-a3b-no-reasoning',
            'llama3-70b', 'llama3-70b-base', 'qwen3-32b-no-reasoning',
            'gemma3-27b', 'gemma3-27b-base'
        ]
        
        if self.model_name in large_models:
            server_command.extend([
                '--enforce-eager',
            ])
            # Reduce max length for large models
            server_command[server_command.index('--max-model-len') + 1] = '3000'
            server_command[server_command.index('--gpu-memory-utilization') + 1] = '0.9'
        
        # Phi3 models
        if 'phi3' in self.model_name.lower():
            server_command.extend([
                '--enforce-eager',
            ])
        
        # LoRA support
        if self.load_lora and self.lora_path:
            server_command.extend([
                '--enable-lora',
                '--enable-lora-bias',
                '--max-lora-rank', '128',
                '--lora-modules', f'{{"name": "{self.model_name}", "path": "{self.lora_path}"}}'
            ])
        
        # Quantization
        if self.quantization:
            server_command.extend([
                '--quantization', 'bitsandbytes',
            ])
        
        return server_command
    
    def init_llm(self) -> None:
        """Initialize vLLM server and client."""
        if self._is_initialized:
            logger.warning("vLLM client already initialized")
            return
        
        logger.info("Starting vLLM server...")
        
        # Build server command
        server_command = self._build_server_command()
        logger.info("Server command: %s", ' '.join(server_command))
        
        # Set environment variables
        env = os.environ.copy()
        env['VLLM_API_KEY'] = 'anonymous123'
        
        # Start server process
        self.server = subprocess.Popen(
            args=server_command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, 
            universal_newlines=True,
            bufsize=1  
        )
        
        # Start a thread to monitor server output
        import threading
        
        def monitor_server_output():
            """Monitor server output in real-time"""
            try:
                for line in iter(self.server.stdout.readline, ''):
                    if line and self.verbose:
                        # Filter out noise, only show important information
                        line = line.strip()
                        if any(keyword in line.lower() for keyword in [
                            'loading', 'loaded', 'ready', 'started', 'error', 'warning', 
                            'model', 'gpu', 'memory', 'tensor', 'vllm', 'server'
                        ]):
                            print(f"[vLLM] {line}")
            except Exception as e:
                logger.debug("Server output monitoring error: %s", e)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_server_output, daemon=True)
        monitor_thread.start()
        
        # Wait for server to start
        server_running = False
        max_attempts = 120  # 20 minutes for large models
        attempt = 0
        
        logger.info("Waiting for vLLM server to start...")
        if self.verbose:
            print(f"[INFO] Starting vLLM server, waiting up to {max_attempts * 10} seconds...")
        
        while not server_running and attempt < max_attempts:
            time.sleep(10)
            attempt += 1
            
            # Show progress
            if self.verbose and attempt % 6 == 0:  # Show progress every minute
                print(f"[INFO] Waited {attempt * 10} seconds, continuing to wait for server startup...")
            
            try:
                result = subprocess.run(
                    [
                        "curl", f"http://localhost:{self.vllm_api_port}/v1/models",
                        "--header", "Authorization: Bearer anonymous123",
                        "--connect-timeout", "5"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False
                )
                
                if self.model_name in result.stdout:
                    server_running = True
                    logger.info("vLLM server started successfully")
                    if self.verbose:
                        print(f"[SUCCESS] vLLM server started successfully! (took {attempt * 10} seconds)")
                else:
                    if self.verbose and attempt % 12 == 0:  # Show check status every 2 minutes
                        print(f"[DEBUG] Checking server... (attempt {attempt}/{max_attempts})")
                    
            except subprocess.TimeoutExpired:
                if self.verbose and attempt % 12 == 0:
                    print(f"[DEBUG] Server check timeout... (attempt {attempt}/{max_attempts})")
            except Exception as e:
                if self.verbose and attempt % 12 == 0:
                    print(f"[DEBUG] Server check failed: {e} (attempt {attempt}/{max_attempts})")
        
        if not server_running:
            self.kill_server()
            raise RuntimeError("vLLM server failed to start after maximum attempts")
        
        # Create sync client
        self.client = OpenAI(
            api_key='anonymous123',
            base_url=f'http://localhost:{self.vllm_api_port}/v1',
        )
        
        self._is_initialized = True
        logger.info("vLLM client initialized successfully")
    
    def kill_server(self) -> None:
        """Kill the vLLM server process."""
        if self.server:
            logger.info("Stopping vLLM server...")
            self.server.send_signal(signal.SIGINT)
            try:
                self.server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Server didn't stop gracefully, forcing termination")
                self.server.kill()
                self.server.wait()
            self.server = None
            time.sleep(5)  # Wait for cleanup
            logger.info("vLLM server stopped")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.kill_server()
    
    def __enter__(self):
        """Context manager entry."""
        if not self._is_initialized:
            self.init_llm()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.kill_server()
    
    # LanguageModel compatible methods
    
    def get_response(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Get a response from the language model with a single prompt."""
        try:
            if not self._is_initialized:
                self.init_llm()
            
            messages = [{"role": "user", "content": prompt}]
            return self._chat_completion(messages, temperature, max_tokens)
        except Exception as e:
            logger.error("get_response failed: %s", e)
            return None
    
    def get_chat_response(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Get a response from the language model using chat format."""
        try:
            if not self._is_initialized:
                self.init_llm()
            
            return self._chat_completion(messages, temperature, max_tokens)
        except Exception as e:
            logger.error("get_chat_response failed: %s", e)
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
        return self.chat_completion(instructions, input_text, temperature, max_tokens)
    
    # Internal methods
    
    def _chat_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """Sync chat completion."""
        if not self.client:
            raise RuntimeError("vLLM client not initialized")
        
        try:
            # Add generation prompt for non-GPT models
            extra_body = {}
            if 'gpt' not in self.model_name.lower():
                extra_body['add_generation_prompt'] = True
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=self.top_p,
                extra_body=extra_body
            )
            
            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                logger.warning("Empty response from vLLM server")
                return None
            
        except Exception as e:
            logger.error("Chat completion failed: %s", e)
            return None
    
    def process_with_semaphore(
        self,
        message: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **_kwargs
    ) -> Tuple[str, str]:
        """Process with semaphore for rate limiting (compatible with reference code)."""
        response = self._chat_completion(message, temperature, max_tokens)
        reasoning = ""  # vLLM doesn't separate reasoning content
        return response or "", reasoning
    
    # Utility methods
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider and model."""
        return {
            "provider": "vllm",
            "model": self.model_name,
            "model_path": self.model_path,
            "client_type": "VLLMClient",
            "initialized": self._is_initialized
        }
    
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._is_initialized


# Factory function for LanguageModel compatibility
def create_vllm_client(
    model_name: str,
    model_path: str,
    **_kwargs
) -> VLLMClient:
    """Create vLLM client with LanguageModel-like interface."""
    return VLLMClient(model_name=model_name, model_path=model_path, **_kwargs)


# Example usage
if __name__ == "__main__":
    # Example usage
    example_model_path = "/path/to/your/trained/model"
    example_model_name = "my-trained-model"
    
    # Create client
    client = VLLMClient(
        model_name=example_model_name,
        model_path=example_model_path,
        temperature=0.7,
        max_tokens=1024
    )
    
    # Use as context manager (recommended)
    with client:
        response = client.get_response("Hello, how are you?")
        print(f"Response: {response}")
    
    # Or initialize manually
    # client.init_llm()
    # response = client.get_response("Hello, how are you?")
    # client.kill_server()
