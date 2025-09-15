#!/usr/bin/env python3
"""
Example usage of vLLM client with LanguageModel interface
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.LanguageModel import LanguageModel


def main():
    """Example usage of vLLM client."""
    
    # Example 1: Using LanguageModel with vLLM (automatic detection)
    print("=== Example 1: Automatic vLLM detection ===")
    
    # This will automatically detect as vLLM because it contains "trained"
    model_path = "/path/to/your/trained/model"
    model = LanguageModel(
        model_name="my-trained-model",
        model_path=model_path,  # Pass model_path as a parameter
        temperature=0.7,
        max_tokens=1024
    )
    
    print(f"Provider: {model.get_provider_info()}")
    
    # Example 2: Direct vLLM client usage
    print("\n=== Example 2: Direct vLLM client ===")
    
    from models.vllm_client import VLLMClient
    
    client = VLLMClient(
        model_name="my-trained-model",
        model_path=model_path,
        temperature=0.7,
        max_tokens=1024,
        vllm_api_port=8001,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75
    )
    
    # Use as context manager (recommended)
    with client:
        response = client.get_response("Hello, how are you?")
        print(f"Response: {response}")
    
    # Example 3: Using with different model types
    print("\n=== Example 3: Different model types ===")
    
    # For reward models
    reward_model = LanguageModel(
        model_name="reward-model",
        model_path="/path/to/reward/model",
        temperature=0.0  # Low temperature for reward models
    )
    
    # For SFT models
    sft_model = LanguageModel(
        model_name="sft-model",
        model_path="/path/to/sft/model",
        temperature=0.7
    )
    
    print(f"Reward model provider: {reward_model.get_provider_info()}")
    print(f"SFT model provider: {sft_model.get_provider_info()}")
    
    # Example 4: Chat completion
    print("\n=== Example 4: Chat completion ===")
    
    with client:
        system_prompt = "You are a helpful assistant."
        user_input = "What is the capital of France?"
        
        response = client.chat_completion(system_prompt, user_input)
        print(f"Chat response: {response}")
    
    # Example 5: Batch processing with semaphore
    print("\n=== Example 5: Batch processing ===")
    
    import asyncio
    
    async def batch_inference():
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
        
        messages_list = [
            [{"role": "user", "content": f"Question {i}: What is {i} + {i}?"}]
            for i in range(10)
        ]
        
        tasks = [
            client.process_with_semaphore(semaphore, messages, temperature=0.0)
            for messages in messages_list
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, (response, reasoning) in enumerate(results):
            print(f"Question {i}: {response}")
    
    # Run batch inference
    with client:
        asyncio.run(batch_inference())


if __name__ == "__main__":
    main()
