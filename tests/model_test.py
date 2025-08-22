#!/usr/bin/env python3
"""
Basic API calling tests for GPT-4o and Qwen-Plus models
"""

import os
import sys

# Add src path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.LanguageModel import LanguageModel


def test_gpt4o():
    """Test GPT-4o model"""
    print("=== Testing GPT-4o ===")
    
    try:
        model = LanguageModel("gpt-4o")
        print(f"Model initialized successfully: {model.get_provider_info()}")
        
        # Test get_response
        print("\n1. Testing get_response:")
        response = model.get_response("Say hello", max_tokens=10)
        print(f"   Response: {response}")
        
        # Test get_chat_response
        print("\n2. Testing get_chat_response:")
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Say hello"}
        ]
        response = model.get_chat_response(messages, max_tokens=10)
        print(f"   Response: {response}")
        
    except Exception as e:
        print(f"GPT-4o test failed: {e}")


def test_qwen():
    """Test Qwen-Plus model"""
    print("\n=== Testing Qwen-Plus ===")
    
    try:
        model = LanguageModel("qwen-plus")
        print(f"Model initialized successfully: {model.get_provider_info()}")
        
        # Test get_response
        print("\n1. Testing get_response:")
        response = model.get_response("Say hello", max_tokens=10)
        print(f"   Response: {response}")
        
        # Test get_chat_response
        print("\n2. Testing get_chat_response:")
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Say hello"}
        ]
        response = model.get_chat_response(messages, max_tokens=10)
        print(f"   Response: {response}")
        
    except Exception as e:
        print(f"Qwen-Plus test failed: {e}")


if __name__ == "__main__":
    print("Basic API Calling Tests")
    print("Required environment variables:")
    print("- OPENAI_API_KEY (for GPT-4o)")
    print("- ZHI_API_KEY (for Qwen-Plus)")
    print("-" * 40)
    
    test_gpt4o()
    test_qwen()
    
    print("\nTests completed!")
