#!/usr/bin/env python3
"""
Test script for loading and using trained models with vLLM client
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.LanguageModel import LanguageModel


def test_trained_model():
    """Test loading and using a trained model"""
    
    # Path to trained model
    model_path = "/ibex/project/c2328/sft_tools/LLaMA-Factory/saves/qwen3_4b/full/deliberation_sft_compair/checkpoint-99"
    
    print("🚀 Testing trained model loading")
    print(f"Model path: {model_path}")
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    print("✅ Model path exists")
    
    # Create model instance with verbose output
    print("\n=== Creating model instance ===")
    model = LanguageModel(
        model_name="qwen3-4b-deliberation-sft",
        model_path=model_path,
        temperature=0.7,
        max_tokens=512,
        verbose=True  # Enable verbose output
    )
    
    print(f"✅ Model created successfully, Provider: {model.get_provider_info()['provider']}")
    
    # Test inference
    test_prompt = "Hello, please introduce yourself briefly."
    print(f"\n=== Starting inference test ===")
    print(f"Test question: {test_prompt}")
    
    try:
        # Use context manager
        with model.client as client:
            print("✅ vLLM server started successfully")
            print("🤔 Generating response...")
            
            response = client.get_response(test_prompt)
            print(f"✅ Response: {response}")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    
    print("\n🎉 Test successful!")
    return True


def test_quiet_mode():
    """Test quiet mode (no verbose output)"""
    
    model_path = "/ibex/project/c2328/sft_tools/LLaMA-Factory/saves/qwen3_4b/full/deliberation_sft_compair/checkpoint-99"
    
    print("\n🔇 Testing quiet mode")
    
    # Create model instance with quiet output
    print("\n=== Creating model instance (quiet mode) ===")
    model = LanguageModel(
        model_name="qwen3-4b-deliberation-sft",
        model_path=model_path,
        temperature=0.7,
        max_tokens=512,
        verbose=False  # Disable verbose output
    )
    
    print(f"✅ Model created successfully, Provider: {model.get_provider_info()['provider']}")
    
    # Test inference
    test_prompt = "What is artificial intelligence?"
    print(f"\n=== Starting inference test ===")
    print(f"Test question: {test_prompt}")
    
    try:
        # Use context manager
        with model.client as client:
            print("✅ vLLM server started successfully (quiet mode)")
            print("🤔 Generating response...")
            
            response = client.get_response(test_prompt)
            print(f"✅ Response: {response}")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False
    
    print("\n🎉 Quiet mode test successful!")
    return True


def test_chat_completion():
    """Test chat completion functionality"""
    
    model_path = "/ibex/project/c2328/sft_tools/LLaMA-Factory/saves/qwen3_4b/full/deliberation_sft_compair/checkpoint-99"
    
    print("\n💬 Testing chat completion")
    
    model = LanguageModel(
        model_name="qwen3-4b-deliberation-sft",
        model_path=model_path,
        temperature=0.7,
        verbose=False
    )
    
    try:
        with model.client as client:
            # Test system + user input
            response = client.chat_completion(
                system_prompt="You are a helpful AI assistant.",
                input_text="Explain machine learning in simple terms."
            )
            print(f"✅ Chat completion response: {response}")
            
            # Test multi-turn conversation
            messages = [
                {"role": "system", "content": "You are a professional AI assistant."},
                {"role": "user", "content": "What is deep learning?"},
                {"role": "assistant", "content": "Deep learning is a subset of machine learning..."},
                {"role": "user", "content": "How does it differ from traditional machine learning?"}
            ]
            
            response = client.get_chat_response(messages)
            print(f"✅ Multi-turn conversation response: {response}")
            
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False
    
    print("\n🎉 Chat completion test successful!")
    return True


if __name__ == "__main__":
    print("🧪 Starting model testing...")
    
    # Test verbose mode
    success1 = test_trained_model()
    
    # Test quiet mode
    success2 = test_quiet_mode()
    
    # Test chat completion
    success3 = test_chat_completion()
    
    if success1 and success2 and success3:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
