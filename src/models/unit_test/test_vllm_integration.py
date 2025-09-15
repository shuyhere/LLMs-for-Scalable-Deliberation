#!/usr/bin/env python3
"""
Test script for vLLM integration with LanguageModel
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from models.LanguageModel import LanguageModel
        from models.vllm_client import VLLMClient
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_provider_detection():
    """Test provider detection logic."""
    print("\nTesting provider detection...")
    
    from models.LanguageModel import LanguageModel
    
    # Test cases
    test_cases = [
        ("/path/to/model", "vllm"),
        ("trained-model", "vllm"),
        ("reward-model", "vllm"),
        ("sft-model", "vllm"),
        ("gpt-4", "openai"),
        ("qwen-plus", "zhizengzeng"),
        ("llama-2", "zhizengzeng"),
    ]
    
    for model_name, expected_provider in test_cases:
        # Create a mock LanguageModel to test detection
        class MockLanguageModel(LanguageModel):
            def __init__(self, model_name):
                self.model_name = model_name
                self.provider = self._detect_provider(model_name)
        
        mock_model = MockLanguageModel(model_name)
        actual_provider = mock_model.provider
        
        if actual_provider == expected_provider:
            print(f"✓ {model_name} -> {actual_provider}")
        else:
            print(f"✗ {model_name} -> {actual_provider} (expected {expected_provider})")
            return False
    
    return True

def test_vllm_client_creation():
    """Test vLLM client creation without starting server."""
    print("\nTesting vLLM client creation...")
    
    try:
        from models.vllm_client import VLLMClient
        
        # Test basic creation
        client = VLLMClient(
            model_name="test-model",
            model_path="/fake/path",
            temperature=0.7,
            max_tokens=1024
        )
        
        print("✓ VLLMClient created successfully")
        
        # Test provider info
        info = client.get_provider_info()
        expected_keys = ["provider", "model", "model_path", "client_type", "initialized"]
        
        if all(key in info for key in expected_keys):
            print("✓ Provider info contains expected keys")
        else:
            print(f"✗ Provider info missing keys: {set(expected_keys) - set(info.keys())}")
            return False
        
        # Test initialization status
        if not client.is_initialized():
            print("✓ Client correctly reports not initialized")
        else:
            print("✗ Client incorrectly reports as initialized")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ VLLMClient creation failed: {e}")
        return False

def test_language_model_creation():
    """Test LanguageModel creation with vLLM detection."""
    print("\nTesting LanguageModel creation...")
    
    try:
        from models.LanguageModel import LanguageModel
        
        # Test vLLM model creation
        model = LanguageModel(
            model_name="test-trained-model",
            model_path="/fake/path",
            temperature=0.7
        )
        
        print("✓ LanguageModel created successfully")
        
        # Test provider detection
        if model.is_vllm():
            print("✓ Correctly detected as vLLM provider")
        else:
            print(f"✗ Incorrectly detected as {model.provider} provider")
            return False
        
        # Test provider info
        info = model.get_provider_info()
        if info["provider"] == "vllm":
            print("✓ Provider info correct")
        else:
            print(f"✗ Provider info incorrect: {info}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ LanguageModel creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running vLLM integration tests...\n")
    
    tests = [
        test_imports,
        test_provider_detection,
        test_vllm_client_creation,
        test_language_model_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
