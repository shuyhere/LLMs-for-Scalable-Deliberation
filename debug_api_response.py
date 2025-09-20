#!/usr/bin/env python3
"""
Debug script to test zhizengzeng API response format
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.LanguageModel import LanguageModel
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_api_response():
    """Test API response format"""
    try:
        # Initialize model
        model = LanguageModel(model_name="deepseek-reasoner", temperature=0.7)
        
        # Simple test message
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        print("Testing API response...")
        print(f"Model: {model.model_name}")
        print(f"Provider: {model.provider}")
        
        # Make request
        response = model.get_chat_response(messages)
        
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        if response:
            print("✅ API call successful!")
        else:
            print("❌ API call failed - no response")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_response()
