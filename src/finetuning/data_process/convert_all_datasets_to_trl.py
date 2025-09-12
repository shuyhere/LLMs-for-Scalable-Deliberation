#!/usr/bin/env python3
"""
Convert all RL datasets to TRL format for reward model training.
"""

import os
from pathlib import Path
import subprocess
import sys

def convert_all_datasets():
    """Convert all RL datasets to TRL format."""
    
    # Define dataset paths - all available datasets
    datasets = {
        "perspective": "datasets/rl_datasets/perspective_rl_dataset.jsonl",
        "informativeness": "datasets/rl_datasets/informativeness_rl_dataset.jsonl", 
        "neutrality": "datasets/rl_datasets/neutrality_rl_dataset.jsonl",
        "policy": "datasets/rl_datasets/policy_rl_dataset.jsonl",
        "all_dimensions": "datasets/rl_datasets/all_dimensions_rl_dataset.jsonl"
    }
    
    # Output directory
    output_dir = "datasets/rl_datasets/trl_format"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("🔄 Converting all RL datasets to TRL format...")
    
    for dimension, input_path in datasets.items():
        if not Path(input_path).exists():
            print(f"⚠️  Skipping {dimension}: {input_path} not found")
            continue
            
        output_path = Path(output_dir) / f"{dimension}_trl_dataset.jsonl"
        
        print(f"\n📝 Converting {dimension} dataset...")
        print(f"   Input:  {input_path}")
        print(f"   Output: {output_path}")
        
        try:
            # Run conversion script
            cmd = [
                sys.executable, 
                "src/finetuning/convert_to_trl_format.py",
                "--input-path", input_path,
                "--output-path", str(output_path),
                "--format", "chat"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {dimension} dataset converted successfully")
            else:
                print(f"❌ Error converting {dimension}: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error converting {dimension}: {e}")
    
    print(f"\n🎉 Conversion completed! Check output directory: {output_dir}")

if __name__ == "__main__":
    convert_all_datasets()
