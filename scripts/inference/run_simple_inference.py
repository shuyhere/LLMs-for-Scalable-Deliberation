#!/usr/bin/env python3
"""
Simple script to run reward model inference with preset paths
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.inference.simple_reward_inference import main

if __name__ == "__main__":
    # Set the arguments directly
    sys.argv = [
        "simple_reward_inference.py",
        "--model_path", "/ibex/project/c2328/LLMs-Scalable-Deliberation/outputs/reward_models/informativeness_reward_model",
        "--test_data_path", "datasets/rl_datasets/trl_format/informativeness_trl_dataset/test.jsonl",
        "--output_path", "results/test_reward_model_inference/informativeness_scores.jsonl"
    ]
    
    # Create output directory
    Path("results/test_reward_model_inference").mkdir(parents=True, exist_ok=True)
    
    # Run the main function
    main()
