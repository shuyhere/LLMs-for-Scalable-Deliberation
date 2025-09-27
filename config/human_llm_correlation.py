#!/usr/bin/env python3
"""
Configuration file for human-LLM correlation experiments
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths
DEFAULT_ANNOTATION_PATH = PROJECT_ROOT / 'annotation/summary-rating/annotation_output/full'
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'results/llm_human_correlation'

# Model configurations
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SAMPLE_SIZE = 100

# Debug settings
DEFAULT_DEBUG = False

# Experiment settings
DEFAULT_RANDOM_SEED = 42
