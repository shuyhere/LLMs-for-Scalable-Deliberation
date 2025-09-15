#!/bin/bash

# Inference script for testing trained comparison models

# Set paths
TEST_DATA_PATH="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_dataset/comparison_split/test.jsonl"
OUTPUT_BASE_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation/outputs/inference_results"
MODEL_BASE_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation/outputs/comparison_models"

echo "Running inference on trained comparison models..."
echo "Test data: $TEST_DATA_PATH"

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Binary Classification Inference
echo ""
echo "=== Testing Binary Classifier ==="
BINARY_MODEL_PATH="$MODEL_BASE_DIR/binary_classifier"
BINARY_OUTPUT_DIR="$OUTPUT_BASE_DIR/binary_classifier"

if [ -d "$BINARY_MODEL_PATH" ]; then
    echo "Testing binary classifier model..."
    python /ibex/project/c2328/LLMs-Scalable-Deliberation/scripts/inference/test_comparison_models.py \
        --model_path "$BINARY_MODEL_PATH" \
        --test_data_path "$TEST_DATA_PATH" \
        --output_dir "$BINARY_OUTPUT_DIR" \
        --model_type binary \
        --base_model_name "microsoft/deberta-v3-large"
    
    echo "Binary classifier inference completed!"
else
    echo "Warning: Binary classifier model not found at $BINARY_MODEL_PATH"
fi

# Regression Inference
echo ""
echo "=== Testing Regression Model ==="
REGRESSION_MODEL_PATH="$MODEL_BASE_DIR/regression"
REGRESSION_OUTPUT_DIR="$OUTPUT_BASE_DIR/regression"

if [ -d "$REGRESSION_MODEL_PATH" ]; then
    echo "Testing regression model..."
    python /ibex/project/c2328/LLMs-Scalable-Deliberation/scripts/inference/test_comparison_models.py \
        --model_path "$REGRESSION_MODEL_PATH" \
        --test_data_path "$TEST_DATA_PATH" \
        --output_dir "$REGRESSION_OUTPUT_DIR" \
        --model_type regression \
        --base_model_name "microsoft/deberta-v3-large"
    
    echo "Regression model inference completed!"
else
    echo "Warning: Regression model not found at $REGRESSION_MODEL_PATH"
fi

echo ""
echo "Inference completed!"
echo "Results saved to: $OUTPUT_BASE_DIR"

# Show summary if results exist
if [ -f "$BINARY_OUTPUT_DIR/binary_metrics.json" ]; then
    echo ""
    echo "=== Binary Classifier Results ==="
    cat "$BINARY_OUTPUT_DIR/binary_metrics.json" | jq '.'
fi

if [ -f "$REGRESSION_OUTPUT_DIR/regression_metrics.json" ]; then
    echo ""
    echo "=== Regression Model Results ==="
    cat "$REGRESSION_OUTPUT_DIR/regression_metrics.json" | jq '.'
fi