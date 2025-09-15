#!/bin/bash

# Direct vLLM evaluation script
# This script uses vLLM directly without the vLLM client wrapper

echo "🚀 Starting direct vLLM evaluation..."

# Activate virtual environment
source /ibex/project/c2328/LLMs-Scalable-Deliberation/.venv/bin/activate

# Set environment variables for GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Model and data paths
MODEL_PATH="/ibex/project/c2328/sft_tools/LLaMA-Factory/saves/qwen3_4b/full/deliberation_sft_compair/checkpoint-198"
TEST_PATH="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_annotation_format/alpace/comparison_alpaca/test.jsonl"
OUTPUT_FILE="/ibex/project/c2328/LLMs-Scalable-Deliberation/evaluation_results_direct.json"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Model path: $MODEL_PATH"
echo "Test data: $TEST_PATH"
echo "Output file: $OUTPUT_FILE"

# Run evaluation with limited samples first for testing
echo "🧪 Running evaluation with limited samples (10) for testing..."
python /ibex/project/c2328/LLMs-Scalable-Deliberation/scripts/evaluate_judge/evaluate_with_vllm_direct.py \
    --model_path "$MODEL_PATH" \
    --test_path "$TEST_PATH" \
    --output_file "${OUTPUT_FILE}_test.json" \
    --batch_size 5 \
    --max_samples 10

if [ $? -eq 0 ]; then
    echo "✅ Test evaluation completed successfully!"
    
    # Ask user if they want to run full evaluation
    echo "🔄 Test completed. Do you want to run full evaluation? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🚀 Running full evaluation..."
        python /ibex/project/c2328/LLMs-Scalable-Deliberation/scripts/evaluate_judge/evaluate_with_vllm_direct.py \
            --model_path "$MODEL_PATH" \
            --test_path "$TEST_PATH" \
            --output_file "$OUTPUT_FILE" \
            --batch_size 10
        
        if [ $? -eq 0 ]; then
            echo "✅ Full evaluation completed successfully!"
        else
            echo "❌ Full evaluation failed!"
            exit 1
        fi
    else
        echo "⏹️ Skipping full evaluation as requested."
    fi
else
    echo "❌ Test evaluation failed!"
    exit 1
fi

echo "✅ All evaluations completed!"
