#!/bin/bash

# Test script to verify OOD evaluation works properly
# This runs a quick test with just 1 epoch and frequent evaluation

# Set paths
TRAIN_DATA="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood/train.jsonl"
IN_DIST_TEST="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood/test.jsonl"
OOD_TEST="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood/ood_test.jsonl"
OUTPUT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/test_ood_eval"

echo "Testing OOD evaluation with frequent steps..."
echo "This should show evaluation results every 25 steps on both in-dist and OOD test sets"
echo ""

# Run training with OOD evaluation
python /ibex/project/c2328/LLMs-Scalable-Deliberation/src/finetuning/sft_train_multioutput_regression.py \
    --data "$TRAIN_DATA" \
    --in-dist-test "$IN_DIST_TEST" \
    --ood-test "$OOD_TEST" \
    --out "$OUTPUT_DIR" \
    --model microsoft/deberta-v3-base \
    --epochs 1 \
    --lr 5e-5 \
    --batch 8 \
    --max-len 512 \
    --seed 42 \
    --eval-every-steps 25 

echo ""
echo "Test completed!"
echo "Check above for evaluation results on both test sets"