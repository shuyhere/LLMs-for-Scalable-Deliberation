#!/bin/bash

# Example script for training with OOD evaluation
# This script demonstrates how to use the enhanced training script with separate test datasets

# Set paths
TRAIN_DATA="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood_fixed_ood/train.jsonl"
IN_DIST_TEST="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood_fixed_ood/test.jsonl"
OOD_TEST="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood_fixed_ood/ood_test.jsonl"
OUTPUT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation/results/deberta_regression_ood"

# Run training with OOD evaluation
python /ibex/project/c2328/LLMs-Scalable-Deliberation/src/finetuning/sft_train_multioutput_regression.py \
    --data "$TRAIN_DATA" \
    --in-dist-test "$IN_DIST_TEST" \
    --ood-test "$OOD_TEST" \
    --out "$OUTPUT_DIR" \
    --model microsoft/deberta-v3-base \
    --epochs 20 \
    --lr 4e-5 \
    --batch 8 \
    --max-len 4096 \
    --seed 42 \
    --wandb \
    --wandb-project "llm_comment_summary_regression_ood" \
    --wandb-run-name "deberta-ood-eval"

echo "Training completed with OOD evaluation!"
echo "Results saved to: $OUTPUT_DIR"
