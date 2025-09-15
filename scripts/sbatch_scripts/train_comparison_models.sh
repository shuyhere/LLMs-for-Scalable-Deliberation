#!/bin/bash

# Training script for comparison models (binary classification and regression)

# Set paths
DATASET_PATH="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/sft_dataset/comparison_sft_dataset.jsonl"
OUTPUT_BASE_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation/outputs/comparison_models"
MODEL_NAME="microsoft/deberta-v3-large"

# Create output directories
mkdir -p "$OUTPUT_BASE_DIR"

echo "Training comparison models..."
echo "Dataset: $DATASET_PATH"
echo "Model: $MODEL_NAME"

# Binary Classification Training
echo ""
echo "=== Training Binary Classifier ==="
BINARY_OUTPUT_DIR="$OUTPUT_BASE_DIR/binary_classifier"

python /ibex/project/c2328/LLMs-Scalable-Deliberation/src/finetuning/comparison_binary_classifier.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$BINARY_OUTPUT_DIR" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 50 \
    --logging_steps 5 \
    --seed 42 \
    --max_length 4096 \
    --gradient_accumulation_steps 4 \
    --gradient_clip_val 1.0 \
    --lr_scheduler_type cosine \
    --label_smoothing 0.0 \
    --use_hard_sampling \
    --use_augmentation \
    --fp16 \
    --dataloader_num_workers 4 \
    --save_total_limit 3

echo "Binary classifier training completed!"

# Binary Classification with Cross Validation
echo ""
echo "=== Training Binary Classifier with Cross Validation ==="
BINARY_CV_OUTPUT_DIR="$OUTPUT_BASE_DIR/binary_classifier_cv"

python /ibex/project/c2328/LLMs-Scalable-Deliberation/src/finetuning/comparison_binary_classifier.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$BINARY_CV_OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 50 \
    --logging_steps 5 \
    --seed 42 \
    --max_length 4096 \
    --gradient_accumulation_steps 4 \
    --gradient_clip_val 1.0 \
    --lr_scheduler_type cosine \
    --label_smoothing 0.0 \
    --use_hard_sampling \
    --use_augmentation \
    --cross_validation \
    --cv_folds 5 \
    --fp16 \
    --dataloader_num_workers 4 \
    --save_total_limit 3

echo "Binary classifier cross validation completed!"

# Regression Training
echo ""
echo "=== Training Regression Model ==="
REGRESSION_OUTPUT_DIR="$OUTPUT_BASE_DIR/regression"

python /ibex/project/c2328/LLMs-Scalable-Deliberation/src/finetuning/comparison_regression.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$REGRESSION_OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 100 \
    --seed 42 \
    --max_length 4096 \
    --gradient_accumulation_steps 1 \
    --bf16 \
    --dataloader_num_workers 4 \
    --save_total_limit 3

echo "Regression model training completed!"

echo ""
echo "All training completed!"
echo "Binary classifier saved to: $BINARY_OUTPUT_DIR"
echo "Binary classifier CV saved to: $BINARY_CV_OUTPUT_DIR"
echo "Regression model saved to: $REGRESSION_OUTPUT_DIR"

# Show final results
if [ -f "$BINARY_OUTPUT_DIR/eval_results.json" ]; then
    echo ""
    echo "=== Binary Classifier Results ==="
    cat "$BINARY_OUTPUT_DIR/eval_results.json"
fi

if [ -f "$BINARY_CV_OUTPUT_DIR/cv_summary.json" ]; then
    echo ""
    echo "=== Binary Classifier Cross Validation Results ==="
    cat "$BINARY_CV_OUTPUT_DIR/cv_summary.json"
fi

if [ -f "$REGRESSION_OUTPUT_DIR/eval_results.json" ]; then
    echo ""
    echo "=== Regression Model Results ==="
    cat "$REGRESSION_OUTPUT_DIR/eval_results.json"
fi
