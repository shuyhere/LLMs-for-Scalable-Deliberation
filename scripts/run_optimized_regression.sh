#!/bin/bash
#SBATCH --job-name=optimized_regression
#SBATCH --output=logs/optimized_regression_%j.log
#SBATCH --error=logs/optimized_regression_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Training parameters optimized to prevent overfitting
MODEL="microsoft/deberta-v3-base"
DATA="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/split_data/train.jsonl"
OUTPUT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation/checkpoints_optimized/deberta_regression_optimized_$(date +%Y%m%d_%H%M%S)"

echo "Starting MSE + Ranking loss training (NO correlation loss)..."
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Output: $OUTPUT_DIR"

python /ibex/project/c2328/LLMs-Scalable-Deliberation/src/finetuning/sft_train_mse_ranking.py \
    --data "$DATA" \
    --model "$MODEL" \
    --out "$OUTPUT_DIR" \
    --epochs 20 \
    --lr 3e-5 \
    --batch 8 \
    --accumulation-steps 2 \
    --max-len 4096 \
    --eval-ratio 0.15 \
    --patience 10 \
    --warmup-ratio 0.1 \
    --weight-decay 0.05 \
    --dropout 0.2 \
    --gradient-clip 1.0 \
    --eval-steps 25 \
    --mse-weight 0.7 \
    --rank-weight 0.3 \
    --wandb \
    --wandb-project "llm_regression_optimized" \
    --wandb-run-name "optimized_deberta_$(date +%Y%m%d_%H%M%S)" \
    --seed 42

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"