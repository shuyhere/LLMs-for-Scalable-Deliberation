#!/bin/bash

# Project directory
PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

TRAIN_DATA="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood_fixed_ood/train.jsonl"
IN_DIST_TEST="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood_fixed_ood/test.jsonl"
OOD_TEST="/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/summary_rating_dataset/comment_summary_ratings_ood_fixed_ood/ood_test.jsonl"

# Sweep spaces (edit as needed)
MODELS=(
  "microsoft/deberta-v3-base"
  "microsoft/deberta-v3-large"
  "allenai/longformer-base-4096"
)
LRS=("1e-5" "2e-5" "4e-5")
BATCHES=(8)
EPOCHS=(10)
MAX_LENGTHS=(4096)
SEEDS=(42)
DROPOUTS=(0.1 0.2)
WARMUP_RATIOS=(0.1 0.15)
WEIGHT_DECAYS=(0.01 0.001)
GRADIENT_CLIPS=(1.0 2.0)
EVAL_EVERY_STEPS=50

# Activations (set one of these true via flags below per run if needed)
USE_TANH=false
USE_SIGMOID=false
USE_RELU=false
USE_LEAKY_RELU=false
USE_ELU=false

# Normalization flag
NO_NORMALIZE=false

# System parameters
TIME="12:00:00"
MEMORY="32G"
CPUS=8
GPUS_LINE="#SBATCH --gres=gpu:a100:1"

# Wandb
WANDB_PROJECT_BASE="llm_comment_summary_regression_ood"

# Output roots
OUTPUT_ROOT="$PROJECT_DIR/results/deberta_regression_ood_sweeps"
LOG_ROOT="$PROJECT_DIR/logs/deberta_regression_ood_sweeps"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

submit_job() {
  local model="$1"
  local lr="$2"
  local batch="$3"
  local epochs="$4"
  local max_len="$5"
  local seed="$6"
  local dropout="$7"
  local warmup_ratio="$8"
  local weight_decay="$9"
  local gradient_clip="${10}"

  local model_tag
  model_tag=$(basename "$model")

  local run_tag="${model_tag}_bs${batch}_lr${lr}_ml${max_len}_ep${epochs}_dr${dropout}_wd${weight_decay}_wr${warmup_ratio}_gc${gradient_clip}_S${seed}"
  local out_dir="$OUTPUT_ROOT/${run_tag}"
  local log_dir="$LOG_ROOT/${model_tag}"
  mkdir -p "$out_dir" "$log_dir"

  # Build extra command args
  local cmd_args=""
  if [ "$USE_TANH" = true ]; then cmd_args+=" --use-tanh"; fi
  if [ "$USE_SIGMOID" = true ]; then cmd_args+=" --use-sigmoid"; fi
  if [ "$USE_RELU" = true ]; then cmd_args+=" --use-relu"; fi
  if [ "$USE_LEAKY_RELU" = true ]; then cmd_args+=" --use-leaky-relu"; fi
  if [ "$USE_ELU" = true ]; then cmd_args+=" --use-elu"; fi
  if [ "$NO_NORMALIZE" = true ]; then cmd_args+=" --no-normalize"; fi

  local wandb_project="${WANDB_PROJECT_BASE}_${model_tag}"
  local wandb_run_name="${run_tag}"

  echo "Submitting OOD training: ${run_tag}"

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ood_reg_${model_tag}_S${seed}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${log_dir}/job_%j_${run_tag}.out
#SBATCH --error=${log_dir}/job_%j_${run_tag}.err
${GPUS_LINE}

# Load environment
cd ${PROJECT_DIR}
source .venv/bin/activate

# Run training with explicit in-dist and OOD test sets
python3 ${PROJECT_DIR}/src/finetuning/sft_train_multioutput_regression.py \
  --data "${TRAIN_DATA}" \
  --in-dist-test "${IN_DIST_TEST}" \
  --ood-test "${OOD_TEST}" \
  --model "${model}" \
  --out "${out_dir}" \
  --epochs ${epochs} \
  --lr ${lr} \
  --batch ${batch} \
  --max-len ${max_len} \
  --seed ${seed} \
  --wandb \
  --wandb-project "${wandb_project}" \
  --wandb-run-name "${wandb_run_name}" \
  --eval-every-steps ${EVAL_EVERY_STEPS} \
  --warmup-ratio ${warmup_ratio} \
  --weight-decay ${weight_decay} \
  --gradient-clip ${gradient_clip} \
  --label-smoothing 0.0 \
  --dropout ${dropout} \
  --lr-scheduler linear \
  ${cmd_args}
EOF
}

# Iterate over sweep space and submit all combinations
for model in "${MODELS[@]}"; do
  for lr in "${LRS[@]}"; do
    for batch in "${BATCHES[@]}"; do
      for epochs in "${EPOCHS[@]}"; do
        for max_len in "${MAX_LENGTHS[@]}"; do
          for seed in "${SEEDS[@]}"; do
            for dropout in "${DROPOUTS[@]}"; do
              for warmup_ratio in "${WARMUP_RATIOS[@]}"; do
                for weight_decay in "${WEIGHT_DECAYS[@]}"; do
                  for gradient_clip in "${GRADIENT_CLIPS[@]}"; do
                    submit_job "$model" "$lr" "$batch" "$epochs" "$max_len" "$seed" "$dropout" "$warmup_ratio" "$weight_decay" "$gradient_clip"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All sweep jobs submitted. Check queue with: squeue -u \$USER"
