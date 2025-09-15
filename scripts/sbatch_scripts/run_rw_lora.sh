#!/bin/bash

# Define project directory
PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

# Common parameters - Using LoRA for memory efficiency
DATASET_DIR="$PROJECT_DIR/datasets/rl_datasets/trl_format"
OUTPUT_DIR="$PROJECT_DIR/outputs/reward_models"
MODEL_NAME="Qwen/Qwen3-8B"  # Back to 8B but with LoRA
MAX_LENGTH=4096  # Reduced length
TRAIN_BATCH_SIZE=4  # Moderate batch size
EVAL_BATCH_SIZE=4   
MAX_GRAD_NORM=10.0
NUM_EPOCHS=3
LEARNING_RATE=2e-5  # Adjusted LR for LoRA
EVAL_STEPS=10
LOGGING_STEPS=1
SAVE_STEPS=100
SEED=42
TIME="12:00:00"  
MEMORY="64G"    
CPUS=8          
LOG_DIR="$PROJECT_DIR/logs/reward_models" 
GPUS_LINE="#SBATCH --gres=gpu:a100:1"  # 4 GPUs with LoRA
PARTITION_LINE="#SBATCH --partition=a100" 
EVAL_SPLIT=0.1
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
GRAD_ACCUM_STEPS=4  
WEIGHT_DECAY=0.01
WAND_PROJECT="reward-modeling-Qwen3-8b-lora-v0"
RUN_NAME_PREFIX="rw_${MODEL_NAME##*/}_lora_S${SEED}"

# LoRA parameters
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Define dimensions to train
# DIMENSIONS=("perspective" "informativeness" "neutrality" "policy")
DIMENSIONS=("informativeness")
# Function to submit reward model training job
submit_reward_model_job() {
    local dimension=$1
    local dataset_path="$DATASET_DIR/${dimension}_trl_dataset.jsonl"
    local train_path="$DATASET_DIR/${dimension}_trl_dataset/train.jsonl"
    local test_path="$DATASET_DIR/${dimension}_trl_dataset/test.jsonl"
    local output_path="$OUTPUT_DIR/${dimension}_reward_model_lora"
    
    echo "Submitting reward model training job for dimension: $dimension"
    echo "Dataset: $dataset_path"
    echo "Train override: $train_path"
    echo "Test override: $test_path"
    echo "Output: $output_path"
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=reward_${dimension}_${MODEL_NAME##*/}_lora_S${SEED}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/job_%j_reward_${dimension}_${MODEL_NAME##*/}_lora_S${SEED}.out
#SBATCH --error=${LOG_DIR}/job_%j_reward_${dimension}_${MODEL_NAME##*/}_lora_S${SEED}.err
#SBATCH --account=conf-icl-2025.09.24-wangd0d
${GPUS_LINE}
${PARTITION_LINE}

# Optimize CUDA memory behavior
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Run reward model training with accelerate and LoRA
accelerate launch --num_processes=1 --main_process_port=29500 \\
    $PROJECT_DIR/src/finetuning/reward_modeling.py \\
    --model_name_or_path $MODEL_NAME \\
    --dataset_path $dataset_path \\
    --train_path $train_path \\
    --test_path $test_path \\
    --eval_split $EVAL_SPLIT \\
    --output_dir $output_path \\
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \\
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \\
    --max_grad_norm $MAX_GRAD_NORM \\
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \\
    --num_train_epochs $NUM_EPOCHS \\
    --learning_rate $LEARNING_RATE \\
    --weight_decay $WEIGHT_DECAY \\
    --lr_scheduler_type $LR_SCHEDULER_TYPE \\
    --warmup_ratio $WARMUP_RATIO \\
    --eval_strategy steps \\
    --eval_steps $EVAL_STEPS \\
    --logging_steps $LOGGING_STEPS \\
    --save_steps $SAVE_STEPS \\
    --save_total_limit 3 \\
    --load_best_model_at_end \\
    --metric_for_best_model eval_accuracy \\
    --greater_is_better \\
    --max_length $MAX_LENGTH \\
    --bf16 \\
    --gradient_checkpointing \\
    --dataloader_num_workers 2 \\
    --use_peft \\
    --lora_r $LORA_R \\
    --lora_alpha $LORA_ALPHA \\
    --lora_dropout $LORA_DROPOUT \\
    --seed $SEED \\
    --report_to wandb \\
    --wandb_project $WAND_PROJECT \\
    --run_name ${RUN_NAME_PREFIX}_${dimension}
EOF
}

# Submit jobs for all dimensions
for dimension in "${DIMENSIONS[@]}"; do
    submit_reward_model_job $dimension
    echo "Submitted job for $dimension dimension"
    sleep 2 
done

echo "All reward model training jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in: $LOG_DIR"
