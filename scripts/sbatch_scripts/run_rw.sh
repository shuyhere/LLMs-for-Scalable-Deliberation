#!/bin/bash

# Define project directory
PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

# Common parameters
DATASET_DIR="$PROJECT_DIR/datasets/rl_datasets/trl_format"
OUTPUT_DIR="$PROJECT_DIR/outputs/reward_models"
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
MAX_LENGTH=8192
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
MAX_GRAD_NORM=10.0
NUM_EPOCHS=30
LEARNING_RATE=1e-4
EVAL_STEPS=20
LOGGING_STEPS=1
SAVE_STEPS=100
SEED=42
TIME="12:00:00"  
MEMORY="64G"    
CPUS=8          
LOG_DIR="$PROJECT_DIR/logs/reward_models" 
GPUS_LINE="#SBATCH --gres=gpu:a100:1"  # Example GPU line
PARTITION_LINE="#SBATCH --partition=a100" # Example partition line
EVAL_SPLIT=0.3
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
GRAD_ACCUM_STEPS=4
WEIGHT_DECAY=0.01
WAND_PROJECT="reward-modeling"
RUN_NAME_PREFIX="rw_${MODEL_NAME##*/}_S${SEED}"

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Define dimensions to train
DIMENSIONS=("perspective" "informativeness" "neutrality" "policy")
# DIMENSIONS=("informativeness")
# DIMENSIONS=("neutrality")
# DIMENSIONS=("policy")
# DIMENSIONS=("perspective")

# Function to submit reward model training job
submit_reward_model_job() {
    local dimension=$1
    local dataset_path="$DATASET_DIR/${dimension}_trl_dataset.jsonl"
    local output_path="$OUTPUT_DIR/${dimension}_reward_model"
    
    echo "Submitting reward model training job for dimension: $dimension"
    echo "Dataset: $dataset_path"
    echo "Output: $output_path"
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=reward_${dimension}_${MODEL_NAME##*/}_S${SEED}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/job_%j_reward_${dimension}_${MODEL_NAME##*/}_S${SEED}.out
#SBATCH --error=${LOG_DIR}/job_%j_reward_${dimension}_${MODEL_NAME##*/}_S${SEED}.err
#SBATCH --account=conf-icl-2025.09.24-wangd0d
${GPUS_LINE}
${PARTITION_LINE}

# Optimize CUDA memory behavior
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Run reward model training
python3 $PROJECT_DIR/src/finetuning/reward_modeling.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_path $dataset_path \
    --eval_split $EVAL_SPLIT \
    --output_dir $output_path \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --max_grad_norm $MAX_GRAD_NORM \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --warmup_ratio $WARMUP_RATIO \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit 1 \
    --max_length $MAX_LENGTH \
    --bf16 \
    --seed $SEED \
    --report_to wandb \
    --wandb_project $WAND_PROJECT \
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
