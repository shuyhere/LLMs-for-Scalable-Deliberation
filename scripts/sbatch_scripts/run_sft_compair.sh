#!/bin/bash

# Define project directory
PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

# Common parameters
DATA_FILE="$PROJECT_DIR/datasets/finetuning_dataset/summary_rating_extracted/train.jsonl"
TEST_FILE="$PROJECT_DIR/datasets/finetuning_dataset/summary_rating_extracted/test.jsonl"
MODEL_NAME="allenai/longformer-base-4096"
OUTPUT_DIR="$PROJECT_DIR/outputs/sft_modeling/$MODEL_NAME/"
MAX_LENGTH=4096
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
NUM_EPOCHS=8
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.2
EVAL_SPLIT=0.1
SEED=42
TIME="24:00:00"  
MEMORY="32G"    
CPUS=8          
LOG_DIR="$PROJECT_DIR/logs/sft"  # Specify your log directory
GPUS_LINE="#SBATCH --gres=gpu:a100:1"  # Example GPU line
PARTITION_LINE="#SBATCH --partition=a100" # Example partition line

# Wandb
WANDB_PROJECT="LLMs-Scalable-Deliberation-sft-training-longformer-base-4096"

# Derive data tag
DATA_TAG="orig"
if [[ "$DATA_FILE" == *"train_aug.jsonl" ]]; then
  DATA_TAG="aug"
fi

# Submit regression job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=regression_${MODEL_NAME}_S${SEED}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/job_%j_regression_${MODEL_NAME}_S${SEED}.out
#SBATCH --error=${LOG_DIR}/job_%j_regression_${MODEL_NAME}_S${SEED}.err
#SBATCH --account=conf-icl-2025.09.24-wangd0d
${GPUS_LINE}
${PARTITION_LINE}

python3 $PROJECT_DIR/src/finetuning/sft_train_regression.py \
  --dataset-path $DATA_FILE \
  --test-file $TEST_FILE \
  --output-dir $OUTPUT_DIR/regression \
  --model-name $MODEL_NAME \
  --max-length $MAX_LENGTH \
  --batch-size $TRAIN_BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --epochs $NUM_EPOCHS \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --warmup-ratio $WARMUP_RATIO \
  --eval-split $EVAL_SPLIT \
  --seed $SEED \
  --wandb_project $WANDB_PROJECT \
  --run_name "sft-regression_${MODEL_NAME##*/}_${DATA_TAG}_S${SEED}"
EOF

# Submit classification job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=classification_${MODEL_NAME}_S${SEED}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/job_%j_classification_${MODEL_NAME}_S${SEED}.out
#SBATCH --error=${LOG_DIR}/job_%j_classification_${MODEL_NAME}_S${SEED}.err
#SBATCH --account=conf-icl-2025.09.24-wangd0d
${GPUS_LINE}
${PARTITION_LINE}

python3 $PROJECT_DIR/src/finetuning/sft_train_multiclass.py \
  --dataset-path $DATA_FILE \
  --test-file $TEST_FILE \
  --output-dir $OUTPUT_DIR/classification \
  --model-name $MODEL_NAME \
  --max-length $MAX_LENGTH \
  --batch-size $TRAIN_BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --epochs $NUM_EPOCHS \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --warmup-ratio $WARMUP_RATIO \
  --eval-split $EVAL_SPLIT \
  --seed $SEED \
  --wandb_project $WANDB_PROJECT \
  --run_name "sft-multiclass_${MODEL_NAME##*/}_${DATA_TAG}_S${SEED}"
EOF