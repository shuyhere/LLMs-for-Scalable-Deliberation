#!/bin/bash

# Define project directory
PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

# Common parameters
DATA_FILE="$PROJECT_DIR/datasets/finetuning_dataset/summary_rating_extracted.jsonl"
OUTPUT_DIR="$PROJECT_DIR/outputs/checkpoints/roberta-5class-eva/summary-rating"
MODEL_NAME="allenai/longformer-base-4096"
MAX_LENGTH=2048
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
NUM_EPOCHS=30
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

# Submit regression job
# sbatch <<EOF
# #!/bin/bash
# #SBATCH --job-name=regression_${MODEL_NAME}_S${SEED}
# #SBATCH --time=${TIME}
# #SBATCH --mem=${MEMORY}
# #SBATCH --cpus-per-task=${CPUS}
# #SBATCH --output=${LOG_DIR}/job_%j_regression_${MODEL_NAME}_S${SEED}.out
# #SBATCH --error=${LOG_DIR}/job_%j_regression_${MODEL_NAME}_S${SEED}.err
# #SBATCH --account=conf-icl-2025.09.24-wangd0d
# ${GPUS_LINE}
# ${PARTITION_LINE}

# python3 $PROJECT_DIR/src/finetuning/sft_train_5_class.py \
#   --dataset-path $DATA_FILE \
#   --output-dir $OUTPUT_DIR/regression \
#   --model-name $MODEL_NAME \
#   --max-length $MAX_LENGTH \
#   --batch-size $TRAIN_BATCH_SIZE \
#   --eval-batch-size $EVAL_BATCH_SIZE \
#   --epochs $NUM_EPOCHS \
#   --learning-rate $LEARNING_RATE \
#   --weight-decay $WEIGHT_DECAY \
#   --warmup-ratio $WARMUP_RATIO \
#   --eval-split $EVAL_SPLIT \
#   --seed $SEED
# EOF

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
  --data-file $DATA_FILE \
  --output-dir $OUTPUT_DIR/classification \
  --model-name $MODEL_NAME \
  --max-length $MAX_LENGTH \
  --per-device-train-batch-size $TRAIN_BATCH_SIZE \
  --per-device-eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --learning-rate $LEARNING_RATE \
  --weight-decay $WEIGHT_DECAY \
  --warmup-ratio $WARMUP_RATIO \
  --eval-split $EVAL_SPLIT \
  --seed $SEED
EOF