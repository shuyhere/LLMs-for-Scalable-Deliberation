#!/bin/bash

# Define project directory
PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

# Data parameters
# DATA_FILE="$PROJECT_DIR/datasets/summary_rating_dataset/comment_summary_ratings.jsonl"
DATA_FILE="$PROJECT_DIR/datasets/summary_rating_dataset/split_data/train.jsonl"
MODEL_NAME="microsoft/deberta-v3-base"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/deberta_regression_base_v12_pair_split_relu"

# Training parameters
MAX_LENGTH=4096
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
NUM_EPOCHS=20
LEARNING_RATE=4e-5
WEIGHT_DECAY=1e-4
WARMUP_RATIO=0.1
EVAL_RATIO=0.16
SEED=42
EVAL_EVERY_STEPS=25


# System parameters
TIME="12:00:00"  
MEMORY="32G"    
CPUS=8          
LOG_DIR="$PROJECT_DIR/logs/multioutput_regression_base_v12_pair_split_relu"
GPUS_LINE="#SBATCH --gres=gpu:a100:1"

# Wandb parameters
WANDB_PROJECT="llm_comment_summary_regression_deberta"
WANDB_RUN_NAME="new_regression_deberta-v3-base_v12_pair_split_relu"

# Advanced training parameters
GRADIENT_CLIP=1.0
LABEL_SMOOTHING=0.0
DROPOUT=0.1
LR_SCHEDULER="linear"

# Data augmentation parameters
AUGMENT=false
AUGMENT_PROB=0.3

# Model parameters
USE_TANH=false
USE_SIGMOID=false
USE_RELU=true
USE_LEAKY_RELU=false
USE_ELU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch)
            TRAIN_BATCH_SIZE="$2"
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max-len)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --eval-ratio)
            EVAL_RATIO="$2"
            shift 2
            ;;
        --eval-steps)
            EVAL_EVERY_STEPS="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-run-name)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        --augment)
            AUGMENT=true
            shift
            ;;
        --use-tanh)
            USE_TANH=true
            USE_SIGMOID=false
            USE_RELU=false
            USE_LEAKY_RELU=false
            USE_ELU=false
            shift
            ;;
        --use-sigmoid)
            USE_SIGMOID=true
            USE_TANH=false
            USE_RELU=false
            USE_LEAKY_RELU=false
            USE_ELU=false
            shift
            ;;
        --use-relu)
            USE_RELU=true
            USE_TANH=false
            USE_SIGMOID=false
            USE_LEAKY_RELU=false
            USE_ELU=false
            shift
            ;;
        --use-leaky-relu)
            USE_LEAKY_RELU=true
            USE_TANH=false
            USE_SIGMOID=false
            USE_RELU=false
            USE_ELU=false
            shift
            ;;
        --use-elu)
            USE_ELU=true
            USE_TANH=false
            USE_SIGMOID=false
            USE_RELU=false
            USE_LEAKY_RELU=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --data FILE              Data file path (default: $DATA_FILE)"
            echo "  --model MODEL            Model name (default: $MODEL_NAME)"
            echo "  --output DIR             Output directory (default: $OUTPUT_DIR)"
            echo "  --batch SIZE             Batch size (default: $TRAIN_BATCH_SIZE)"
            echo "  --epochs N               Number of epochs (default: $NUM_EPOCHS)"
            echo "  --lr RATE                Learning rate (default: $LEARNING_RATE)"
            echo "  --max-len LENGTH         Max sequence length (default: $MAX_LENGTH)"
            echo "  --eval-ratio RATIO       Evaluation ratio (default: $EVAL_RATIO)"
            echo "  --eval-steps STEPS       Evaluation every N steps (default: $EVAL_EVERY_STEPS)"
            echo "  --time TIME              Job time limit (default: $TIME)"
            echo "  --memory MEMORY          Memory requirement (default: $MEMORY)"
            echo "  --wandb-project PROJECT  WANDB project name (default: $WANDB_PROJECT)"
            echo "  --wandb-run-name NAME    WANDB run name (default: $WANDB_RUN_NAME)"
            echo "  --augment                Enable data augmentation"
            echo "  --use-tanh               Use tanh activation for output (range [-1, 1])"
            echo "  --use-sigmoid            Use sigmoid activation for output (range [0, 1])"
            echo "  --use-relu               Use ReLU activation for output (range [0, +inf))"
            echo "  --use-leaky-relu         Use LeakyReLU activation for output (range (-inf, +inf))"
            echo "  --use-elu                Use ELU activation for output (range [-1, +inf))"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Derive run name from parameters
MODEL_TAG=$(basename "$MODEL_NAME")
DATA_TAG=$(basename "$(dirname "$DATA_FILE")")
RUN_NAME="${WANDB_RUN_NAME}_${MODEL_TAG}_${DATA_TAG}_S${SEED}"

# Build command arguments
CMD_ARGS=""
if [ "$AUGMENT" = true ]; then
    CMD_ARGS="$CMD_ARGS --augment"
fi
if [ "$USE_TANH" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-tanh"
fi
if [ "$USE_SIGMOID" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-sigmoid"
fi
if [ "$USE_RELU" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-relu"
fi
if [ "$USE_LEAKY_RELU" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-leaky-relu"
fi
if [ "$USE_ELU" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-elu"
fi

echo "Submitting multioutput regression training job..."
echo "Data: $DATA_FILE"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $TRAIN_BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "WANDB Project: $WANDB_PROJECT"
echo "WANDB Run Name: $RUN_NAME"

# Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=multioutput_regression_${MODEL_TAG}_S${SEED}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/job_%j_multioutput_regression_${MODEL_TAG}_S${SEED}.out
#SBATCH --error=${LOG_DIR}/job_%j_multioutput_regression_${MODEL_TAG}_S${SEED}.err
#SBATCH --account=conf-icl-2025.09.24-wangd0d
${GPUS_LINE}

# Load environment
cd $PROJECT_DIR
source .venv/bin/activate

# Run training
python3 $PROJECT_DIR/src/finetuning/sft_train_multioutput_regression.py \
  --data "$DATA_FILE" \
  --model "$MODEL_NAME" \
  --out "$OUTPUT_DIR" \
  --epochs $NUM_EPOCHS \
  --lr $LEARNING_RATE \
  --batch $TRAIN_BATCH_SIZE \
  --max-len $MAX_LENGTH \
  --eval-ratio $EVAL_RATIO \
  --seed $SEED \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$RUN_NAME" \
  --eval-every-steps $EVAL_EVERY_STEPS \
  --warmup-ratio $WARMUP_RATIO \
  --weight-decay $WEIGHT_DECAY \
  --gradient-clip $GRADIENT_CLIP \
  --label-smoothing $LABEL_SMOOTHING \
  --dropout $DROPOUT \
  --lr-scheduler $LR_SCHEDULER \
  --augment-prob $AUGMENT_PROB \
  $CMD_ARGS
EOF

echo "Job submitted successfully!"
echo "Check status with: squeue -u \$USER"
echo "View logs in: $LOG_DIR"
