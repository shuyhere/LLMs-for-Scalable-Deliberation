#!/bin/bash

# Define project directory
PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

# Data parameters
DATA_FILE="$PROJECT_DIR/datasets/summary_rating_dataset/comment_summary_ratings.jsonl"
# Alternative split data path:
# DATA_FILE="$PROJECT_DIR/datasets/summary_rating_dataset/split_data/train.jsonl"

MODEL_NAME="Qwen/Qwen3-0.6B"  # Qwen 0.5B model
OUTPUT_DIR="$PROJECT_DIR/checkpoints/qwen_regression_base/qwen3-0.6b"

# Training parameters
MAX_LENGTH=4096  # Qwen typically uses 2048 context length
TRAIN_BATCH_SIZE=4  # Smaller batch size for LLM
EVAL_BATCH_SIZE=4
NUM_EPOCHS=10
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.15
EVAL_RATIO=0.1
SEED=42
EVAL_EVERY_STEPS=50
GRADIENT_ACCUMULATION_STEPS=4  # For larger effective batch size

# System parameters
TIME="24:00:00"  # Longer time for LLM training
MEMORY="48G"     # More memory for LLM
CPUS=8
LOG_DIR="$PROJECT_DIR/logs/qwen_regression/qwen3-0.6b"
GPUS_LINE="#SBATCH --gres=gpu:a100:1"

# Wandb parameters
WANDB_PROJECT="llm_comment_summary_regression_qwen"
WANDB_RUN_NAME="qwen_regression_base-qwen3-0.6b"

# Advanced training parameters
GRADIENT_CLIP=1.5
DROPOUT=0.15
LR_SCHEDULER="cosine"  # Often better for LLMs

# Data augmentation parameters
AUGMENT=true
AUGMENT_PROB=0.3

# Model parameters
USE_TANH=false
USE_SIGMOID=true  # Normalize to [0,1] for stability
USE_LORA=false
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
FREEZE_BASE_MODEL=false

# Normalization parameter
NO_NORMALIZE=false

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
        --use-lora)
            USE_LORA=true
            shift
            ;;
        --lora-r)
            LORA_R="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --freeze-base-model)
            FREEZE_BASE_MODEL=true
            shift
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --use-tanh)
            USE_TANH=true
            USE_SIGMOID=false
            shift
            ;;
        --use-sigmoid)
            USE_SIGMOID=true
            USE_TANH=false
            shift
            ;;
        --no-normalize)
            NO_NORMALIZE=true
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
            echo "  --use-lora               Use LoRA for parameter-efficient fine-tuning"
            echo "  --lora-r R               LoRA rank (default: $LORA_R)"
            echo "  --lora-alpha ALPHA       LoRA alpha (default: $LORA_ALPHA)"
            echo "  --freeze-base-model      Freeze base model parameters"
            echo "  --gradient-accumulation-steps STEPS  Gradient accumulation steps (default: $GRADIENT_ACCUMULATION_STEPS)"
            echo "  --use-tanh               Use tanh activation for output (range [-1, 1])"
            echo "  --use-sigmoid            Use sigmoid activation for output (range [0, 1])"
            echo "  --no-normalize           Disable normalization and use original label range [-1, 7]"
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
MODEL_TAG=$(basename "$MODEL_NAME" | sed 's/\//-/g')
DATA_TAG=$(basename "$(dirname "$DATA_FILE")")

# Add LoRA tag if using LoRA
if [ "$USE_LORA" = true ]; then
    RUN_NAME="${WANDB_RUN_NAME}_lora_r${LORA_R}_a${LORA_ALPHA}"
    OUTPUT_DIR="${OUTPUT_DIR}_lora"
elif [ "$FREEZE_BASE_MODEL" = true ]; then
    RUN_NAME="${WANDB_RUN_NAME}_frozen"
    OUTPUT_DIR="${OUTPUT_DIR}_frozen"
fi

RUN_NAME="${RUN_NAME}_${MODEL_TAG}_${DATA_TAG}_S${SEED}"

# Build command arguments
CMD_ARGS=""
if [ "$AUGMENT" = true ]; then
    CMD_ARGS="$CMD_ARGS --augment"
fi
if [ "$USE_LORA" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-lora --lora-r $LORA_R --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DROPOUT"
fi
if [ "$FREEZE_BASE_MODEL" = true ]; then
    CMD_ARGS="$CMD_ARGS --freeze-base-model"
fi
if [ "$USE_TANH" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-tanh"
fi
if [ "$USE_SIGMOID" = true ]; then
    CMD_ARGS="$CMD_ARGS --use-sigmoid"
fi
if [ "$NO_NORMALIZE" = true ]; then
    CMD_ARGS="$CMD_ARGS --no-normalize"
fi

echo "Submitting Qwen regression training job..."
echo "Data: $DATA_FILE"
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $TRAIN_BATCH_SIZE (x${GRADIENT_ACCUMULATION_STEPS} gradient accumulation)"
echo "Effective batch size: $((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA enabled: $USE_LORA"
echo "Freeze base model: $FREEZE_BASE_MODEL"
echo "WANDB Project: $WANDB_PROJECT"
echo "WANDB Run Name: $RUN_NAME"

# Submit job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=qwen_regression_${MODEL_TAG}_S${SEED}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --output=${LOG_DIR}/job_%j_qwen_regression_${MODEL_TAG}_S${SEED}.out
#SBATCH --error=${LOG_DIR}/job_%j_qwen_regression_${MODEL_TAG}_S${SEED}.err
#SBATCH --account=conf-icl-2025.09.24-wangd0d
${GPUS_LINE}

# Load environment
cd $PROJECT_DIR
source .venv/bin/activate

# Set CUDA environment variables for better performance
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# Run training
python3 $PROJECT_DIR/src/finetuning/sft_train_qwen_regression.py \
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
  --dropout $DROPOUT \
  --lr-scheduler $LR_SCHEDULER \
  --augment-prob $AUGMENT_PROB \
  --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
  $CMD_ARGS

echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
EOF

echo "Job submitted successfully!"
echo "Check status with: squeue -u \$USER"
echo "View logs in: $LOG_DIR"

# Provide example commands for different training configurations
echo ""
echo "Example commands for different configurations:"
echo "1. With LoRA (parameter-efficient):"
echo "   bash $0 --use-lora --lora-r 16 --lora-alpha 32 --epochs 5"
echo ""
echo "2. With frozen base model (only train head):"
echo "   bash $0 --freeze-base-model --lr 1e-3 --epochs 10"
echo ""
echo "3. Full fine-tuning with smaller learning rate:"
echo "   bash $0 --lr 2e-5 --epochs 3"