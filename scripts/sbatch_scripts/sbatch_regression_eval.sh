#!/bin/bash

# Script for batch submitting regression evaluation jobs for all model-size combinations
# Usage: ./sbatch_regression_eval.sh [--eval-model-path PATH] [--models MODEL1,MODEL2] [--sizes SIZE1,SIZE2]

# Define project directory (relative to script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default evaluation model path
DEFAULT_EVAL_MODEL_PATH="$PROJECT_DIR/model_path"

# Available models and sample sizes

AVAILABLE_MODELS=("gpt-5" "web-rev-claude-sonnet-4-20250514" "gemini-2.5-flash" "deepseek-reasoner" "grok-4-latest" "TA/openai/gpt-oss-120b" "TA/openai/gpt-oss-20b" "qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-30b-a3b" "qwen3-235b-a22b" "qwen3-32b" "web-rev-claude-opus-4-20250514" "deepseek-chat" "gemini-2.5-pro" "deepseek-reasoner" "gpt-4o-mini")

# AVAILABLE_MODELS=("gpt-5-mini" )
# AVAILABLE_MODELS=("gpt-4o-mini")

AVAILABLE_SIZES=(500 1000)

# Parse command line arguments
EVAL_MODEL_PATH="$DEFAULT_EVAL_MODEL_PATH"
SELECTED_MODELS=()
SELECTED_SIZES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-model-path)
            EVAL_MODEL_PATH="$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra SELECTED_MODELS <<< "$2"
            shift 2
            ;;
        --sizes)
            IFS=',' read -ra SELECTED_SIZES <<< "$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --eval-model-path PATH   Path to trained DeBERTa model (default: $DEFAULT_EVAL_MODEL_PATH)"
            echo "  --models MODEL1,MODEL2   Comma-separated list of models to evaluate"
            echo "                          Available: ${AVAILABLE_MODELS[*]}"
            echo "  --sizes SIZE1,SIZE2     Comma-separated list of sample sizes"
            echo "                          Available: ${AVAILABLE_SIZES[*]}"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Evaluate all models and sizes"
            echo "  $0"
            echo ""
            echo "  # Evaluate specific models with all sizes"
            echo "  $0 --models deepseek-chat,gpt-4o-mini"
            echo ""
            echo "  # Evaluate all models with specific sizes"
            echo "  $0 --sizes 10,50"
            echo ""
            echo "  # Evaluate specific model-size combinations"
            echo "  $0 --models deepseek-chat --sizes 10,25"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Use all models/sizes if not specified
if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
    SELECTED_MODELS=("${AVAILABLE_MODELS[@]}")
fi
if [ ${#SELECTED_SIZES[@]} -eq 0 ]; then
    SELECTED_SIZES=("${AVAILABLE_SIZES[@]}")
fi

# Check if eval model path exists
if [[ ! -d "$EVAL_MODEL_PATH" ]]; then
    echo "Error: Evaluation model path does not exist: $EVAL_MODEL_PATH"
    exit 1
fi

# Create necessary directories
BASE_OUTPUT_DIR="$PROJECT_DIR/results/regression_evaluation_scaled"
LOG_DIR="$PROJECT_DIR/logs/regression_evaluation"
mkdir -p "$LOG_DIR"

# Print configuration
echo "=========================================="
echo "Regression Evaluation Batch Submission"
echo "=========================================="
echo "Evaluation Model: $EVAL_MODEL_PATH"
echo "Models to evaluate: ${SELECTED_MODELS[*]}"
echo "Sample sizes: ${SELECTED_SIZES[*]}"
echo "Output base directory: $BASE_OUTPUT_DIR"
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Counter for submitted jobs
SUBMITTED_JOBS=0

# Submit a job for each model-size combination
for MODEL_NAME in "${SELECTED_MODELS[@]}"; do
    for SAMPLE_SIZE in "${SELECTED_SIZES[@]}"; do
        
        # Create output directory for this combination
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}/${SAMPLE_SIZE}"
        mkdir -p "$OUTPUT_DIR"
        
        # Job name
        JOB_NAME="eval_${MODEL_NAME}_${SAMPLE_SIZE}"
        
        # Create temporary sbatch script
        TEMP_SCRIPT="temp_${JOB_NAME}_$$.sh"
        
        cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --account=conf-icl-2025.09.24-wangd0d

# Set working directory
cd $PROJECT_DIR

echo "=========================================="
echo "Starting regression evaluation at \\$(date)"
echo "Model: $MODEL_NAME"
echo "Sample Size: $SAMPLE_SIZE"
echo "Eval Model: $EVAL_MODEL_PATH"
echo "=========================================="

# Activate environment
source .venv/bin/activate

# Run batch evaluation script
python scripts/batch_regression_evaluate.py \
    --model-path "$EVAL_MODEL_PATH" \
    --summary-dir "$PROJECT_DIR/results/summary_model_for_evaluation_scaled" \
    --comments-dir "$PROJECT_DIR/datasets/minority" \
    --model-names "$MODEL_NAME" \
    --sample-sizes $SAMPLE_SIZE \
    --n-summaries 3 \
    --output "${OUTPUT_DIR}/evaluation_results.json" \
    --per-comment-csv "${OUTPUT_DIR}/per_comment_scores.csv" \
    --device cuda

echo "=========================================="
echo "Evaluation completed at \\$(date)"
echo "Results saved to: ${OUTPUT_DIR}/evaluation_results.json"
echo "Per-comment scores saved to: ${OUTPUT_DIR}/per_comment_scores.csv"
echo "=========================================="
EOF
        
        # Submit the job
        echo "Submitting job: ${JOB_NAME}"
        sbatch "$TEMP_SCRIPT"
        
        # Clean up temporary script
        rm -f "$TEMP_SCRIPT"
        
        SUBMITTED_JOBS=$((SUBMITTED_JOBS + 1))
        
        # Small delay to avoid overwhelming the scheduler
        sleep 0.5
    done
done

echo ""
echo "=========================================="
echo "Submission Summary"
echo "=========================================="
echo "Total jobs submitted: $SUBMITTED_JOBS"
echo "Check status with: squeue -u \$USER"
echo "Check logs in: $LOG_DIR/"
echo ""
echo "Results will be saved to:"
for MODEL_NAME in "${SELECTED_MODELS[@]}"; do
    for SAMPLE_SIZE in "${SELECTED_SIZES[@]}"; do
        echo "  ${BASE_OUTPUT_DIR}/${MODEL_NAME}/${SAMPLE_SIZE}/evaluation_results.json"
        echo "  ${BASE_OUTPUT_DIR}/${MODEL_NAME}/${SAMPLE_SIZE}/per_comment_scores.csv"
    done
done
echo "=========================================="