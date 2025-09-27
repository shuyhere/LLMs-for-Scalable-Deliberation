#!/bin/bash

# Script to resume checkpoint jobs and process missing data for human-LLM correlation evaluation
# Usage: ./resume_checkpoint_jobs.sh [mode] [model_name] [temperature]
# Mode: 'resume' for resuming checkpoints, 'missing' for processing missing data
# Model: specific model name to process (e.g., gpt-5-nano)
# Temperature: temperature for model generation (default: 1.0)

List of all models to process
EVAL_MODELS=("gpt-4o-mini" "gemini-2.5-flash-lite"  "web-rev-claude-3-7-sonnet-20250219" "web-rev-claude-sonnet-4-20250514" "gemini-2.5-flash" "deepseek-reasoner" "grok-4-latest" "qwen3-0.6b" "qwen3-1.7b" "qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-30b-a3b" "qwen3-235b-a22b" "gpt-5" "gpt-5-mini" "qwen3-32b" "web-rev-claude-opus-4-20250514" "deepseek-chat" "gemini-2.5-pro")

# EVAL_MODELS=("gpt-5-mini")

# Default values
DEFAULT_MODE="missing"
DEFAULT_TEMPERATURE="1.0"

# Parse command line arguments
MODE=${1:-$DEFAULT_MODE}
TEMPERATURE=${2:-$DEFAULT_TEMPERATURE}

# Validate mode
if [ "$MODE" != "resume" ] && [ "$MODE" != "missing" ]; then
    echo "Error: Invalid mode. Use 'resume' or 'missing'"
    exit 1
fi

# Show configuration
echo "Will process the following models:"
printf "  %s\n" "${EVAL_MODELS[@]}"

# Checkpoint directory
CHECKPOINT_DIR="results/eval_llm_human_correlation"

if [ "$MODE" == "resume" ]; then
    echo "Resume Checkpoint Jobs for Human-LLM Correlation Evaluation"
else
    echo "Process Missing Data for Human-LLM Correlation Evaluation"
fi
echo "=========================================================="
echo "Mode: ${MODE}"
echo "Temperature: ${TEMPERATURE}"
echo "Results Directory: ${CHECKPOINT_DIR}"
echo "=========================================================="

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory $CHECKPOINT_DIR does not exist!"
    exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Function to submit a job
submit_job() {
    local model_name=$1
    local job_name="${MODE}_correlation_${model_name}"
    
    echo "Submitting job for model: $model_name"
    
    # Create sbatch script content
    cat > "temp_${job_name}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=eval_correlation_logs/${job_name}_%j.out
#SBATCH --error=eval_correlation_logs/${job_name}_%j.err
#SBATCH --time=1-20:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

echo "Running human-LLM correlation evaluation: Model=${model_name}, Temperature=${TEMPERATURE} at \$(date)"
echo "=================================================="

if [ "$MODE" == "resume" ]; then
    # Run human-LLM correlation evaluation with resume
    python scripts/batch_human_aligned_evaluation_summaries.py \
        --model "${model_name}" \
        --temperature ${TEMPERATURE} \
        --output-dir results/eval_llm_human_correlation \
        --resume \
        --checkpoint-interval 5
else
    # Process missing data
    python scripts/batch_human_aligned_evaluation_summaries.py \
        --model "${model_name}" \
        --temperature ${TEMPERATURE} \
        --results-dir results/eval_llm_human_correlation \
        --check-coverage \
        --process-missing \
        --show-missing-ids \
        --target-model "${model_name}"
fi

# Check exit status
if [ \$? -eq 0 ]; then
    echo "=================================================="
    echo "Correlation evaluation completed successfully at \$(date)"
    echo "Results saved to: results/eval_llm_human_correlation/"
else
    echo "=================================================="
    echo "Correlation evaluation failed or was interrupted at \$(date)"
    echo "Check logs for details: eval_correlation_logs/${job_name}_*.err"
    exit 1
fi
EOF

    # Submit the job
    local job_id=$(sbatch "temp_${job_name}.sh" | awk '{print $4}')
    echo "  Submitted job: ${job_name} with ID: ${job_id}"
    
    # Clean up temporary script
    rm "temp_${job_name}.sh"
    
    return 0
}

# Process models
process_model() {
    local current_model=$1
    local job_id=""
    
    echo "Processing model: $current_model"
    
    # Submit job
    job_id=$(submit_job "$current_model" | grep -o '[0-9]\+')
    if [ -n "$job_id" ]; then
        echo "Job ID: $job_id"
        echo "Log files:"
        echo "  eval_correlation_logs/${MODE}_correlation_${current_model}_${job_id}.out"
        echo "  eval_correlation_logs/${MODE}_correlation_${current_model}_${job_id}.err"
    fi
}

# Process all models in EVAL_MODELS
echo "Processing all models..."
echo "This will submit ${#EVAL_MODELS[@]} jobs."
read -p "Continue? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    for model in "${EVAL_MODELS[@]}"; do
        echo "=================================================="
        process_model "$model"
        echo
    done
    echo "All jobs submitted. Use 'squeue -u \$USER' to check status."
else
    echo "Operation cancelled."
    exit 0
fi

echo ""
echo "Job submitted successfully!"
echo ""
echo "To check job status, run:"
echo "  squeue -u \$USER"
echo ""
echo "To monitor logs, run:"
if [ "$MODE" == "resume" ]; then
    echo "  tail -f eval_correlation_logs/resume_correlation_*_*.out"
else
    echo "  tail -f eval_correlation_logs/missing_correlation_*_*.out"
fi
