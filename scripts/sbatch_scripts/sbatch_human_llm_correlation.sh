#!/bin/bash

# Master script for submitting multiple human-LLM correlation evaluation sbatch jobs
# Usage: ./sbatch_human_llm_correlation.sh [evaluation_model] [sample_size] [email]

# Default values
DEFAULT_EVAL_MODEL="all"
DEFAULT_SAMPLE_SIZE="0"
DEFAULT_TEMPERATURE="1.0"

# Parse command line arguments
EVAL_MODEL=${1:-$DEFAULT_EVAL_MODEL}
SAMPLE_SIZE=${2:-$DEFAULT_SAMPLE_SIZE}
TEMPERATURE=${3:-$DEFAULT_TEMPERATURE}

EVAL_MODELS=("gpt-4o-mini" "gemini-2.5-flash-lite"  "web-rev-claude-3-7-sonnet-20250219" "web-rev-claude-sonnet-4-20250514" "gemini-2.5-flash" "deepseek-reasoner" "grok-4-latest" "TA/openai/gpt-oss-120b" "TA/openai/gpt-oss-20b" "qwen3-0.6b" "qwen3-1.7b" "qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-30b-a3b" "qwen3-235b-a22b" "gpt-5" "qwen3-32b" "web-rev-claude-opus-4-20250514" "deepseek-chat" "gemini-2.5-pro")

# EVAL_MODELS=("gpt-5-nano" "gpt-5-mini")


# Ensure logs directory exists
mkdir -p eval_correlation_logs

# Function to submit a single sbatch job
submit_job() {
    local eval_model=$1
    local sample_size=$2
    local temperature=$3
    local job_name="correlation_${eval_model}_${sample_size}samples"
    
    # Create sbatch script content
    cat > "temp_${job_name}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=eval_correlation_logs/${job_name}_%j.out
#SBATCH --error=eval_correlation_logs/${job_name}_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Set working directory
cd /ibex/project/c2328/LLMs-Scalable-Deliberation

echo "Starting human-LLM correlation evaluation: Model=${eval_model}, Sample_Size=${sample_size}, Temperature=${temperature} at \$(date)"
echo "=================================================="

# Check if checkpoint exists and resume if possible
checkpoint_file="results/eval_llm_human_correlation/checkpoint_${eval_model}.json"
if [ -f "\$checkpoint_file" ]; then
    echo "Found existing checkpoint: \$checkpoint_file"
    echo "Resuming from checkpoint..."
    resume_flag="--resume"
else
    echo "No checkpoint found, starting fresh evaluation"
    resume_flag=""
fi

# Run human-LLM correlation evaluation with debug mode
python scripts/batch_human_aligned_evaluation_summaries.py \
    --model ${eval_model} \
    --sample-size ${sample_size} \
    --temperature ${temperature} \
    --output-dir results/eval_llm_human_correlation \
    \$resume_flag

# Check exit status
if [ \$? -eq 0 ]; then
    echo "=================================================="
    echo "Correlation evaluation completed successfully at \$(date)"
    echo "Results saved to: results/eval_llm_human_correlation/"
else
    echo "=================================================="
    echo "Correlation evaluation failed or was interrupted at \$(date)"
    echo "Check logs for details: correlation_logs/${job_name}_*.err"
    exit 1
fi
EOF

    # Submit the job
    local job_id=$(sbatch "temp_${job_name}.sh" | awk '{print $4}')
    echo "Submitted job: ${job_name} with ID: ${job_id}"
    
    # Clean up temporary script
    rm "temp_${job_name}.sh"
    
    return 0
}

# Function to check job status
check_job_status() {
    echo "Checking job status..."
    squeue -u $USER
}

# Function to show checkpoint files
show_checkpoints() {
    echo "Available checkpoint files:"
    find results/eval_llm_human_correlation/ -name "checkpoint_*.json" 2>/dev/null | head -20
    echo ""
    echo "To resume a specific job, use:"
    echo "python scripts/batch_human_aligned_evaluation_summaries.py --model <model> --sample-size <size> --resume"
}

# Function to show results
show_results() {
    echo "Available correlation results:"
    find results/llm_human_correlation/ -name "human_llm_correlation_*.json" 2>/dev/null | head -20
    echo ""
    echo "To analyze results, use:"
    echo "python scripts/analyze_correlation_results.py"
}

# Main execution logic
echo "Human-LLM Correlation Evaluation Job Submitter"
echo "=============================================="
echo "Evaluation Model: ${EVAL_MODEL}"
echo "Sample Size: ${SAMPLE_SIZE}"
echo "Temperature: ${TEMPERATURE}"
echo "=============================================="

# Check if evaluation model is valid
if [[ ! " ${EVAL_MODELS[@]} " =~ " ${EVAL_MODEL} " ]] && [[ "$EVAL_MODEL" != "all" ]]; then
    echo "Error: Invalid evaluation model '${EVAL_MODEL}'"
    echo "Available evaluation models: ${EVAL_MODELS[*]}"
    exit 1
fi

# Check if sample size is valid
if ! [[ "$SAMPLE_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: Sample size must be a positive integer"
    exit 1
fi

# Submit jobs based on parameters
if [[ "$EVAL_MODEL" == "all" ]]; then
    # Submit jobs for ALL evaluation models
    echo "Submitting jobs for ALL evaluation models..."
    for eval_model in "${EVAL_MODELS[@]}"; do
        submit_job "$eval_model" "$SAMPLE_SIZE" "$TEMPERATURE"
        sleep 1  # Small delay between submissions
    done
    echo "All jobs submitted successfully!"
else
    # Single evaluation model
    submit_job "$EVAL_MODEL" "$SAMPLE_SIZE" "$TEMPERATURE"
    echo "Single job submitted successfully!"
fi