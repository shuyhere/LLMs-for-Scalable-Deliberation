#!/bin/bash

# Master script for submitting multiple evaluation sbatch jobs
# Usage: ./sbatch_eval.sh [evaluation_model] [summary_model] [dataset] [email]

# Default values
DEFAULT_EVAL_MODEL="all"
DEFAULT_SUMMARY_MODEL="all"
DEFAULT_DATASET="all"

# Parse command line arguments
EVAL_MODEL=${1:-$DEFAULT_EVAL_MODEL}
SUMMARY_MODEL=${2:-$DEFAULT_SUMMARY_MODEL}
DATASET=${3:-$DEFAULT_DATASET}

# Available models and datasets
EVAL_MODELS=("gpt-4o-mini" "gpt-5-nano" "gemini-2.5-flash-lite" "web-rev-claude-3-7-sonnet-20250219")
# SUMMARY_MODELS=("gpt-5-nano" "gemini-2.5-flash-lite" "web-rev-claude-3-7-sonnet-20250219" "gpt-4o-mini")
SUMMARY_MODELS=("web-rev-claude-3-7-sonnet-20250219")
DATASETS=("protest" "gun_use" "operation" "bowling-green")

# Function to submit a single sbatch job
submit_job() {
    local eval_model=$1
    local summary_model=$2
    local dataset=$3
    local job_name="eval_${eval_model}_${summary_model}_${dataset}"
    
    # Create sbatch script content
    cat > "temp_${job_name}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=evalsum_logs/${job_name}_%j.out
#SBATCH --error=evalsum_logs/${job_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Set working directory
cd /ibex/project/c2328/LLMs-Scalable-Deliberation

echo "Starting evaluation: Eval_Model=${eval_model}, Summary_Model=${summary_model}, Dataset=${dataset} at \$(date)"
echo "=================================================="

# Check if checkpoint exists and resume if possible
checkpoint_file="evalsum_logs/evaluation_checkpoint_${eval_model}_${summary_model}_${dataset}.json"
if [ -f "\$checkpoint_file" ]; then
    echo "Found existing checkpoint: \$checkpoint_file"
    echo "Resuming from checkpoint..."
    resume_flag="--resume"
else
    echo "No checkpoint found, starting fresh evaluation"
    resume_flag=""
fi

# Run evaluation with checkpoint support for this specific model and dataset
uv run scripts/batch_evaluate_summaries.py \\
    --evaluation-model ${eval_model} \\
    --summary-model ${summary_model} \\
    --dataset ${dataset} \\
    --config config/batch_evaluation_config.yaml \\
    --checkpoint "\$checkpoint_file" \\
    \$resume_flag

# Check exit status
if [ \$? -eq 0 ]; then
    echo "=================================================="
    echo "Evaluation completed successfully at \$(date)"
    echo "Checkpoint saved to: \$checkpoint_file"
else
    echo "=================================================="
    echo "Evaluation failed or was interrupted at \$(date)"
    echo "Checkpoint saved to: \$checkpoint_file"
    echo "You can resume later with: --resume flag"
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
    find evalsum_logs/ -name "evaluation_checkpoint_*.json" 2>/dev/null | head -20
    echo ""
    echo "To resume a specific job, use:"
    echo "uv run scripts/batch_evaluate_summaries.py --checkpoint evalsum_logs/evaluation_checkpoint_<model>_<dataset>.json --resume"
}

# Main execution logic
echo "LLMs-Scalable-Deliberation Evaluation Job Submitter"
echo "=================================================="
echo "Evaluation Model: ${EVAL_MODEL}"
echo "Summary Model: ${SUMMARY_MODEL}"
echo "Dataset: ${DATASET}"
echo "=================================================="

# Check if evaluation model is valid
if [[ ! " ${EVAL_MODELS[@]} " =~ " ${EVAL_MODEL} " ]] && [[ "$EVAL_MODEL" != "all" ]]; then
    echo "Error: Invalid evaluation model '${EVAL_MODEL}'"
    echo "Available evaluation models: ${EVAL_MODELS[*]}"
    exit 1
fi

# Check if summary model is valid
if [[ ! " ${SUMMARY_MODELS[@]} " =~ " ${SUMMARY_MODEL} " ]] && [[ "$SUMMARY_MODEL" != "all" ]]; then
    echo "Error: Invalid summary model '${SUMMARY_MODEL}'"
    echo "Available summary models: ${SUMMARY_MODELS[*]}"
    exit 1
fi

# Check if dataset is valid
if [[ ! " ${DATASETS[@]} " =~ " ${DATASET} " ]] && [[ "$DATASET" != "all" ]]; then
    echo "Error: Invalid dataset '${DATASET}'"
    echo "Available datasets: ${DATASETS[*]}"
    exit 1
fi

# Submit jobs based on parameters
if [[ "$EVAL_MODEL" == "all" && "$SUMMARY_MODEL" == "all" && "$DATASET" == "all" ]]; then
    # Submit jobs for ALL evaluation models, ALL summary models, and ALL datasets
    echo "Submitting jobs for ALL evaluation models, ALL summary models, and ALL datasets..."
    for eval_model in "${EVAL_MODELS[@]}"; do
        for summary_model in "${SUMMARY_MODELS[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                submit_job "$eval_model" "$summary_model" "$dataset"
                sleep 1  # Small delay between submissions
            done
        done
    done
    echo "All jobs submitted successfully!"
elif [[ "$EVAL_MODEL" == "all" && "$SUMMARY_MODEL" == "all" ]]; then
    # Submit jobs for ALL evaluation models, ALL summary models, and specific dataset
    echo "Submitting jobs for ALL evaluation models, ALL summary models, and dataset: $DATASET"
    for eval_model in "${EVAL_MODELS[@]}"; do
        for summary_model in "${SUMMARY_MODELS[@]}"; do
            submit_job "$eval_model" "$summary_model" "$DATASET"
            sleep 1
        done
    done
    echo "All jobs for dataset $DATASET submitted successfully!"
elif [[ "$SUMMARY_MODEL" == "all" && "$DATASET" == "all" ]]; then
    # Submit jobs for specific evaluation model, ALL summary models, and ALL datasets
    echo "Submitting jobs for evaluation model: $EVAL_MODEL, ALL summary models, and ALL datasets"
    for summary_model in "${SUMMARY_MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            submit_job "$EVAL_MODEL" "$summary_model" "$dataset"
            sleep 1
        done
    done
    echo "All jobs for evaluation model $EVAL_MODEL submitted successfully!"
elif [[ "$SUMMARY_MODEL" == "all" ]]; then
    # Submit jobs for specific evaluation model, ALL summary models, and specific dataset
    echo "Submitting jobs for evaluation model: $EVAL_MODEL, ALL summary models, and dataset: $DATASET"
    for summary_model in "${SUMMARY_MODELS[@]}"; do
        submit_job "$EVAL_MODEL" "$summary_model" "$DATASET"
        sleep 1
    done
    echo "All jobs for evaluation model $EVAL_MODEL and dataset $DATASET submitted successfully!"
elif [[ "$DATASET" == "all" ]]; then
    # Submit jobs for specific evaluation model, specific summary model, and ALL datasets
    echo "Submitting jobs for evaluation model: $EVAL_MODEL, summary model: $SUMMARY_MODEL, and ALL datasets"
    for dataset in "${DATASETS[@]}"; do
        submit_job "$EVAL_MODEL" "$SUMMARY_MODEL" "$dataset"
        sleep 1
    done
    echo "All jobs for evaluation model $EVAL_MODEL and summary model $SUMMARY_MODEL submitted successfully!"
else
    # Single evaluation model, summary model, and dataset
    submit_job "$EVAL_MODEL" "$SUMMARY_MODEL" "$DATASET"
    echo "Single job submitted successfully!"
fi

echo ""
echo "Use 'squeue -u $USER' to check job status"
echo "Use 'scancel <job_id>' to cancel a specific job"
echo "Use 'scancel -u $USER' to cancel all your jobs"
echo ""
echo "Checkpoint files are saved in evalsum_logs/ directory"
echo "Use --resume flag to continue interrupted evaluations"
echo ""
echo "Quick commands:"
echo "  ./sbatch_eval.sh status    # Check job status"
echo "  ./sbatch_eval.sh checkpoints # Show available checkpoints"
echo ""
echo "Usage examples:"
echo "  ./sbatch_eval.sh gpt-4o-mini gpt-5-nano protest    # Evaluate gpt-5-nano summaries with gpt-4o-mini"
echo "  ./sbatch_eval.sh gpt-5-nano all all                # Evaluate all summary models with gpt-5-nano"
echo "  ./sbatch_eval.sh all gpt-5-nano all                # Evaluate gpt-5-nano summaries with all evaluation models"
echo ""
echo "Note: Each combination runs in a separate sbatch job for parallel processing!"
