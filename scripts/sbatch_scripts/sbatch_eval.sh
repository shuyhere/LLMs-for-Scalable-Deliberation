#!/bin/bash

# Master script for submitting multiple evaluation sbatch jobs
# Usage: ./sbatch_eval.sh [model] [dataset] [email]

# Default values
DEFAULT_MODEL="all"
DEFAULT_DATASET="all"

# Parse command line arguments
MODEL=${1:-$DEFAULT_MODEL}
DATASET=${2:-$DEFAULT_DATASET}

# Available models and datasets
MODELS=("gpt-4o-mini" "gpt-5-nano" "gemini-2.5-flash-lite" "web-rev-claude-3-7-sonnet-20250219")
DATASETS=("protest" "gun_use" "operation" "bowling-green")

# Function to submit a single sbatch job
submit_job() {
    local model=$1
    local dataset=$2
    local job_name="eval_${model}_${dataset}"
    
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

echo "Starting evaluation: Model=${model}, Dataset=${dataset} at \$(date)"
echo "=================================================="

# Check if checkpoint exists and resume if possible
checkpoint_file="evalsum_logs/evaluation_checkpoint_${model}_${dataset}.json"
if [ -f "\$checkpoint_file" ]; then
    echo "Found existing checkpoint: \$checkpoint_file"
    echo "Resuming from checkpoint..."
    resume_flag="--resume"
else
    echo "No checkpoint found, starting fresh evaluation"
    resume_flag=""
fi

# Run evaluation with checkpoint support for this specific dataset
uv run scripts/batch_evaluate_summaries.py \\
    --evaluation-model ${model} \\
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
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "=================================================="

# Check if model is valid
if [[ ! " ${MODELS[@]} " =~ " ${MODEL} " ]] && [[ "$MODEL" != "all" ]]; then
    echo "Error: Invalid model '${MODEL}'"
    echo "Available models: ${MODELS[*]}"
    exit 1
fi

# Check if dataset is valid
if [[ ! " ${DATASETS[@]} " =~ " ${DATASET} " ]] && [[ "$DATASET" != "all" ]]; then
    echo "Error: Invalid dataset '${DATASET}'"
    echo "Available datasets: ${DATASETS[*]}"
    exit 1
fi

# Submit jobs based on parameters
if [[ "$MODEL" == "all" && "$DATASET" == "all" ]]; then
    # Submit jobs for ALL models and ALL datasets
    echo "Submitting jobs for ALL models and ALL datasets..."
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            submit_job "$model" "$dataset"
            sleep 1  # Small delay between submissions
        done
    done
    echo "All jobs submitted successfully!"
elif [[ "$MODEL" == "all" ]]; then
    # Submit jobs for ALL models and specific dataset
    echo "Submitting jobs for ALL models and dataset: $DATASET"
    for model in "${MODELS[@]}"; do
        submit_job "$model" "$DATASET"
        sleep 1
    done
    echo "All jobs for dataset $DATASET submitted successfully!"
elif [[ "$DATASET" == "all" ]]; then
    # Submit jobs for specific model and ALL datasets
    echo "Submitting jobs for model: $MODEL and ALL datasets"
    for dataset in "${DATASETS[@]}"; do
        submit_job "$MODEL" "$dataset"
        sleep 1
    done
    echo "All jobs for model $MODEL submitted successfully!"
else
    # Single model and dataset
    submit_job "$MODEL" "$DATASET"
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
echo "Note: Each dataset runs in a separate sbatch job for parallel processing!"
