#!/bin/bash

# Script to resume checkpoint jobs for human-LLM correlation evaluation
# Usage: ./resume_checkpoint_jobs.sh [sample_size] [temperature]

# Default values
DEFAULT_SAMPLE_SIZE="0"
DEFAULT_TEMPERATURE="1.0"

# Parse command line arguments
SAMPLE_SIZE=${1:-$DEFAULT_SAMPLE_SIZE}
TEMPERATURE=${2:-$DEFAULT_TEMPERATURE}

# Checkpoint directory
CHECKPOINT_DIR="results/eval_llm_human_correlation"

echo "Resume Checkpoint Jobs for Human-LLM Correlation Evaluation"
echo "=========================================================="
echo "Sample Size: ${SAMPLE_SIZE}"
echo "Temperature: ${TEMPERATURE}"
echo "Checkpoint Directory: ${CHECKPOINT_DIR}"
echo "=========================================================="

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory $CHECKPOINT_DIR does not exist!"
    exit 1
fi

# Find all checkpoint files
checkpoint_files=($(find "$CHECKPOINT_DIR" -name "checkpoint_*.json" -type f))

if [ ${#checkpoint_files[@]} -eq 0 ]; then
    echo "No checkpoint files found in $CHECKPOINT_DIR"
    exit 0
fi

echo "Found ${#checkpoint_files[@]} checkpoint files:"
for file in "${checkpoint_files[@]}"; do
    echo "  - $(basename "$file")"
done
echo ""

# Function to submit a resume job
submit_resume_job() {
    local checkpoint_file=$1
    local model_name=$(basename "$checkpoint_file" | sed 's/checkpoint_\(.*\)\.json/\1/')
    local job_name="resume_correlation_${model_name}_${SAMPLE_SIZE}samples"
    
    echo "Submitting resume job for model: $model_name"
    
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

echo "Resuming human-LLM correlation evaluation: Model=${model_name}, Sample_Size=${SAMPLE_SIZE}, Temperature=${TEMPERATURE} at \$(date)"
echo "=================================================="

# Check if checkpoint exists
checkpoint_file="${CHECKPOINT_DIR}/checkpoint_${model_name}.json"
if [ -f "\$checkpoint_file" ]; then
    echo "Found checkpoint: \$checkpoint_file"
    echo "Resuming from checkpoint..."
    
    # Run human-LLM correlation evaluation with resume
    python scripts/batch_human_aligned_evaluation_summaries.py \
        --model ${model_name} \
        --sample-size ${SAMPLE_SIZE} \
        --temperature ${TEMPERATURE} \
        --output-dir results/eval_llm_human_correlation \
        --resume \
        --checkpoint-interval 5
    
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
else
    echo "Error: Checkpoint file \$checkpoint_file not found!"
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

# Submit resume jobs for all checkpoint files
echo "Submitting resume jobs..."
for checkpoint_file in "${checkpoint_files[@]}"; do
    submit_resume_job "$checkpoint_file"
    sleep 1  # Small delay between submissions
done

echo ""
echo "All resume jobs submitted successfully!"
echo ""
echo "To check job status, run:"
echo "  squeue -u \$USER"
echo ""
echo "To monitor logs, run:"
echo "  tail -f eval_correlation_logs/resume_correlation_*.out"
