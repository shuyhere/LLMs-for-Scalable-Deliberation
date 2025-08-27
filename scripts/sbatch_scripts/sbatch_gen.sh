#!/bin/bash

# Optimized batch summarization script for LLMs-Scalable-Deliberation with SLURM job submission
# Supports multiple models and datasets with parallel job submission
# Usage: ./run_batch_summarization.sh [options]

set -e

# Default values
MODELS=("gpt-5-nano" "gemini-2.5-flash-lite" "web-rev-claude-3-7-sonnet-20250219" "gpt-4o-mini")
DATASETS=("datasets/protest.json" "datasets/bowling-green.json" "datasets/gun_use.json" "datasets/operation.json" "datasets/GenAI.json" "datasets/LouisvilleCivicAssembly.json")
NUM_SAMPLES=210
OUTPUT_BASE_DIR="results/summary/${NUM_SAMPLES}"
DEBUG_MODE=""
DRY_RUN=""
MAX_PARALLEL_JOBS=5
SLURM_TIME="12:00:00"
SLURM_MEMORY="32G"
SLURM_CPUS=4
SLURM_GPUS=""
SLURM_PARTITION=""
SKIP_EXISTING=""
FORCE_REGENERATE=""
CUSTOM_SYSTEM_PROMPT=""
CUSTOM_USER_PROMPT=""

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --models MODELS          Comma-separated list of models or 'all' for default models"
    echo "  -d, --datasets DATASETS      Comma-separated list of dataset paths or 'all' for default datasets"
    echo "  -n, --num-samples N          Number of comments to use per dataset (default: 100)"
    echo "  -o, --output-dir DIR         Base output directory (default: results/summary)"
    echo "  --debug                      Debug mode (only first 10 samples)"
    echo "  --skip-existing              Skip jobs if output files already exist"
    echo "  --force-regenerate           Force regeneration even if outputs exist"
    echo "  --dry-run                    Show what would be submitted without actually submitting"
    echo "  --max-jobs N                 Maximum parallel jobs (default: 5)"
    echo "  --time TIME                  Job time limit (default: 12:00:00)"
    echo "  --memory MEM                 Memory per job (default: 32G)"
    echo "  --cpus N                     CPUs per job (default: 4)"
    echo "  --gpus N                     GPUs per job (default: none)"
    echo "  --partition PART             SLURM partition to use"
    echo "  --custom-system-prompt PROMPT Custom system prompt for analysis"
    echo "  --custom-user-prompt PROMPT   Custom user prompt template for analysis"
    echo "  -h, --help                   Show this help"
    echo ""
    echo "Default models:"
    printf "  %s\n" "${MODELS[@]}"
    echo ""
    echo "Default datasets:"
    printf "  %s\n" "${DATASETS[@]}"
    echo ""
    echo "Examples:"
    echo "  $0 --debug --dry-run                          # Test run"
    echo "  $0 -m gpt-4o-mini,gpt-3.5-turbo -d all       # Specific models, all datasets"
    echo "  $0 --max-jobs 10 --skip-existing              # More parallel jobs, skip existing"
    echo "  $0 --custom-system-prompt 'Custom prompt'      # Use custom system prompt"
}

# Function to check if output exists
check_output_exists() {
    local output_dir="$1"
    local model="$2"
    local dataset_name="$3"
    
    local model_dir="$output_dir/$model"
    local dataset_dir="$model_dir/$dataset_name"
    local summary_file="$dataset_dir/summary_${dataset_name}.json"
    
    [[ -f "$summary_file" ]]
}

# Parse command line arguments
MODELS_INPUT=""
DATASETS_INPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS_INPUT="$2"
            shift 2
            ;;
        -d|--datasets)
            DATASETS_INPUT="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="--debug"
            NUM_SAMPLES=10
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING="true"
            shift
            ;;
        --force-regenerate)
            FORCE_REGENERATE="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --max-jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --memory)
            SLURM_MEMORY="$2"
            shift 2
            ;;
        --cpus)
            SLURM_CPUS="$2"
            shift 2
            ;;
        --gpus)
            SLURM_GPUS="--gres=gpu:$2"
            shift 2
            ;;
        --partition)
            SLURM_PARTITION="--partition=$2"
            shift 2
            ;;
        --custom-system-prompt)
            CUSTOM_SYSTEM_PROMPT="$2"
            shift 2
            ;;
        --custom-user-prompt)
            CUSTOM_USER_PROMPT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set default values if not provided
if [[ -z "$MODELS_INPUT" ]]; then
    MODELS_INPUT="all"
fi

if [[ -z "$DATASETS_INPUT" ]]; then
    DATASETS_INPUT="all"
fi

# Parse models
if [[ "$MODELS_INPUT" == "all" ]]; then
    MODELS_TO_USE=("${MODELS[@]}")
else
    IFS=',' read -ra MODELS_TO_USE <<< "$MODELS_INPUT"
fi

# Parse datasets
if [[ "$DATASETS_INPUT" == "all" ]]; then
    DATASETS_TO_USE=("${DATASETS[@]}")
else
    IFS=',' read -ra DATASETS_TO_USE <<< "$DATASETS_INPUT"
fi

# Create timestamp for this batch run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_OUTPUT_DIR="${OUTPUT_BASE_DIR}/"

echo "=========================================="
echo "LLMs-Scalable-Deliberation Batch Summarization"
echo "=========================================="
echo "Models: ${MODELS_TO_USE[*]}"
echo "Datasets: ${DATASETS_TO_USE[*]}"
echo "Number of samples: $NUM_SAMPLES"
echo "Base Output Dir: $BATCH_OUTPUT_DIR"
echo "Debug Mode: ${DEBUG_MODE:-"Disabled"}"
echo "Skip Existing: ${SKIP_EXISTING:-"No"}"
echo "Force Regenerate: ${FORCE_REGENERATE:-"No"}"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "Dry Run: ${DRY_RUN:-"No"}"
echo "=========================================="

# Create batch output directory
mkdir -p "$BATCH_OUTPUT_DIR"
LOG_DIR="results/logs/batch_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
# Create temporary job submission script template
TEMP_SCRIPTS_DIR=$(mktemp -d)
JOB_SCRIPT_TEMPLATE="$TEMP_SCRIPTS_DIR/job_template.sh"
cat > "$JOB_SCRIPT_TEMPLATE" << 'EOF'
#!/bin/bash
#SBATCH --job-name=SUMMARY_{MODEL}_{DATASET}
#SBATCH --time={TIME}
#SBATCH --mem={MEMORY}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --output={LOG_DIR}/job_%j_{MODEL}_{DATASET}.out
#SBATCH --error={LOG_DIR}/job_%j_{MODEL}_{DATASET}.err
{GPUS_LINE}
{PARTITION_LINE}

set -e

echo "=========================================="
echo "Starting Summarization Job"
echo "Model: {MODEL}"
echo "Dataset: {DATASET}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Change to script directory
cd {SCRIPT_DIR}

# Set up environment
# module load python/3.9
# source .venv/bin/activate

# Check if output already exists (unless force regenerate)
if [[ "{FORCE_REGENERATE}" != "true" ]] && [[ -n "{SKIP_EXISTING}" ]]; then
    if check_output_exists "{OUTPUT_DIR}" "{MODEL}" "{DATASET_NAME}"; then
        echo "Output already exists, skipping..."
        exit 0
    fi
fi

# Run summarization
python3 scripts/batch_gen_summarization.py \
    --model "{MODEL}" \
    --datasets "{DATASET}" \
    --num-samples {NUM_SAMPLES} \
    --output-dir "{OUTPUT_DIR}" \
    {CUSTOM_SYSTEM_PROMPT_LINE} \
    {CUSTOM_USER_PROMPT_LINE} \
    {DEBUG_FLAG}

echo "=========================================="
echo "Job completed successfully"
echo "Results saved to: {OUTPUT_DIR}"
echo "=========================================="
EOF

# Function to wait for jobs
wait_for_jobs() {
    local max_jobs="$1"
    local job_pattern="$2"
    
    while true; do
        local running_jobs=$(squeue -u $USER --name="$job_pattern" --states=RUNNING,PENDING 2>/dev/null | wc -l)
        running_jobs=$((running_jobs - 1))  # Subtract header line
        if [[ $running_jobs -lt 0 ]]; then
            running_jobs=0
        fi
        
        if [[ $running_jobs -lt $max_jobs ]]; then
            break
        fi
        
        echo "Waiting for jobs to complete (current: $running_jobs, max: $max_jobs)..."
        sleep 10
    done
}

# Function to submit job
submit_job() {
    local model="$1"
    local dataset="$2"
    local dataset_name="$3"
    local output_dir="$4"
    
    # Create temporary job script from template
    local job_script="${TEMP_SCRIPTS_DIR}/job_${model//[^a-zA-Z0-9]/_}_${dataset_name}.sh"
    
    # Prepare custom prompt lines
    local custom_system_prompt_line=""
    local custom_user_prompt_line=""
    
    if [[ -n "$CUSTOM_SYSTEM_PROMPT" ]]; then
        custom_system_prompt_line="--custom-system-prompt \"$CUSTOM_SYSTEM_PROMPT\""
    fi
    
    if [[ -n "$CUSTOM_USER_PROMPT" ]]; then
        custom_user_prompt_line="--custom-user-prompt \"$CUSTOM_USER_PROMPT\""
    fi
    
    # Replace placeholders in template - use proper escaping for sed
    sed -e "s|{MODEL}|${model}|g" \
        -e "s|{DATASET}|${dataset}|g" \
        -e "s|{DATASET_NAME}|${dataset_name}|g" \
        -e "s|{NUM_SAMPLES}|${NUM_SAMPLES}|g" \
        -e "s|{TIME}|${SLURM_TIME}|g" \
        -e "s|{MEMORY}|${SLURM_MEMORY}|g" \
        -e "s|{CPUS}|${SLURM_CPUS}|g" \
        -e "s|{OUTPUT_DIR}|${output_dir}|g" \
        -e "s|{SCRIPT_DIR}|$(pwd)|g" \
        -e "s|{LOG_DIR}|${BATCH_OUTPUT_DIR}/logs|g" \
        -e "s|{DEBUG_FLAG}|${DEBUG_MODE}|g" \
        -e "s|{SKIP_EXISTING}|${SKIP_EXISTING}|g" \
        -e "s|{FORCE_REGENERATE}|${FORCE_REGENERATE}|g" \
        -e "s|{CUSTOM_SYSTEM_PROMPT_LINE}|${custom_system_prompt_line}|g" \
        -e "s|{CUSTOM_USER_PROMPT_LINE}|${custom_user_prompt_line}|g" \
        -e "s|{GPUS_LINE}|${SLURM_GPUS}|g" \
        -e "s|{PARTITION_LINE}|${SLURM_PARTITION}|g" \
        "$JOB_SCRIPT_TEMPLATE" > "$job_script"
    
    chmod +x "$job_script"
    
    # Check if output exists and should skip
    if [[ "$SKIP_EXISTING" == "true" && "$FORCE_REGENERATE" != "true" ]]; then
        if check_output_exists "$output_dir" "$model" "$dataset_name"; then
            echo "  Output already exists, skipping..."
            return 0
        fi
    fi
    
    # Submit job
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  DRY RUN: Would submit job for $model + $dataset_name"
        echo "    Script: $job_script"
        echo "    Output: $output_dir"
        return 0
    else
        local job_id=$(sbatch "$job_script" | grep -o '[0-9]*')
        echo "    Job ID: $job_id"
        sleep 1  # Small delay to avoid overwhelming scheduler
        return 0
    fi
}

# Arrays to track submitted jobs
SUBMITTED_JOBS=()
TOTAL_JOBS=0

echo ""
echo "=========================================="
echo "Submitting Summarization Jobs"
echo "=========================================="

# Submit jobs for each model-dataset combination
for model in "${MODELS_TO_USE[@]}"; do
    for dataset in "${DATASETS_TO_USE[@]}"; do
        # Extract dataset name from path
        dataset_name=$(basename "$dataset" .json)
        
        echo "Submitting job for ${model} + ${dataset_name}"
        
        # Wait if we've reached max parallel jobs
        wait_for_jobs "$MAX_PARALLEL_JOBS" "SUMMARY_*"
        
        # Submit job
        submit_job "$model" "$dataset" "$dataset_name" "$BATCH_OUTPUT_DIR"
        
        SUBMITTED_JOBS+=("${model}_${dataset_name}")
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo ""
echo "=========================================="
echo "Batch Submission Summary"
echo "=========================================="
echo "Total jobs: $TOTAL_JOBS"
echo "Models: ${#MODELS_TO_USE[@]}"
echo "Datasets: ${#DATASETS_TO_USE[@]}"
echo "Base output directory: $BATCH_OUTPUT_DIR"

if [[ "$DRY_RUN" != "true" ]]; then
    echo ""
    echo "Job submission completed successfully!"
    echo "To monitor jobs, use: squeue -u \$USER --name='SUMMARY_*'"
    echo "To check results, examine the output directory: $BATCH_OUTPUT_DIR"
    echo ""
    echo "Job naming convention: SUMMARY_{MODEL}_{DATASET}"
    echo "Log files: $BATCH_OUTPUT_DIR/logs/"
else
    echo ""
    echo "This was a dry run. Remove --dry-run to submit actual jobs."
fi

# Clean up temporary directory
if [[ -d "$TEMP_SCRIPTS_DIR" ]]; then
    rm -rf "$TEMP_SCRIPTS_DIR"
fi

echo "=========================================="