#!/bin/bash

# Optimized batch summarization script for LLMs-Scalable-Deliberation with SLURM job submission
# Supports multiple models and datasets with parallel job submission
# Usage: ./run_batch_summarization.sh [options]

set -e

# Default values
MODELS=("gpt-5-mini" "web-rev-claude-sonnet-4-20250514" "gemini-2.5-flash" "deepseek-reasoner" "grok-4-latest" "TA/openai/gpt-oss-120b" "TA/openai/gpt-oss-20b" "qwen3-0.6b" "qwen3-1.7b" "qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-30b-a3b" "qwen3-235b-a22b" "qwen3-32b" "web-rev-claude-opus-4-20250514" "deepseek-chat" "gemini-2.5-pro" "gpt-5" "gpt-4o-mini")

# MODELS=( "qwen3-0.6b" "qwen3-1.7b" )



# MODELS=("gpt-4o-mini")

# MODELS=("gpt-5")
MODELS=("gpt-5" "qwen3-32b" "web-rev-claude-opus-4-20250514" "deepseek-chat" "gemini-2.5-pro")

# DATASETS=("/ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/minority/Binary-Tariff_Policy.json"
# /ibex/project/c2328/LLMs-Scalable-Deliberation/datasets/minority/Openqa-AI_changes_human_life.json
# )
DATASETS=("datasets/annotation_data_old_backup/previous/gun_use.json")

NUM_SAMPLES_ARRAY=(10 20 30 50 90)
SAMPLE_TIMES=3
OUTPUT_BASE_DIR="results/dataset_construction_for_ood_annotation"
DEBUG_MODE=""
DRY_RUN=""
MAX_PARALLEL_JOBS=50
SLURM_TIME="12:00:00"
SLURM_MEMORY="32G"
SLURM_CPUS=4
SLURM_GPUS=""
SLURM_PARTITION=""
SKIP_EXISTING=""
FORCE_REGENERATE=""
CUSTOM_SYSTEM_PROMPT=""
CUSTOM_USER_PROMPT=""
MISSING_ONLY=""
MISSING_FILE=""

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --models MODELS          Comma-separated list of models or 'all' for default models"
    echo "  -d, --datasets DATASETS      Comma-separated list of dataset paths or 'all' for default datasets"
    echo "  -n, --num-samples N          Number of comments to use per dataset (comma-separated or 'all' for default range)"
    echo "  -s, --sample-times N         Number of times to sample with each model (default: 1)"
    echo "  -o, --output-dir DIR         Base output directory (default: results/summary)"
    echo "  --debug                      Debug mode (only first 10 samples)"
    echo "  --skip-existing              Skip jobs if output files already exist"
    echo "  --force-regenerate           Force regeneration even if outputs exist"
    echo "  --dry-run                    Show what would be submitted without actually submitting"
    echo "  --max-jobs N                 Maximum parallel jobs (default: 50)"
    echo "  --batch-size N               Jobs per batch (default: 10)"
    echo "  --time TIME                  Job time limit (default: 12:00:00)"
    echo "  --memory MEM                 Memory per job (default: 32G)"
    echo "  --cpus N                     CPUs per job (default: 4)"
    echo "  --gpus N                     GPUs per job (default: none)"
    echo "  --partition PART             SLURM partition to use"
    echo "  --custom-system-prompt PROMPT Custom system prompt for analysis"
    echo "  --custom-user-prompt PROMPT   Custom user prompt template for analysis"
    echo "  --missing-only               Only submit jobs for missing data (requires --models)"
    echo "  --missing-file FILE          Use specific missing data file (default: auto-generate)"
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
    echo "  $0 -s 3 --max-jobs 10 --skip-existing         # 3 samples per model, more parallel jobs"
    echo "  $0 --custom-system-prompt 'Custom prompt'      # Use custom system prompt"
    echo "  $0 --missing-only -m qwen3-1.7b,qwen3-0.6b   # Only submit missing data for specific models"
}

# Function to check if output exists
check_output_exists() {
    local output_dir="$1"
    local model="$2"
    local num_samples="$3"
    local dataset_name="$4"
    local sample_id="$5"
    
    local model_dir="$output_dir/$model"
    local num_samples_dir="$model_dir/$num_samples"
    local summary_file="$num_samples_dir/${dataset_name}_summary_${sample_id}.json"
    
    [[ -f "$summary_file" ]]
}

# Parse command line arguments
MODELS_INPUT=""
DATASETS_INPUT=""
NUM_SAMPLES_INPUT=""

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
            NUM_SAMPLES_INPUT="$2"
            shift 2
            ;;
        -s|--sample-times)
            SAMPLE_TIMES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE="--debug"
            NUM_SAMPLES_ARRAY=(10)
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
        --batch-size)
            BATCH_SIZE="$2"
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
        --missing-only)
            MISSING_ONLY="true"
            shift
            ;;
        --missing-file)
            MISSING_FILE="$2"
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

# Parse num_samples
if [[ -z "$NUM_SAMPLES_INPUT" ]]; then
    NUM_SAMPLES_INPUT="all"
fi

if [[ "$NUM_SAMPLES_INPUT" == "all" ]]; then
    NUM_SAMPLES_TO_USE=("${NUM_SAMPLES_ARRAY[@]}")
else
    IFS=',' read -ra NUM_SAMPLES_TO_USE <<< "$NUM_SAMPLES_INPUT"
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

# Handle missing-only mode
if [[ "$MISSING_ONLY" == "true" ]]; then
    if [[ "$MODELS_INPUT" == "all" ]]; then
        echo "Error: --missing-only requires specific models to be specified with -m/--models"
        echo "Example: $0 --missing-only -m qwen3-1.7b,qwen3-0.6b"
        exit 1
    fi
    
    echo "=========================================="
    echo "MISSING DATA ANALYSIS MODE"
    echo "=========================================="
    echo "Analyzing missing data for models: ${MODELS_TO_USE[*]}"
    
    # Generate missing data file if not provided
    if [[ -z "$MISSING_FILE" ]]; then
        MISSING_FILE="/tmp/missing_data_$(date +%Y%m%d_%H%M%S).txt"
        echo "Generating missing data analysis..."
        
        # Use the check_missing_details.py script to generate missing data
        python3 scripts/check_missing_details.py --results-dir "$OUTPUT_BASE_DIR" --models "${MODELS_TO_USE[*]}" > "$MISSING_FILE" 2>&1
        
        if [[ $? -ne 0 ]]; then
            echo "Error: Failed to analyze missing data"
            cat "$MISSING_FILE"
            exit 1
        fi
        
        echo "Missing data analysis saved to: $MISSING_FILE"
    fi
    
    # Parse missing data and create targeted job combinations
    echo "Parsing missing data to create targeted job submissions..."
    
    # Create a temporary file for missing combinations
    MISSING_COMBINATIONS_FILE="/tmp/missing_combinations_$(date +%Y%m%d_%H%M%S).txt"
    
    # Extract missing combinations from the analysis
    # This is a simplified approach - in practice, you might want to parse the Python script output more carefully
    python3 -c "
import sys
import os
sys.path.append('scripts')
from check_missing_details import analyze_model_missing_data, get_expected_combinations

# Get expected combinations
expected_models, expected_datasets, expected_sample_counts, expected_sample_times = get_expected_combinations()

# Parse the missing data analysis
missing_combinations = []
models_to_check = '${MODELS_TO_USE[*]}'.split()

for model in models_to_check:
    if model not in expected_models:
        continue
        
    # Check what's missing for this model
    model_dir = '${OUTPUT_BASE_DIR}/' + model
    if not os.path.exists(model_dir):
        # All combinations are missing
        for dataset in expected_datasets:
            for sample_count in expected_sample_counts:
                for sample_time in expected_sample_times:
                    missing_combinations.append(f'{model}|{dataset}|{sample_count}|{sample_time}')
        continue
    
    # Check each expected combination
    for dataset in expected_datasets:
        for sample_count in expected_sample_counts:
            for sample_time in expected_sample_times:
                expected_file = f'{dataset}_summary_{sample_time}.json'
                file_path = os.path.join(model_dir, str(sample_count), expected_file)
                
                if not os.path.exists(file_path):
                    missing_combinations.append(f'{model}|{dataset}|{sample_count}|{sample_time}')
                else:
                    # Check if file has main_points
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'summaries' not in data or 'main_points' not in data['summaries']:
                                missing_combinations.append(f'{model}|{dataset}|{sample_count}|{sample_time}')
                    except:
                        missing_combinations.append(f'{model}|{dataset}|{sample_count}|{sample_time}')

# Write missing combinations to file
with open('${MISSING_COMBINATIONS_FILE}', 'w') as f:
    for combo in missing_combinations:
        f.write(combo + '\n')

print(f'Found {len(missing_combinations)} missing combinations')
" > /tmp/missing_analysis.log 2>&1
    
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to parse missing data"
        cat /tmp/missing_analysis.log
        exit 1
    fi
    
    # Read missing combinations
    if [[ -f "$MISSING_COMBINATIONS_FILE" ]]; then
        MISSING_COUNT=$(wc -l < "$MISSING_COMBINATIONS_FILE")
        echo "Found $MISSING_COUNT missing combinations"
        
        if [[ $MISSING_COUNT -eq 0 ]]; then
            echo "No missing data found for the specified models!"
            exit 0
        fi
    else
        echo "Error: Missing combinations file not created"
        exit 1
    fi
fi

# Create timestamp for this batch run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_OUTPUT_DIR="${OUTPUT_BASE_DIR}/"

echo "=========================================="
echo "LLMs-Scalable-Deliberation Batch Summarization"
echo "=========================================="
echo "Models: ${MODELS_TO_USE[*]}"
echo "Datasets: ${DATASETS_TO_USE[*]}"
echo "Number of samples: ${NUM_SAMPLES_TO_USE[*]}"
echo "Sample times: $SAMPLE_TIMES"
echo "Base Output Dir: $BATCH_OUTPUT_DIR"
echo "Debug Mode: ${DEBUG_MODE:-"Disabled"}"
echo "Skip Existing: ${SKIP_EXISTING:-"No"}"
echo "Force Regenerate: ${FORCE_REGENERATE:-"No"}"
echo "Max Parallel Jobs: $MAX_PARALLEL_JOBS"
echo "Batch Size: $BATCH_SIZE"
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
#SBATCH --job-name=SUMMARY_{MODEL}_{DATASET}_S{SAMPLE_ID}
#SBATCH --time={TIME}
#SBATCH --mem={MEMORY}
#SBATCH --cpus-per-task={CPUS}
#SBATCH --output={LOG_DIR}/job_%j_{MODEL}_{DATASET}_S{SAMPLE_ID}.out
#SBATCH --error={LOG_DIR}/job_%j_{MODEL}_{DATASET}_S{SAMPLE_ID}.err
{GPUS_LINE}
{PARTITION_LINE}

set -e

echo "=========================================="
echo "Starting Summarization Job"
echo "Model: {MODEL}"
echo "Dataset: {DATASET}"
echo "Sample ID: {SAMPLE_ID}"
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
    if check_output_exists "{OUTPUT_DIR}" "{MODEL}" "{NUM_SAMPLES}" "{DATASET_NAME}" "{SAMPLE_ID}"; then
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
    --sample-id {SAMPLE_ID} \
    {CUSTOM_SYSTEM_PROMPT_LINE} \
    {CUSTOM_USER_PROMPT_LINE} \
    {DEBUG_FLAG}

echo "=========================================="
echo "Job completed successfully"
echo "Results saved to: {OUTPUT_DIR}"
echo "=========================================="
EOF

# Function to wait for jobs (optimized)
wait_for_jobs() {
    local max_jobs="$1"
    local job_pattern="$2"
    
    # Check if we need to wait at all
    local running_jobs=$(squeue -u $USER --name="$job_pattern" --states=RUNNING,PENDING --noheader 2>/dev/null | wc -l)
    
    if [[ $running_jobs -lt $max_jobs ]]; then
        return 0  # No need to wait
    fi
    
    echo "Waiting for jobs to complete (current: $running_jobs, max: $max_jobs)..."
    
    # Optimized waiting with exponential backoff
    local wait_time=2
    while true; do
        sleep $wait_time
        running_jobs=$(squeue -u $USER --name="$job_pattern" --states=RUNNING,PENDING --noheader 2>/dev/null | wc -l)
        
        if [[ $running_jobs -lt $max_jobs ]]; then
            break
        fi
        
        # Exponential backoff, max 30 seconds
        wait_time=$((wait_time * 2))
        if [[ $wait_time -gt 30 ]]; then
            wait_time=30
        fi
        
        echo "Still waiting... (current: $running_jobs, max: $max_jobs, next check in ${wait_time}s)"
    done
}

# Function to submit job
submit_job() {
    local model="$1"
    local dataset="$2"
    local dataset_name="$3"
    local num_samples="$4"
    local output_dir="$5"
    local sample_id="$6"
    
    # Create temporary job script from template
    local job_script="${TEMP_SCRIPTS_DIR}/job_${model//[^a-zA-Z0-9]/_}_${dataset_name}_${num_samples}_s${sample_id}.sh"
    
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
        -e "s|{NUM_SAMPLES}|${num_samples}|g" \
        -e "s|{SAMPLE_ID}|${sample_id}|g" \
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
        if check_output_exists "$output_dir" "$model" "$num_samples" "$dataset_name" "$sample_id"; then
            echo "  Output already exists, skipping..."
            return 0
        fi
    fi
    
    # Submit job
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  DRY RUN: Would submit job for $model + $dataset_name + $num_samples samples (sample $sample_id)"
        echo "    Script: $job_script"
        echo "    Output: $output_dir"
        return 0
    else
        local job_id=$(sbatch "$job_script" | grep -o '[0-9]*')
        echo "    Job ID: $job_id"
        # Removed sleep delay for faster submission
        return 0
    fi
}

# Arrays to track submitted jobs
SUBMITTED_JOBS=()
TOTAL_JOBS=0
BATCH_SIZE=10  # Submit jobs in batches for better performance

echo ""
echo "=========================================="
echo "Submitting Summarization Jobs"
echo "=========================================="

# Collect all job combinations first
JOB_COMBINATIONS=()

if [[ "$MISSING_ONLY" == "true" ]]; then
    # Use missing combinations from the analysis
    echo "Loading missing combinations from: $MISSING_COMBINATIONS_FILE"
    
    while IFS='|' read -r model dataset_name sample_count sample_time; do
        # Find the corresponding dataset path
        dataset_path=""
        for dataset in "${DATASETS[@]}"; do
            if [[ "$(basename "$dataset" .json)" == "$dataset_name" ]]; then
                dataset_path="$dataset"
                break
            fi
        done
        
        if [[ -n "$dataset_path" ]]; then
            JOB_COMBINATIONS+=("${model}|${dataset_path}|${dataset_name}|${sample_count}|${sample_time}")
        else
            echo "Warning: Could not find dataset path for $dataset_name"
        fi
    done < "$MISSING_COMBINATIONS_FILE"
    
    echo "Loaded ${#JOB_COMBINATIONS[@]} missing combinations"
else
    # Use all combinations as before
    for model in "${MODELS_TO_USE[@]}"; do
        for dataset in "${DATASETS_TO_USE[@]}"; do
            for num_samples in "${NUM_SAMPLES_TO_USE[@]}"; do
                for sample_id in $(seq 1 $SAMPLE_TIMES); do
                    dataset_name=$(basename "$dataset" .json)
                    JOB_COMBINATIONS+=("${model}|${dataset}|${dataset_name}|${num_samples}|${sample_id}")
                done
            done
        done
    done
fi

echo "Total job combinations: ${#JOB_COMBINATIONS[@]}"
echo "Submitting in batches of $BATCH_SIZE..."

# Submit jobs in batches
for ((i=0; i<${#JOB_COMBINATIONS[@]}; i+=BATCH_SIZE)); do
    echo ""
    echo "Submitting batch $((i/BATCH_SIZE + 1)) of $(((${#JOB_COMBINATIONS[@]} + BATCH_SIZE - 1)/BATCH_SIZE))"
    
    # Wait if we've reached max parallel jobs
    wait_for_jobs "$MAX_PARALLEL_JOBS" "SUMMARY_*"
    
    # Submit batch of jobs
    for ((j=i; j<i+BATCH_SIZE && j<${#JOB_COMBINATIONS[@]}; j++)); do
        IFS='|' read -r model dataset dataset_name num_samples sample_id <<< "${JOB_COMBINATIONS[$j]}"
        
        echo "  Submitting job for ${model} + ${dataset_name} + ${num_samples} samples (sample ${sample_id})"
        
        # Submit job (no individual wait here)
        submit_job "$model" "$dataset" "$dataset_name" "$num_samples" "$BATCH_OUTPUT_DIR" "$sample_id"
        
        SUBMITTED_JOBS+=("${model}_${dataset_name}_${num_samples}_s${sample_id}")
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
    
    # Small delay between batches to avoid overwhelming scheduler
    if [[ $((i + BATCH_SIZE)) -lt ${#JOB_COMBINATIONS[@]} ]]; then
        sleep 2
    fi
done


echo "=========================================="
echo "Batch Submission Summary"
echo "=========================================="
echo "Total jobs: $TOTAL_JOBS"
echo "Models: ${#MODELS_TO_USE[@]}"
echo "Datasets: ${#DATASETS_TO_USE[@]}"
echo "Num samples: ${#NUM_SAMPLES_TO_USE[@]}"
echo "Base output directory: $BATCH_OUTPUT_DIR"

if [[ "$DRY_RUN" != "true" ]]; then
    echo ""
    echo "Job submission completed successfully!"
    echo "To monitor jobs, use: squeue -u \$USER --name='SUMMARY_*'"
    echo "To check results, examine the output directory: $BATCH_OUTPUT_DIR"
    echo ""
    echo "Job naming convention: SUMMARY_{MODEL}_{DATASET}_S{SAMPLE_ID}"
    echo "Log files: $BATCH_OUTPUT_DIR/logs/"
else
    echo ""
    echo "This was a dry run. Remove --dry-run to submit actual jobs."
fi

# Clean up temporary directory
if [[ -d "$TEMP_SCRIPTS_DIR" ]]; then
    rm -rf "$TEMP_SCRIPTS_DIR"
fi

# Clean up missing data files if they were created
if [[ "$MISSING_ONLY" == "true" ]]; then
    if [[ -f "$MISSING_FILE" && "$MISSING_FILE" == /tmp/missing_data_* ]]; then
        rm -f "$MISSING_FILE"
    fi
    if [[ -f "$MISSING_COMBINATIONS_FILE" ]]; then
        rm -f "$MISSING_COMBINATIONS_FILE"
    fi
    if [[ -f "/tmp/missing_analysis.log" ]]; then
        rm -f "/tmp/missing_analysis.log"
    fi
fi

echo "=========================================="