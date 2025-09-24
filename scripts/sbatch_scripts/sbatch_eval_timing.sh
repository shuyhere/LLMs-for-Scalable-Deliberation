#!/bin/bash

# Script to submit evaluation jobs with timing measurements
# Supports two tasks:
#   1) evalsum (summary evaluation, based on sbatch_eval.sh)
#   2) regression (regression evaluation, based on sbatch_regression_eval.sh)
#
# Usage examples:
#   # Summary evaluation timing for all combinations
#   ./sbatch_eval_timing.sh --task evalsum
#
#   # Summary evaluation timing for a specific trio
#   ./sbatch_eval_timing.sh --task evalsum --eval-model gpt-4o-mini --summary-model gpt-5-nano --dataset protest
#
#   # Regression evaluation timing for all model-size combinations
#   ./sbatch_eval_timing.sh --task regression
#
#   # Regression evaluation timing for selected models and sizes
#   ./sbatch_eval_timing.sh --task regression --models deepseek-chat,gpt-4o-mini --sizes 10,50

set -euo pipefail

PROJECT_DIR="/ibex/project/c2328/LLMs-Scalable-Deliberation"

# Defaults
TASK="regression"  # options: regression | evalsum

# evalsum defaults
DEFAULT_EVAL_MODEL="all"
DEFAULT_SUMMARY_MODEL="all"
DEFAULT_DATASET="all"
GEN_NUM_SAMPLES=${NUM_SAMPLES:-210}
SUMMARY_BASE_DIR="$PROJECT_DIR/results/summary_model_for_evaluation"

# regression defaults
DEFAULT_EVAL_MODEL_PATH="$PROJECT_DIR/checkpoints/deberta_regression_base_final_v6_pair_split_noactivation"

# Option values
EVAL_MODEL_PATH="$DEFAULT_EVAL_MODEL_PATH"
EVAL_MODEL="$DEFAULT_EVAL_MODEL"
SUMMARY_MODEL="$DEFAULT_SUMMARY_MODEL"
DATASET="$DEFAULT_DATASET"
SELECTED_MODELS=()
SELECTED_SIZES=()
USER_SUMMARY_DIR=""

AVAILABLE_EVALSUM_EVAL_MODELS=("gpt-4o-mini" "gpt-5-nano" "gemini-2.5-flash-lite" "web-rev-claude-3-7-sonnet-20250219")
AVAILABLE_EVALSUM_SUMMARY_MODELS=("gpt-5-nano" "gemini-2.5-flash-lite" "web-rev-claude-3-7-sonnet-20250219" "gpt-4o-mini")
AVAILABLE_EVALSUM_DATASETS=("protest")

AVAILABLE_REGRESSION_MODELS=("gpt-5-mini" "gpt-5" "web-rev-claude-sonnet-4-20250514" "gemini-2.5-flash" "deepseek-reasoner" "grok-4-latest" "TA/openai/gpt-oss-120b" "TA/openai/gpt-oss-20b" "qwen3-0.6b" "qwen3-1.7b" "qwen3-4b" "qwen3-8b" "qwen3-14b" "qwen3-30b-a3b" "qwen3-235b-a22b" "qwen3-32b" "web-rev-claude-opus-4-20250514" "deepseek-chat" "gemini-2.5-pro" "deepseek-reasoner")
AVAILABLE_REGRESSION_SIZES=(10 20 30 50 70 90 120 160 200 240 300)

print_help() {
    echo "Usage: $0 [--task evalsum|regression] [OPTIONS]"
    echo ""
    echo "Tasks and options:"
    echo "  --task evalsum            Summary evaluation timing (default: regression)"
    echo "    --eval-model NAME       Evaluation model (default: $DEFAULT_EVAL_MODEL)"
    echo "    --summary-model NAME    Summary model (default: $DEFAULT_SUMMARY_MODEL)"
    echo "    --dataset NAME          Dataset (default: $DEFAULT_DATASET)"
    echo "    --summary-dir PATH      Base directory of summaries (default: $SUMMARY_BASE_DIR)"
    echo ""
    echo "  --task regression         Regression evaluation timing"
    echo "    --eval-model-path PATH  Path to trained regression model (default: $DEFAULT_EVAL_MODEL_PATH)"
    echo "    --models CSV            Comma-separated list of models"
    echo "    --sizes CSV             Comma-separated list of sample sizes"
    echo ""
    echo "Examples:"
    echo "  $0 --task evalsum --eval-model gpt-4o-mini --summary-model gpt-5-nano --dataset protest"
    echo "  $0 --task regression --models deepseek-chat,gpt-4o-mini --sizes 10,50"
}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"; shift 2;;
        --eval-model)
            EVAL_MODEL="$2"; shift 2;;
        --summary-model)
            SUMMARY_MODEL="$2"; shift 2;;
        --dataset)
            DATASET="$2"; shift 2;;
        --summary-dir)
            USER_SUMMARY_DIR="$2"; shift 2;;
        --eval-model-path)
            EVAL_MODEL_PATH="$2"; shift 2;;
        --models)
            IFS=',' read -ra SELECTED_MODELS <<< "$2"; shift 2;;
        --sizes)
            IFS=',' read -ra SELECTED_SIZES <<< "$2"; shift 2;;
        --help|-h)
            print_help; exit 0;;
        *)
            echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

mkdir -p "$PROJECT_DIR/logs"

if [[ "$TASK" == "evalsum" ]]; then
    LOG_DIR="$PROJECT_DIR/evalsum_logs"
    mkdir -p "$LOG_DIR"
    if [[ -n "$USER_SUMMARY_DIR" ]]; then
        SUMMARY_BASE_DIR="$USER_SUMMARY_DIR"
    fi

    echo "=========================================="
    echo "Summary Evaluation Timing Submission"
    echo "=========================================="
    echo "Evaluation Model: $EVAL_MODEL"
    echo "Summary Model:    $SUMMARY_MODEL"
    echo "Dataset:          $DATASET"
    echo "Summaries Dir:    $SUMMARY_BASE_DIR"
    echo "Log Dir:          $LOG_DIR"
    echo "=========================================="

    submit_evalsum_job() {
        local eval_model="$1"
        local summary_model="$2"
        local dataset="$3"
        local job_name="evalsum_${eval_model}_${summary_model}_${dataset}"
        local temp_script="temp_${job_name}_$$.sh"

        cat > "$temp_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=$LOG_DIR/${job_name}_%j.out
#SBATCH --error=$LOG_DIR/${job_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

cd $PROJECT_DIR

echo "=========================================="
echo "Starting summary evaluation at \\$(date -Is)"
echo "Eval Model: $eval_model"
echo "Summary Model: $summary_model"
echo "Dataset: $dataset"
echo "=========================================="

checkpoint_file="$LOG_DIR/evaluation_checkpoint_${eval_model}_${summary_model}_${dataset}.json"
if [ -f "\\$checkpoint_file" ]; then
    resume_flag="--resume"
else
    resume_flag=""
fi

START_TS=\\$(date +%s)
START_ISO=\\$(date -Is)

(/usr/bin/time -v uv run scripts/batch_evaluate_summaries.py \
    --results-dir "$SUMMARY_BASE_DIR" \
    --output-dir "$SUMMARY_BASE_DIR" \
    --evaluation-model $eval_model \
    --summary-model $summary_model \
    --dataset $dataset \
    --config config/batch_evaluation_config.yaml \
    --checkpoint "\\$checkpoint_file" \
    \\$resume_flag) 2> $LOG_DIR/${job_name}_%j.time

EXIT_CODE=\\$?
END_TS=\\$(date +%s)
END_ISO=\\$(date -Is)
ELAPSED=\\$((END_TS-START_TS))

{
  echo "job_name=${job_name}"
  echo "start_time=\\$START_ISO"
  echo "end_time=\\$END_ISO"
  echo "elapsed_seconds=\\$ELAPSED"
  echo "exit_code=\\$EXIT_CODE"
} > $LOG_DIR/${job_name}_%j.timing_summary.txt

echo "=========================================="
echo "Completed at \\$(date -Is); elapsed=\\$ELAPSED s; exit=\\$EXIT_CODE"
echo "Timing summary: $LOG_DIR/${job_name}_%j.timing_summary.txt"
echo "/usr/bin/time output: $LOG_DIR/${job_name}_%j.time"
echo "=========================================="

exit \\$EXIT_CODE
EOF

        local submit_output
        submit_output=$(sbatch "$temp_script")
        echo "$submit_output"
        rm -f "$temp_script"
    }

    # Validate selections
    if [[ "$EVAL_MODEL" != "all" ]]; then
        if [[ ! " ${AVAILABLE_EVALSUM_EVAL_MODELS[@]} " =~ " ${EVAL_MODEL} " ]]; then
            echo "Error: Invalid eval model: $EVAL_MODEL"; exit 1
        fi
    fi
    if [[ "$SUMMARY_MODEL" != "all" ]]; then
        if [[ ! " ${AVAILABLE_EVALSUM_SUMMARY_MODELS[@]} " =~ " ${SUMMARY_MODEL} " ]]; then
            echo "Error: Invalid summary model: $SUMMARY_MODEL"; exit 1
        fi
    fi
    if [[ "$DATASET" != "all" ]]; then
        if [[ ! " ${AVAILABLE_EVALSUM_DATASETS[@]} " =~ " ${DATASET} " ]]; then
            echo "Error: Invalid dataset: $DATASET"; exit 1
        fi
    fi

    if [[ "$EVAL_MODEL" == "all" && "$SUMMARY_MODEL" == "all" && "$DATASET" == "all" ]]; then
        for ev in "${AVAILABLE_EVALSUM_EVAL_MODELS[@]}"; do
            for sm in "${AVAILABLE_EVALSUM_SUMMARY_MODELS[@]}"; do
                for ds in "${AVAILABLE_EVALSUM_DATASETS[@]}"; do
                    submit_evalsum_job "$ev" "$sm" "$ds"
                    sleep 0.2
                done
            done
        done
    elif [[ "$EVAL_MODEL" == "all" && "$SUMMARY_MODEL" == "all" ]]; then
        for ev in "${AVAILABLE_EVALSUM_EVAL_MODELS[@]}"; do
            for sm in "${AVAILABLE_EVALSUM_SUMMARY_MODELS[@]}"; do
                submit_evalsum_job "$ev" "$sm" "$DATASET"
                sleep 0.2
            done
        done
    elif [[ "$SUMMARY_MODEL" == "all" && "$DATASET" == "all" ]]; then
        for sm in "${AVAILABLE_EVALSUM_SUMMARY_MODELS[@]}"; do
            for ds in "${AVAILABLE_EVALSUM_DATASETS[@]}"; do
                submit_evalsum_job "$EVAL_MODEL" "$sm" "$ds"
                sleep 0.2
            done
        done
    elif [[ "$SUMMARY_MODEL" == "all" ]]; then
        for sm in "${AVAILABLE_EVALSUM_SUMMARY_MODELS[@]}"; do
            submit_evalsum_job "$EVAL_MODEL" "$sm" "$DATASET"
            sleep 0.2
        done
    elif [[ "$DATASET" == "all" ]]; then
        for ds in "${AVAILABLE_EVALSUM_DATASETS[@]}"; do
            submit_evalsum_job "$EVAL_MODEL" "$SUMMARY_MODEL" "$ds"
            sleep 0.2
        done
    else
        submit_evalsum_job "$EVAL_MODEL" "$SUMMARY_MODEL" "$DATASET"
    fi

elif [[ "$TASK" == "regression" ]]; then
    LOG_DIR="$PROJECT_DIR/logs/regression_evaluation"
    mkdir -p "$LOG_DIR"
    BASE_OUTPUT_DIR="$PROJECT_DIR/results/regression_evaluation"
    echo "=========================================="
    echo "Regression Evaluation Timing Submission"
    echo "=========================================="
    echo "Eval Model Path: $EVAL_MODEL_PATH"
    echo "Output base dir: $BASE_OUTPUT_DIR"
    echo "Log dir:         $LOG_DIR"
    echo "=========================================="

    if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
        SELECTED_MODELS=("${AVAILABLE_REGRESSION_MODELS[@]}")
    fi
    if [ ${#SELECTED_SIZES[@]} -eq 0 ]; then
        SELECTED_SIZES=("${AVAILABLE_REGRESSION_SIZES[@]}")
    fi

    if [[ ! -d "$EVAL_MODEL_PATH" ]]; then
        echo "Error: Evaluation model path does not exist: $EVAL_MODEL_PATH"; exit 1
    fi

    submit_regression_job() {
        local model_name="$1"
        local sample_size="$2"
        local output_dir="$BASE_OUTPUT_DIR/${model_name}/${sample_size}"
        mkdir -p "$output_dir"
        local job_name="eval_${model_name}_${sample_size}"
        local temp_script="temp_${job_name}_$$.sh"

        cat > "$temp_script" << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=$LOG_DIR/${job_name}_%j.out
#SBATCH --error=$LOG_DIR/${job_name}_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --account=conf-icl-2025.09.24-wangd0d

cd $PROJECT_DIR

echo "=========================================="
echo "Starting regression evaluation at \\$(date -Is)"
echo "Model: $model_name"
echo "Sample Size: $sample_size"
echo "Eval Model: $EVAL_MODEL_PATH"
echo "=========================================="

source .venv/bin/activate

START_TS=\\$(date +%s)
START_ISO=\\$(date -Is)

(/usr/bin/time -v python scripts/batch_regression_evaluate.py \
    --model-path "$EVAL_MODEL_PATH" \
    --summary-dir "$PROJECT_DIR/results/summary_model_for_evaluation" \
    --comments-dir "$PROJECT_DIR/datasets/annotation_V0_V1_dataset" \
    --model-names "$model_name" \
    --sample-sizes $sample_size \
    --n-summaries 3 \
    --output "$output_dir/evaluation_results.json" \
    --device cuda) 2> $LOG_DIR/${job_name}_%j.time

EXIT_CODE=\\$?
END_TS=\\$(date +%s)
END_ISO=\\$(date -Is)
ELAPSED=\\$((END_TS-START_TS))

{
  echo "job_name=${job_name}"
  echo "model_name=$model_name"
  echo "sample_size=$sample_size"
  echo "start_time=\\$START_ISO"
  echo "end_time=\\$END_ISO"
  echo "elapsed_seconds=\\$ELAPSED"
  echo "exit_code=\\$EXIT_CODE"
} > $LOG_DIR/${job_name}_%j.timing_summary.txt

echo "=========================================="
echo "Completed at \\$(date -Is); elapsed=\\$ELAPSED s; exit=\\$EXIT_CODE"
echo "Timing summary: $LOG_DIR/${job_name}_%j.timing_summary.txt"
echo "/usr/bin/time output: $LOG_DIR/${job_name}_%j.time"
echo "=========================================="

exit \\$EXIT_CODE
EOF

        local submit_output
        submit_output=$(sbatch "$temp_script")
        echo "$submit_output"
        rm -f "$temp_script"
    }

    for m in "${SELECTED_MODELS[@]}"; do
        for s in "${SELECTED_SIZES[@]}"; do
            submit_regression_job "$m" "$s"
            sleep 0.2
        done
    done

else
    echo "Error: Unknown task '$TASK'"; print_help; exit 1
fi
