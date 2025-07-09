#!/bin/bash

# Testing Scripts for Speech-Integrated Modeling for Behavioral Coding in Counseling
# This script provides examples of testing commands for different modalities and tasks
# described in the SIGDIAL 2025 paper.

# Set your experiment paths
SAVE_DIR="./logs"  # Change this to your log directory
EXPERIMENT_NAME="speech_mi_experiment"

# You need to update these paths to point to your trained models
# Example format: ./logs/classification_experiment/your_model_name/checkpoint_epoch_X_step_Y.pt
SPEECH_CHECKPOINT_PATH="./logs/classification_experiment/${EXPERIMENT_NAME}_speech_classification/checkpoint_epoch_100_step_160000.pt"
SPEECH_LORA_PATH="./logs/classification_experiment/${EXPERIMENT_NAME}_speech_classification/lora_checkpoint_epoch_100_step_160000"
TEXT_LORA_PATH="./logs/classification_experiment/${EXPERIMENT_NAME}_text_classification/lora_checkpoint_epoch_100_step_160000"
TEXTAA_LORA_PATH="./logs/classification_experiment/${EXPERIMENT_NAME}_textAA_classification/lora_checkpoint_epoch_100_step_160000"

echo "Starting testing experiments..."
echo "Save directory: $SAVE_DIR"
echo "Make sure to update checkpoint paths in this script before running!"

##################################################################
# CLASSIFICATION TESTING
##################################################################

echo "\n=== CLASSIFICATION TASK TESTING ==="

# 1. Speech-based Classification Testing
echo "Testing speech-based classification model..."
python core/run_experiment.py --mode test --modality speech \
    --run_name "test_${EXPERIMENT_NAME}_speech_classification" \
    --save_dir "$SAVE_DIR" \
    --test_batch_size 2 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path "$SPEECH_CHECKPOINT_PATH" \
    --lora_checkpoint_path "$SPEECH_LORA_PATH" \
    --use_lora --use_audio_eos --only_hq_sessions

# 2. Text-only Classification Testing (baseline)
echo "Testing text-only classification model..."
python core/run_experiment.py --mode test --modality text \
    --run_name "test_${EXPERIMENT_NAME}_text_classification" \
    --save_dir "$SAVE_DIR" \
    --test_batch_size 2 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --lora_checkpoint_path "$TEXT_LORA_PATH" \
    --use_lora --use_audio_eos --only_hq_sessions

# 3. Text + Audio Analysis Classification Testing
echo "Testing text + audio analysis classification model..."
python core/run_experiment.py --mode test --modality textAA \
    --run_name "test_${EXPERIMENT_NAME}_textAA_classification" \
    --save_dir "$SAVE_DIR" \
    --test_batch_size 2 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --lora_checkpoint_path "$TEXTAA_LORA_PATH" \
    --use_lora --use_audio_eos --only_hq_sessions

##################################################################
# FORECASTING TESTING
##################################################################

echo "\n=== FORECASTING TASK TESTING ==="

# Update these paths to point to your forecasting models
SPEECH_FORECAST_CHECKPOINT="./logs/forecasting_experiment/${EXPERIMENT_NAME}_speech_forecasting/checkpoint_epoch_4_step_8000.pt"
SPEECH_FORECAST_LORA="./logs/forecasting_experiment/${EXPERIMENT_NAME}_speech_forecasting/lora_checkpoint_epoch_4_step_8000"
TEXT_FORECAST_LORA="./logs/forecasting_experiment/${EXPERIMENT_NAME}_text_forecasting/lora_checkpoint_epoch_4_step_8000"
TEXTAA_FORECAST_LORA="./logs/forecasting_experiment/${EXPERIMENT_NAME}_textAA_forecasting/lora_checkpoint_epoch_18_step_30000"

# 1. Speech-based Forecasting Testing
echo "Testing speech-based forecasting model..."
python core/run_experiment.py --mode test --modality speech \
    --run_name "test_${EXPERIMENT_NAME}_speech_forecasting" \
    --save_dir "$SAVE_DIR" \
    --test_batch_size 2 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path "$SPEECH_FORECAST_CHECKPOINT" \
    --lora_checkpoint_path "$SPEECH_FORECAST_LORA" \
    --use_lora --use_audio_eos --task forecasting --only_hq_sessions

# 2. Text-only Forecasting Testing
echo "Testing text-only forecasting model..."
python core/run_experiment.py --mode test --modality text \
    --run_name "test_${EXPERIMENT_NAME}_text_forecasting" \
    --save_dir "$SAVE_DIR" \
    --test_batch_size 2 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --lora_checkpoint_path "$TEXT_FORECAST_LORA" \
    --use_lora --use_audio_eos --task forecasting --only_hq_sessions

# 3. Text + Audio Analysis Forecasting Testing
echo "Testing text + audio analysis forecasting model..."
python core/run_experiment.py --mode test --modality textAA \
    --run_name "test_${EXPERIMENT_NAME}_textAA_forecasting" \
    --save_dir "$SAVE_DIR" \
    --test_batch_size 2 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --lora_checkpoint_path "$TEXTAA_FORECAST_LORA" \
    --use_lora --use_audio_eos --task forecasting --only_hq_sessions

echo "\nTesting script completed!"
echo "Check results in: $SAVE_DIR"
echo "Results are saved as test_results.json in each experiment directory"