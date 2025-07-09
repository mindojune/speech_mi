#!/bin/bash

# Training Scripts for Speech-Integrated Modeling for Behavioral Coding in Counseling
# This script provides examples of training commands for different modalities and tasks
# described in the SIGDIAL 2025 paper.

# Set your experiment name and save directory
EXPERIMENT_NAME="speech_mi_experiment"
SAVE_DIR="./logs"  # Change this to your desired log directory

# Configuration notes:
# - batch_size 2 works well for most setups (batch_size 4 may cause memory issues)
# - Use --only_hq_sessions to train only on high-quality MI sessions
# - --freeze_encoder keeps the audio encoder frozen during training
# - --use_lora enables LoRA fine-tuning for efficiency
# - --use_audio_eos adds end-of-sequence tokens for audio

echo "Starting training experiments..."
echo "Save directory: $SAVE_DIR"
echo "Experiment name: $EXPERIMENT_NAME"

##################################################################
# CLASSIFICATION EXPERIMENTS
##################################################################

echo "\n=== CLASSIFICATION TASK ==="

# 1. Speech-based Classification
echo "Training speech-based classification model..."
python core/run_experiment.py --mode train --modality speech \
    --run_name "${EXPERIMENT_NAME}_speech_classification" \
    --save_dir "$SAVE_DIR" \
    --batch_size 2 --test_batch_size 4 --datatype float16 \
    --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000 --learning_rate 1e-4 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos --only_hq_sessions

# 2. Text-only Classification (baseline)
echo "Training text-only classification model..."
python core/run_experiment.py --mode train --modality text \
    --run_name "${EXPERIMENT_NAME}_text_classification" \
    --save_dir "$SAVE_DIR" \
    --batch_size 2 --test_batch_size 4 --datatype float16 \
    --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000 --learning_rate 1e-4 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos --only_hq_sessions

# 3. Text + Audio Analysis Classification
echo "Training text + audio analysis classification model..."
python core/run_experiment.py --mode train --modality textAA \
    --run_name "${EXPERIMENT_NAME}_textAA_classification" \
    --save_dir "$SAVE_DIR" \
    --batch_size 2 --test_batch_size 4 --datatype float16 \
    --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000 --learning_rate 1e-4 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos --only_hq_sessions

##################################################################
# FORECASTING EXPERIMENTS
##################################################################

echo "\n=== FORECASTING TASK ==="

# 1. Speech-based Forecasting
echo "Training speech-based forecasting model..."
python core/run_experiment.py --mode train --modality speech \
    --run_name "${EXPERIMENT_NAME}_speech_forecasting" \
    --save_dir "$SAVE_DIR" \
    --batch_size 2 --test_batch_size 4 --datatype float16 \
    --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000 --learning_rate 1e-4 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos \
    --task forecasting --only_hq_sessions

# 2. Text-only Forecasting (baseline)
echo "Training text-only forecasting model..."
python core/run_experiment.py --mode train --modality text \
    --run_name "${EXPERIMENT_NAME}_text_forecasting" \
    --save_dir "$SAVE_DIR" \
    --batch_size 2 --test_batch_size 4 --datatype float16 \
    --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000 --learning_rate 1e-4 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos \
    --task forecasting --only_hq_sessions

# 3. Text + Audio Analysis Forecasting
echo "Training text + audio analysis forecasting model..."
python core/run_experiment.py --mode train --modality textAA \
    --run_name "${EXPERIMENT_NAME}_textAA_forecasting" \
    --save_dir "$SAVE_DIR" \
    --batch_size 2 --test_batch_size 4 --datatype float16 \
    --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000 --learning_rate 1e-4 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos \
    --task forecasting --only_hq_sessions

##################################################################
# CHECKPOINT CONTINUATION EXAMPLE
##################################################################

# Example of how to continue training from a checkpoint
# Uncomment and modify the paths below to continue training

# CHECKPOINT_PATH="./logs/classification_experiment/your_model/checkpoint_epoch_X_step_Y.pt"
# LORA_PATH="./logs/classification_experiment/your_model/lora_checkpoint_epoch_X_step_Y"
# 
# echo "Continuing training from checkpoint..."
# python core/run_experiment.py --mode train --modality textAA \
#     --run_name "${EXPERIMENT_NAME}_textAA_continued" \
#     --save_dir "$SAVE_DIR" \
#     --batch_size 2 --test_batch_size 4 --datatype float16 \
#     --steps 200000 --data_length -1 -1 -1 \
#     --validation_interval 2000 --learning_rate 1e-4 --grad_accum_interval 64 \
#     --freeze_encoder --use_lora --use_audio_eos \
#     --only_hq_sessions --lora_checkpoint_path "$LORA_PATH"

echo "\nTraining script completed!"
echo "Check logs in: $SAVE_DIR"
echo "Use run_test.sh to evaluate trained models"