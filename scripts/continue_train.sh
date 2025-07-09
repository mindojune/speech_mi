#!/bin/bash

# Continue Training from Checkpoint
# This script shows how to resume training from a saved checkpoint
# Useful for extending training or fine-tuning with different hyperparameters

# Set your experiment paths
SAVE_DIR="./logs"
EXPERIMENT_NAME="speech_mi_experiment"

# Update these paths to point to your existing checkpoints
# Format: ./logs/task_experiment/run_name/checkpoint_epoch_X_step_Y.pt
CHECKPOINT_PATH="./logs/classification_experiment/${EXPERIMENT_NAME}_speech_classification/checkpoint_epoch_5_step_38000.pt"
LORA_CHECKPOINT_PATH="./logs/classification_experiment/${EXPERIMENT_NAME}_speech_classification/lora_checkpoint_epoch_5_step_38000"

echo "Continuing training from checkpoint..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "LoRA checkpoint: $LORA_CHECKPOINT_PATH"
echo "Make sure to update checkpoint paths in this script before running!"

##################################################################
# CONTINUE SPEECH TRAINING
##################################################################

echo "\n=== CONTINUING SPEECH TRAINING ==="

# Continue training speech model from checkpoint
python core/run_experiment.py --mode train --modality speech \
    --run_name "${EXPERIMENT_NAME}_speech_continued" \
    --save_dir "$SAVE_DIR" \
    --batch_size 2 --test_batch_size 2 --datatype float16 \
    --steps 100000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --lora_checkpoint_path "$LORA_CHECKPOINT_PATH" \
    --learning_rate 1e-4 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos --only_hq_sessions

##################################################################
# CONTINUE WITH DIFFERENT HYPERPARAMETERS
##################################################################

echo "\n=== CONTINUING WITH DIFFERENT LEARNING RATE ==="

# Example: Continue with different learning rate
python core/run_experiment.py --mode train --modality speech \
    --run_name "${EXPERIMENT_NAME}_speech_continued_lr5e5" \
    --save_dir "$SAVE_DIR" \
    --batch_size 1 --test_batch_size 1 --datatype float16 \
    --steps 100000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --learning_rate 5e-5 --grad_accum_interval 64 \
    --freeze_encoder --use_lora --use_audio_eos --only_hq_sessions

##################################################################
# CONTINUE WITHOUT LORA (FULL MODEL TRAINING)
##################################################################

echo "\n=== CONTINUING WITHOUT LORA ==="

# Example: Continue training without LoRA (full model fine-tuning)
# Note: This will require more GPU memory
python core/run_experiment.py --mode train --modality speech \
    --run_name "${EXPERIMENT_NAME}_speech_continued_full" \
    --save_dir "$SAVE_DIR" \
    --batch_size 1 --test_batch_size 1 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --learning_rate 1e-4 --grad_accum_interval 64 \
    --only_hq_sessions

echo "\nContinue training script completed!"
echo "Check logs in: $SAVE_DIR"
echo "New models will be saved with the updated run names"