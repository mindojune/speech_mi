#!/bin/bash

# Noise Robustness Testing for Speech-Integrated Modeling
# This script tests model performance under different noise conditions (SNR levels)
# as described in the SIGDIAL 2025 paper.

# Set your experiment paths
SAVE_DIR="./logs"
EXPERIMENT_NAME="speech_mi_experiment"

# Update these paths to point to your trained models
CLASSIFICATION_CHECKPOINT="./logs/classification_experiment/${EXPERIMENT_NAME}_speech_classification/checkpoint_epoch_100_step_160000.pt"
CLASSIFICATION_LORA="./logs/classification_experiment/${EXPERIMENT_NAME}_speech_classification/lora_checkpoint_epoch_100_step_160000"

FORECASTING_CHECKPOINT="./logs/forecasting_experiment/${EXPERIMENT_NAME}_speech_forecasting/checkpoint_epoch_4_step_8000.pt"
FORECASTING_LORA="./logs/forecasting_experiment/${EXPERIMENT_NAME}_speech_forecasting/lora_checkpoint_epoch_4_step_8000"

echo "Starting noise robustness experiments..."
echo "Testing different SNR levels: 25, 20, 15, 10, 5, 0 dB"
echo "Make sure to update checkpoint paths in this script before running!"

##################################################################
# CLASSIFICATION NOISE ROBUSTNESS
##################################################################

echo "\n=== CLASSIFICATION NOISE ROBUSTNESS ==="

# Test classification performance at different noise levels
for noise_level in 25 20 15 10 5 0
do
    echo "Testing classification at SNR: ${noise_level} dB"
    python core/run_experiment.py --mode test --modality speech \
        --run_name "test_${EXPERIMENT_NAME}_speech_classification_noise${noise_level}" \
        --save_dir "$SAVE_DIR" \
        --test_batch_size 2 --datatype float16 \
        --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
        --checkpoint_path "$CLASSIFICATION_CHECKPOINT" \
        --lora_checkpoint_path "$CLASSIFICATION_LORA" \
        --use_lora --use_audio_eos --only_hq_sessions \
        --noise_level $noise_level
done

##################################################################
# FORECASTING NOISE ROBUSTNESS
##################################################################

echo "\n=== FORECASTING NOISE ROBUSTNESS ==="

# Test forecasting performance at different noise levels
for noise_level in 25 20 15 10 5 0
do
    echo "Testing forecasting at SNR: ${noise_level} dB"
    python core/run_experiment.py --mode test --modality speech \
        --run_name "test_${EXPERIMENT_NAME}_speech_forecasting_noise${noise_level}" \
        --save_dir "$SAVE_DIR" \
        --test_batch_size 2 --datatype float16 \
        --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
        --checkpoint_path "$FORECASTING_CHECKPOINT" \
        --lora_checkpoint_path "$FORECASTING_LORA" \
        --use_lora --use_audio_eos --task forecasting --only_hq_sessions \
        --noise_level $noise_level
done

##################################################################
# ADDITIONAL NOISE EXPERIMENTS
##################################################################

echo "\n=== ADDITIONAL NOISE EXPERIMENTS ==="

# Test with complete noise replacement (--noise_level -666)
echo "Testing with complete noise replacement..."
python core/run_experiment.py --mode test --modality speech \
    --run_name "test_${EXPERIMENT_NAME}_speech_classification_silence" \
    --save_dir "$SAVE_DIR" \
    --test_batch_size 2 --datatype float16 \
    --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path "$CLASSIFICATION_CHECKPOINT" \
    --lora_checkpoint_path "$CLASSIFICATION_LORA" \
    --use_lora --use_audio_eos --only_hq_sessions \
    --noise_level -666

echo "\nNoise robustness testing completed!"
echo "Check results in: $SAVE_DIR"
echo "Use plot_noise.py to visualize the results:"
echo "python plot_noise.py --task classification --results_dir $SAVE_DIR"
echo "python plot_noise.py --task forecasting --results_dir $SAVE_DIR"