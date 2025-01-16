#!/bin/bash
# 2 / 18000

##################################################################

epoch=100
step=160000 #40000 #50000
runname=speech_epoch_${epoch}_step_${step}_classification
savename=speech_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${loraname}

for noise_level in 25 20 15 10 5 0
do
    echo "Running Classification noise level: $noise_level"
    python run_experiment.py --mode test --modality speech \
        --run_name test_${runname}_noise${noise_level} --test_batch_size 2 \
        --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
        --checkpoint_path $speech_checkpoint_path \
        --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora --use_audio_eos --only_hq_sessions --noise_level $noise_level
done

##################################################################


epoch=4
step=8000 #40000 #50000
runname=speech_epoch_${epoch}_step_${step}_forecasting
savename=speech_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/forecasting_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/forecasting_experiment/${savename}/${loraname}

for noise_level in 25 20 15 10 5 0
do
    echo "Running Forecasting noise level: $noise_level"
    python run_experiment.py --mode test --modality speech \
        --run_name test_${runname}_noise${noise_level} --test_batch_size 2 \
        --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
        --checkpoint_path $speech_checkpoint_path \
        --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora --use_audio_eos --task forecasting --only_hq_sessions --noise_level $noise_level
done

