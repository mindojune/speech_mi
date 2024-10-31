#!/bin/bash


#speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241025_3/checkpoint_epoch_7_step_24000.pt
#speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241025_3/lora_checkpoint_epoch_7_step_24000/
savename=speech_241030_lr1e4_frozen
epoch=5
step=38000
runname=cont_speech_epoch_${epoch}_step_${step}
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$savename/$checkname
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$savename/$loraname

python run_experiment.py --mode train --modality speech \
    --run_name $runname --test_batch_size 1 --batch_size 1 \
    --datatype float16 --steps 100000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --lora_checkpoint_path $speech_lora_checkpoint_path \
    --learning_rate 1e-4 --freeze_encoder --use_lora


#####
savename=speech_241030_lr5e5_encoderonly
epoch=7
step=50000
runname=cont_speech_epoch_${epoch}_step_${step}
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$savename/$checkname
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$savename/$loraname

python run_experiment.py --mode train --modality speech \
    --run_name $runname --test_batch_size 1 --batch_size 1 \
    --datatype float16 --steps 100000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --learning_rate 1e-4 