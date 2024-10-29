#!/bin/bash


#speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241025_3/checkpoint_epoch_7_step_24000.pt
#speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241025_3/lora_checkpoint_epoch_7_step_24000/
runname=speech_241028_1_noeos
epoch=14
step=50000
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$runname/$checkname
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$runname/$loraname

python run_experiment.py --mode test --modality speech \
    --run_name speech_test_241029_0 --batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 500 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --lora_checkpoint_path $speech_lora_checkpoint_path

runname=text_241024_5
epoch=7
step=24000
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
text_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$runname/$checkname
text_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/$runname/$loraname

python run_experiment.py --mode test --modality text \
    --run_name text_test_241028_0 --batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 500 -1 --validation_interval 2000 \
    --checkpoint_path $text_checkpoint_path \
    --lora_checkpoint_path $text_lora_checkpoint_path

