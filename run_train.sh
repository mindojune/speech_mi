#!/bin/bash

# bsz 4 don't work 2 ok


##################################################################
# annomi classification speech [trained] speech_241031_lr1e4_frozen_eos_padding_bug_fixed
python run_experiment.py --mode train --modality speech --run_name speech_241108_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --only_hq_sessions

# annomi classification text [trained]
python run_experiment.py --mode train --modality text --run_name text_241108_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --only_hq_sessions

# annomi classification textAA []
python run_experiment.py --mode train --modality textAA --run_name textAA_250113_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --only_hq_sessions

##################################################################
# annomi forecasting speech [trained]
python run_experiment.py --mode train --modality speech --run_name speech_241108_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --task forecasting --only_hq_sessions

# annomi forecasting text [trained]
python run_experiment.py --mode train --modality text --run_name text_241108_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --task forecasting --only_hq_sessions

# annomi forecasting textAA []
python run_experiment.py --mode train --modality textAA --run_name textAA_250113_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --task forecasting --only_hq_sessions

##################################################################
# annomi response_generation speech [training] 
python run_experiment.py --mode train --modality speech --run_name speech_241121_only_hq_omit_last \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --task response_generation --only_hq_sessions --omit_last_text

# annomi response_generation text [training]
python run_experiment.py --mode train --modality text --run_name text_241108_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --task response_generation --only_hq_sessions #--omit_text


##################################################################
##################################################################
##################################################################
######################
python run_experiment.py --mode train --modality speech --run_name speech_241030_lr5e5_encoderonly \
    --batch_size 1  --test_batch_size 1  --datatype float16 --steps 50000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 5e-5 --grad_accum_interval 64

python run_experiment.py --mode train --modality text --run_name text_241028_0 \
    --batch_size 2     --datatype float16 --steps 50000 --data_length -1 500 8 \
    --validation_interval 2000  --learning_rate 1e-4


##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################



epoch=125 #9
step=200000 #34000 #40000 #50000
runname=textAA_epoch_${epoch}_step_${step}_classification
savename=textAA_250113_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${loraname}

# annomi classification textAA []
python run_experiment.py --mode train --modality textAA --run_name cont_textAA_250113_only_hq \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --only_hq_sessions --lora_checkpoint_path $speech_lora_checkpoint_path  


    