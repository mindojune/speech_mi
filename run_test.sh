#!/bin/bash
# 2 / 18000

##################################################################
# annomi classification speech [trained] speech_241031_lr1e4_frozen_eos_padding_bug_fixed
epoch=9
step=34000 #40000 #50000
runname=speech_epoch_${epoch}_step_${step}_classification
savename=speech_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality speech \
    --run_name test_${runname} --test_batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora  --use_audio_eos --only_hq_sessions

##################################################################
# annomi classification text [trained]
epoch=11 #9
step=38000 #34000 #40000 #50000
runname=text_epoch_${epoch}_step_${step}_classification
savename=text_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality text \
    --run_name test_${runname} --test_batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora  --use_audio_eos --only_hq_sessions


##################################################################
# annomi forecasting speech [training]
epoch=5
step=20000 #40000 #50000
runname=speech_epoch_${epoch}_step_${step}_forecasting
savename=speech_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/forecasting_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/forecasting_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality speech \
    --run_name test_${runname} --test_batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora  --use_audio_eos --task forecasting --only_hq_sessions

##################################################################
# annomi forecasting text [training]
#################################
epoch=5
step=18000 #40000 #50000
runname=text_epoch_${epoch}_step_${step}_forecasting
savename=text_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/forecasting_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/forecasting_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality text \
    --run_name test_${runname} --test_batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora  --use_audio_eos --task forecasting --only_hq_sessions


##################################################################
# annomi response_generation speech [training]
epoch=16
step=56000 #40000 #50000
runname=speech_epoch_${epoch}_step_${step}_response_generation
savename=speech_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/response_generation_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/response_generation_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality speech \
    --run_name test_${runname} --test_batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora  --use_audio_eos --task response_generation \
    --max_new_tokens 40 --only_hq_sessions

##################################################################
# annomi response_generation text [training]    
epoch=11
step=40000 #40000 #50000
runname=text_epoch_${epoch}_step_${step}_response_generation
savename=text_241108_only_hq #cont_speech_epoch_5_step_38000 #speech_241030_lr5e5_encoderonly #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/response_generation_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/response_generation_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality text \
    --run_name test_${runname} --test_batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora  --use_audio_eos --task response_generation \
    --max_new_tokens 40 --only_hq_sessions

##################################################################
##################################################################
##################################################################
epoch=14
step=100000
runname=speech_epoch_${epoch}_step_${step}
savename=cont_speech_epoch_7_step_50000 #speech_test_241029_leftpadded # speech_241030_lr1e4_frozen
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality speech \
    --run_name test_${runname} --test_batch_size 1 \
    --datatype float16 --steps 50000 --data_length -1 500 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path 

#speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241025_3/checkpoint_epoch_7_step_24000.pt
#speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241025_3/lora_checkpoint_epoch_7_step_24000/
# runname=speech_test_241029_freeze_encoder_leftpadded

epoch=3 #14
step=26000 #50000
runname=speech_epoch_${epoch}_step_${step}
savename=speech_241030_lr1e4 #speech_test_241029_leftpadded
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
speech_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${checkname}
speech_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality speech \
    --run_name test_${runname} --test_batch_size 1 \
    --datatype float16 --steps 50000 --data_length -1 500 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --lora_checkpoint_path $speech_lora_checkpoint_path


epoch=9 #7 #
step=34000 #24000
runname=text_epoch_${epoch}_step_${step}
savename=text_241024_5
checkname=checkpoint_epoch_${epoch}_step_${step}.pt
loraname=lora_checkpoint_epoch_${epoch}_step_${step}
text_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${checkname}
text_lora_checkpoint_path=/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/${savename}/${loraname}

python run_experiment.py --mode test --modality text \
    --run_name test_${runname} --batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 500 -1 --validation_interval 2000 \
    --checkpoint_path $text_checkpoint_path \
    --lora_checkpoint_path $text_lora_checkpoint_path

