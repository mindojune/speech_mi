#!/bin/bash

python run_experiment.py --mode test --modality text \
    --run_name text_test_24104_0 --batch_size 2 \
    --datatype float32 --steps 50000 --data_length -1 500 8 --validation_interval 2000 \
    --checkpoint_path /scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/text_241024_0/checkpoint_epoch_0_step_2000.pt \
    --lora_checkpoint_path /scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/text_241024_0/lora_checkpoint_epoch_0_step_2000/


python run_experiment.py --mode test --modality speech \
    --run_name speech_test_241025_0 --batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 500 8 --validation_interval 2000 \
    --checkpoint_path /scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241024_2/checkpoint_epoch_3_step_12000.pt \
    --lora_checkpoint_path /scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/speech_241024_2/lora_checkpoint_epoch_3_step_12000/