#!/bin/bash

# bsz 4 don't work 2 ok

python run_experiment.py --mode train --modality speech --run_name speech_241030_lr1e4_frozen \
    --batch_size 1  --test_batch_size 1  --datatype float16 --steps 50000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder

python run_experiment.py --mode train --modality speech --run_name speech_241030_lr5e5_encoderonly \
    --batch_size 1  --test_batch_size 1  --datatype float16 --steps 50000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 5e-5 --grad_accum_interval 64 --use_lora False

python run_experiment.py --mode train --modality text --run_name text_241028_0 \
    --batch_size 2     --datatype float16 --steps 50000 --data_length -1 500 8 \
    --validation_interval 2000  --learning_rate 1e-4


