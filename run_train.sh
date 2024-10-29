#!/bin/bash

python run_experiment.py --mode train --modality speech --run_name speech_241028_0 \
    --batch_size 2     --datatype float16 --steps 50000 --data_length -1 500 8 \
    --validation_interval 2000  --learning_rate 1e-4 --freeze_encoder

python run_experiment.py --mode train --modality text --run_name text_241028_0 \
    --batch_size 2     --datatype float16 --steps 50000 --data_length -1 500 8 \
    --validation_interval 2000  --learning_rate 1e-4


