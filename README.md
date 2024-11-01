# Speech MI 2024 Fall Project

## TODOs
So for classification on annomi, speech helps. 
The next step is to do:
1. exploratory analysis
    - on how speech helps
    - finegrained analysis
        - for example confusion matrix on the categories
        
2. do experiments on forecasting and other datasets.


## Related Work
- [Multimodal Automatic Coding of Client Behavior in Motivational Interviewing](https://dl.acm.org/doi/pdf/10.1145/3382507.3418853) 
- 

## Instructions

### To train forecasting model on AnnoMI Dataset 
`python run_experiment.py --mode train --modality speech --run_name speech_241101_lr1e4_frozen_eos_padding_bug_fixed \
    --batch_size 2  --test_batch_size 4  --datatype float16 --steps 200000 --data_length -1 -1 -1 \
    --validation_interval 2000  --learning_rate 1e-4 --grad_accum_interval 64 --freeze_encoder --use_lora --use_audio_eos \
    --task forecasting --dataset annomi`

### To test forecasting model on AnnoMI Dataset 
`speech_checkpoint_path=your_saved_model_path`

`speech_lora_checkpoint_path=your_saved_lora_path`

`runname=your_run_name`

`python run_experiment.py --mode test --modality speech \
    --run_name test_${runname} --test_batch_size 2 \
    --datatype float16 --steps 50000 --data_length -1 -1 -1 --validation_interval 2000 \
    --checkpoint_path $speech_checkpoint_path \
    --lora_checkpoint_path $speech_lora_checkpoint_path --use_lora  --use_audio_eos \
    --task forecasting --dataset annomi`