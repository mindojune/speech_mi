# Speech MI 2024 Fall Project


## Note: I've been training testing with = low quality sessions included....
so this may need to change

## Larnell TODOs


### Possible Dataset: ``MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation``
 
- [MELD](https://github.com/declare-lab/MELD)

Try if we can use this dataset for the project.

### Before
Please understand that there are hardcoded paths in the code. You will need to change them to your own paths.
Also __never remove or move any files or scripts in my paths__, which
are 
- `/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/`
- `/nfs/turbo/coe-mihalcea/dojmin/speech_mi`

### Steps

0. Go over the scripts briefly and get a high-level understanding of what's going on. 
`run_experiment.py` is the main script that runs the experiments so it should be your entry point.
__Your goal__ is to be able to __describe what each script and each function does at a high level__.

1. Get a GPU allocation (inside a screen or tmux session) with something like


`srun --account mihalcea_owned1 --partition spgpu2 --cpus-per-task 16 --mem 120GB --gpus 1 \--time 24:00:00 --pty bash`

2. Create a dedicated and separate conda environment. Install the required packages.

`conda create --name speech_mi python=3.11`

`conda activate speech_mi`

`pip install -r requirements.txt`


3. Accessing tensorboard
`ssh -L localhost:16060:localhost:6060 gl`

Then inside a separate tmux / screen session

`tensorboard --logdir=${log_path} --port=6060`

Try with `/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/classification_experiment/`. This contains MY logs for the classification experiment.


4. Try running the training commands below. Make sure that you are providing the correct arguments. Pay mind to the hardcoded paths. For the audios, you can just used the provided paths. 
After you train the model you can run inference with your own checkpoints.

## Instructions

### Install Required Packages
`pip install -r requirements.txt`

If you have a problem installing, install the packages without the specific versions (remove the `==` and the version number) and try again.



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

#################
## For Myself
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
