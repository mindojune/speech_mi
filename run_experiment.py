import argparse
import os
import logging
from tqdm.auto import tqdm


from transformers import AutoProcessor, AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from omegaconf import OmegaConf

import random, audiofile, audresample
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from process_data import process # TODO
from log_writer import LogWriter 
from audio_encoder import AudioEncoder
from utils import set_all_seeds, create_attention_mask, compute_num_audio_embeds

class MyTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = OmegaConf.load(args.config)
        self.logwriter = LogWriter(self.config, os.path.join('experiment', args.run_name))

        # Setup Model
        self.model = AutoModelForCausalLM.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.audio_encoder = AudioEncoder(self.config)
        self.model.to(self.device)
        self.feature_extractor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")


        # Setup Training Parameters
        self.step = 0
        self.val_step = 0
        self.start_epoch = 0
        self.grad_accum_interval = args.grad_accum_interval
        self.num_epochs = self.config.train.epochs
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.audio_encoder.parameters()},
                {'params': self.model.parameters()}
            ],
            lr=self.config.train.optimizer.lr,
            betas=(self.config.train.optimizer.beta1, self.config.train.optimizer.beta2),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=(self.num_epochs * len(self.data['train']) // self.grad_accum_interval),
            power=1
        )
        if args.checkpoint_path:
            self.load_checkpoint(args.checkpoint_path)

    def setup_logging(self):
        log_dir = os.path.join('experiment', self.args.run_name)
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'experiment.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Logging setup complete.')


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio_encoder.load_state_dict(checkpoint["audio_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        if self.device == torch.device(f"cuda:{self.args.gpu_idx}"):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.args.gpu_idx)

        print(f"Loaded checkpoint from {checkpoint_path}.\n")

    def prepare_batch(self, batch):
        # {'inputs': ["client: Dr. Morrow, I'm so relieved I'm not pregnant.\ntherapist: Huh, you feel like you dodged the bullet."], 
        # 'labels': ['neutral'], 
        # 'audio_infos': [{'audio_path': './data/session_audios/high_129.wav', 'begin': '00:00:39', 'end': '00:00:41'}]}

        # first encode the text into embeddings
        # then embedd the audio using the audio encoder
        #   the audio is spliced from the audio path, begin, and end
        # concatenate the two types of embeddings, along the sequence dimension
        # TODO: fix the following
        encoding = self.tokenizer(batch["inputs"], return_tensors="pt", padding="longest", truncation=True, max_length=512)

        if self.args.mode == "speech":
            audio_paths = [x["audio_path"] for x in batch]
            
            audio_features = []
            for idx, fname in enumerate(audio_paths):

                signal, source_rate = audiofile.read(fname)
                signal = audresample.resample(signal, source_rate, 16000)
                audio = signal.squeeze()
                if self.args.max_audio_s:
                    audio = audio[:self.args.max_audio_s*16000]  
                # turn 'begin': '00:00:39', 'end': '00:00:41' into seconds
                begin_time = sum(x * int(t) for x, t in zip([3600, 60, 1], fname["begin"].split(":")))
                end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], fname["end"].split(":")))
                audio = audio[begin_time * 16000:end_time * 16000]
                audio_feature = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding="longest", return_attention_mask=True )
                audio_features += [audio_feature]

            feat_lens = ([ len(x.input_values[0]) for x in audio_features])
            features = pad_sequence([torch.Tensor(x.input_values[0]) for x in audio_features], batch_first=True)

            audio_attention_mask = create_attention_mask(feat_lens)

            encoding["input_features"] = features 
            encoding["audio_attention_mask"] = audio_attention_mask 
    
        # TODO: need to think about the padding / truncation side

        inputs = encoding.to(self.device)
        return inputs      


        return 

    def run_experiment(self):
        self.setup_logging()
        # self.set_seed()
        set_all_seeds(self.args.seed)

        logging.info(f"Running experiment with the following parameters:")
        logging.info(f"Learning Rate: {self.args.learning_rate}")
        logging.info(f"Batch Size: {self.args.batch_size}")
        logging.info(f"Epochs: {self.args.epochs}")
        logging.info(f"Model: {self.args.model}")
        logging.info(f"Dataset: {self.args.dataset}")
        logging.info(f"Task: {self.args.task}")
        logging.info(f"Mode: {self.args.mode}")

        self.dataloaders = process(self.args)
        self.trainloader = self.data["train"]
        self.valloader = self.data["val"]
        self.testloader = self.data["test"]

        if 'train' in self.args.mode:
            self.train()

        if 'test' in self.args.mode:
            self.test()

    def train(self):
        logging.info("Running in train mode")
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            print(f"Epoch {epoch}")
            logging.info(f"Epoch {epoch}/{self.args.epochs}")

            # Training loop.
            self.audio_encoder.train()
            self.optimizer.zero_grad()

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pass

            
            # TODO: Training and validation logic here

    def test(self):
        logging.info("Running in test mode")
        # TODO: Add your testing logic here


def parse_arguments():
    parser = argparse.ArgumentParser(description='ML Experiment')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--model', type=str, default='GeneZC/MiniChat-2-3B', help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='annomi', help='Dataset to use for training')
    parser.add_argument('--task', type=str, default='classification', help='Task type (e.g., classification, forecasting)')
    parser.add_argument('--mode', type=str, nargs='+', choices=['train', 'test'], default=['train'], help='Mode to run the experiment (train and/or test)')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config/config_full.yaml', help='Path to the configuration file')
    parser.add_argument('--grad_accum_interval', type=int, default=16, help='Gradient accumulation interval')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint to resume training')
    parser.add_argument('--mode', type=str, choices=['speech', 'text'], required=True, help='Data mode to use (speech or text)')
    parser.add_argument('--max_audio_s', default=100, type=int, help='Maximum number of seconds to use for audio')
    args = parser.parse_args()
    return args

def main(args):
    trainer = MyTrainer(args)
    trainer.run_experiment()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
