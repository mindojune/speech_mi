import os
os.environ['HF_HOME'] = '/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/.cache'
import argparse
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
        # Now the batch contains
        #  input_ids, attention_mask, input_features, audio_attention_mask
        # input_ids and attention_mask are for the text
        # input_features and audio_attention_mask are for the audio
        # Now, reshape the features so that they can be concatenated
        # (text, audio) along the sequence dimension
        # padding / truncation should be done on the left side.

        # Get the lengths of the text and audio sequences
        """
        text_lengths = encoding["attention_mask"].sum(dim=1)
        audio_lengths = encoding["audio_attention_mask"].sum(dim=1)

        # Calculate the total length after concatenation
        total_lengths = text_lengths + audio_lengths

        # Create a new tensor to hold the concatenated sequences
        max_length = total_lengths.max().item()
        input_ids_padded = torch.full((len(batch["inputs"]), max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask_padded = torch.zeros((len(batch["inputs"]), max_length), dtype=torch.long)

        # Fill in the text and audio sequences
        for i in range(len(batch["inputs"])):
            # NOTE: no special token between text and audio here
            text_len = text_lengths[i].item()
            audio_len = audio_lengths[i].item()

            input_ids_padded[i, :text_len] = encoding["input_ids"][i, :text_len]
            input_ids_padded[i, text_len:text_len + audio_len] = encoding["input_features"][i, :audio_len]

            attention_mask_padded[i, :text_len] = encoding["attention_mask"][i, :text_len]
            attention_mask_padded[i, text_len:text_len + audio_len] = encoding["audio_attention_mask"][i, :audio_len]

        # Update the encoding dictionary with the concatenated sequences
        encoding["input_ids"] = input_ids_padded
        encoding["attention_mask"] = attention_mask_padded
        """
        inputs = encoding.to(self.device)
        return inputs      


    def embed_audio_and_concatenate(self, batch):
        if self.args.mode == "text":
            return batch["input_ids"], batch["attention_mask"]
        padded_audios = batch["input_features"]
        padded_audios = padded_audios.to(self.device)

        # Compute audio embeddings using audio encoder.
        padded_audio_embeds = self.audio_encoder(padded_audios, batch["audio_attention_mask"], None)

        unpadded_audio_embeds = padded_audio_embeds

        num_audio_embeds = [compute_num_audio_embeds(
                len_audio, sr=16000
            ) + 1 for len_audio in torch.sum(batch["audio_attention_mask"], dim=-1).tolist() ]
        unpadded_audio_embeds = pad_sequence([torch.cat([torch.Tensor(x[:y-1, :]), self.audio_encoder.module.eos_embedding],dim=0) for x,y in zip(unpadded_audio_embeds, num_audio_embeds)], batch_first=True).to(unpadded_audio_embeds.device)
        audio_mask = create_attention_mask(num_audio_embeds).to(unpadded_audio_embeds.device)
    
        BSZ = self.args.batch_size
        text_lengths = batch["attention_mask"].sum(dim=1)
        audio_lengths = audio_mask.sum(dim=1)

        # Calculate the total length after concatenation
        total_lengths = text_lengths + audio_lengths

        # Create a new tensor to hold the concatenated sequences
        max_length = total_lengths.max().item()
        input_ids_padded = torch.full((BSZ, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask_padded = torch.zeros((BSZ, max_length), dtype=torch.long)

        # Fill in the text and audio sequences
        for i in range(BSZ):
            # NOTE: no special token between text and audio here
            text_len = text_lengths[i].item()
            audio_len = audio_lengths[i].item()

            input_ids_padded[i, :text_len] = batch["input_ids"][i, :text_len]
            input_ids_padded[i, text_len:text_len + audio_len] = unpadded_audio_embeds[i, :audio_len]

            attention_mask_padded[i, :text_len] = batch["attention_mask"][i, :text_len]
            attention_mask_padded[i, text_len:text_len + audio_len] = audio_mask[i, :audio_len]
        
        # to device
        input_ids_padded = input_ids_padded.to(self.device)
        attention_mask_padded = attention_mask_padded.to(self.device)

        return input_ids_padded, attention_mask_padded 
    
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
            # print(f"Epoch {epoch}")
            logging.info(f"Epoch {epoch}/{self.args.epochs}")

            # Training loop.
            self.audio_encoder.train()
            self.optimizer.zero_grad()

            with tqdm(self.trainloader, unit="batch") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        inputs = self.prepare_batch(batch)
                        input_ids, attention_mask = self.embed_audio_and_concatenate(inputs)
                        outputs = self.model(inputs_embeds=input_ids, attention_mask=attention_mask)
                        # todo COMPUTE LOSS... 

                        loss = TODO
                        tepoch.set_postfix(loss=loss.item())
                    # Normalize loss to account for gradient accumulation and do backward pass.
                    norm_loss /= self.grad_accum_interval
                    scaler.scale(norm_loss).backward()

                    # Weights update.
                    if (
                        ((batch_idx + 1) % self.grad_accum_interval == 0) or
                        (batch_idx + 1 == len(self.train_dataloader))
                    ):
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                    self.step += 1

                    # Logging.
                    if self.step % self.config.log.log_interval == 0:
                        self.writer.log_training({"loss":loss}, self.step)
                        self.writer.log_lr(self.lr_scheduler.get_last_lr()[0], self.step)

                    # Perform validation at interval.
                    if self.step % self.config.log.validation_interval == 0:
                        self.validate(epoch)
            # TODO: Training and validation logic here



    def validate(self, epoch):
        self.audio_encoder.eval()
        # Validation loop
        with tqdm(self.valloader, unit="batch") as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        inputs = self.prepare_batch(batch)
                        input_ids, attention_mask = self.embed_audio_and_concatenate(inputs)
                        outputs = self.model(inputs_embeds=input_ids, attention_mask=attention_mask)
                        # todo COMPUTE LOSS... 
                        loss = TODO
                        vepoch.set_postfix(loss=loss.item())

                self.val_step +=1
                self.writer.log_validation({"loss":loss}, self.val_step)
        save_path = os.path.join("experiment", self.args.run_name, f"checkpoint_{epoch}.pt")
        torch.save(
            {
                "audio_encoder": self.audio_encoder.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "step": self.step,
            },
            save_path,
        )
        # print(f"Saved checkpoint for epoch {epoch} to {save_path}.\n")
        logging.info(f"Saved checkpoint for epoch {epoch} to {save_path}.\n")

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
