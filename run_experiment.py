import os
os.environ['HF_HOME'] = '/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/.cache'
import argparse
import logging
from tqdm.auto import tqdm

from transformers import AutoProcessor, AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from omegaconf import OmegaConf

import random
import audiofile, audresample
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from process_data import process # TODO
from log_writer import LogWriter 
from audio_encoder import AudioEncoder
from audio_llama import AudioLlamaForCausalLM
from utils import set_all_seeds, create_attention_mask, compute_num_audio_embeds
import librosa

class MyTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datatype = torch.float16 if args.datatype == 'float16' else torch.float32
        self.config = OmegaConf.load(args.config)
        self.setup_logging()
        self.logwriter = LogWriter(self.config, os.path.join('experiment', args.run_name))

        # Setup Model
        # self.model = AutoModelForCausalLM.from_pretrained(args.model)
        self.model = AudioLlamaForCausalLM.from_pretrained(
            args.model,
            use_cache=True,
            torch_dtype=self.datatype,
        ).eval()

        for param in self.model.parameters():
            param.requires_grad = False
        logging.info(f"Loaded model {args.model}.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.audio_encoder = AudioEncoder(self.config)
        self.audio_encoder.to(self.device)
        self.model.to(self.device)
        self.feature_extractor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

        # Setup Dataloaders
        self.set_dataloaders()

        # Setup Training Parameters
        self.step = 0
        self.val_step = 0
        self.start_epoch = 0
        self.grad_accum_interval = args.grad_accum_interval
        self.num_epochs = self.args.epochs
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.audio_encoder.parameters()},
                {'params': self.model.parameters()}
            ],
            lr=self.args.learning_rate,
            betas=(self.args.optimizer_beta1, self.args.optimizer_beta2),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=(self.num_epochs * len(self.trainloader) // self.grad_accum_interval),
            power=1
        )
        if args.checkpoint_path:
            self.load_checkpoint(args.checkpoint_path)

    def setup_logging(self):
        log_dir = os.path.join('experiment', self.args.run_name)
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'experiment.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Logging setup complete.')

    def set_dataloaders(self):
        self.dataloaders = process(self.args)
        self.trainloader = self.dataloaders["train"]
        self.valloader = self.dataloaders["dev"]
        self.testloader = self.dataloaders["test"]

        # advance trainloader by 418 so that it starts from 419
        self.trainloader = iter(self.trainloader)
        for _ in range(418):
            next(self.trainloader)



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
        encoding = self.tokenizer(batch["inputs"], return_tensors="pt", padding="longest", truncation=True, max_length=512)
        label_encoding = self.tokenizer(batch["labels"], return_tensors="pt", padding="longest", truncation=True, max_length=512)
        encoding["labels"] = label_encoding["input_ids"]
        # pad to -100
        encoding["labels"] = encoding["labels"].masked_fill(encoding["labels"] == self.tokenizer.pad_token_id, -100)
        
        if self.args.modality == "speech":
            #audio_paths = [x["audio_path"] for x in batch["audio_infos"]]
            audio_infos = batch["audio_infos"]
            
            audio_features = []
            for idx, audio_info in enumerate(audio_infos):
                fname = audio_info["audio_path"]
                begin = audio_info["begin"]
                end = audio_info["end"]

                waveform, sample_rate = torchaudio.load(fname)
                waveform = audresample.remix(waveform, mixdown=True).squeeze()
                signal = audresample.resample(waveform, sample_rate, 16000)
                audio = signal.squeeze()

                # turn 'begin': '00:00:39', 'end': '00:00:41' into seconds
                begin_time = sum(x * int(t) for x, t in zip([3600, 60, 1], begin.split(":")))
                end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))

                audio = audio[begin_time * 16000:end_time * 16000]
                if self.args.max_audio_s:
                    audio = audio[:self.args.max_audio_s*16000]  

                audio_feature = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding="longest", return_attention_mask=True )
                audio_features += [audio_feature]

            feat_lens = ([ len(x.input_values[0]) for x in audio_features])
            features = pad_sequence([torch.Tensor(x.input_values[0]) for x in audio_features], batch_first=True)

            audio_attention_mask = create_attention_mask(feat_lens)

            encoding["input_features"] = features 
            encoding["audio_attention_mask"] = audio_attention_mask 
    

        inputs = encoding.to(self.device)
        return inputs      


    def embed_audio_and_concatenate(self, batch):
        embedded_text= self.model.model.embed_tokens(batch["input_ids"])
        if self.args.modality == "text":
            return embedded_text, batch["attention_mask"]
        padded_audios = batch["input_features"]
        padded_audios = padded_audios.to(self.device)

        
        padded_audio_embeds = self.audio_encoder(padded_audios, batch["audio_attention_mask"], None)

        unpadded_audio_embeds = padded_audio_embeds

        num_audio_embeds = [compute_num_audio_embeds(
            len_audio, sr=16000
            ) + 1 for len_audio in torch.sum(batch["audio_attention_mask"], dim=-1).tolist() ]
        unpadded_audio_embeds = pad_sequence([torch.cat([torch.Tensor(x[:y-1, :]), self.audio_encoder.eos_embedding],dim=0) for x,y in zip(unpadded_audio_embeds, num_audio_embeds)], batch_first=True).to(unpadded_audio_embeds.device)
        audio_mask = create_attention_mask(num_audio_embeds).to(unpadded_audio_embeds.device)
        
        BSZ = self.args.batch_size
        text_lengths = batch["attention_mask"].sum(dim=1)
        audio_lengths = audio_mask.sum(dim=1)

        # Calculate the total length after concatenation
        total_lengths = text_lengths + audio_lengths

        # Create a new tensor to hold the concatenated sequences
        max_length = total_lengths.max().item()
        max_length = int(max_length)

        embedded_text_padded = torch.zeros((BSZ, max_length, embedded_text.size(-1)), dtype=torch.float).to(self.device)

        attention_mask_padded = torch.zeros((BSZ, max_length), dtype=torch.long).to(self.device)

        # Fill in the text and audio sequences
        for i in range(BSZ):
            text_len = int(text_lengths[i].item())
            audio_len = int(audio_lengths[i].item())

            embedded_text_padded[i, :text_len] = embedded_text[i, :text_len]
            embedded_text_padded[i, text_len:text_len + audio_len] = unpadded_audio_embeds[i, :audio_len]

            attention_mask_padded[i, :text_len] = batch["attention_mask"][i, :text_len]
            attention_mask_padded[i, text_len:text_len + audio_len] = audio_mask[i, :audio_len]

        return embedded_text_padded, attention_mask_padded
    
    def run_experiment(self):
        
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
        logging.info(f"Modality: {self.args.modality}")


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
                    with torch.autocast(device_type='cuda', dtype=self.datatype):
                        inputs = self.prepare_batch(batch)
                        input_embeds, attention_mask = self.embed_audio_and_concatenate(inputs)
                        response_input_ids = inputs["labels"].to(self.device)

                        output = self.model(
                            inputs_embeds=input_embeds,
                            labels=response_input_ids,
                            output_hidden_states=True,
                            attention_mask=attention_mask,
                        )
                        loss = output.loss
                        tepoch.set_postfix(loss=loss.item())
                    # Normalize loss to account for gradient accumulation and do backward pass.
                    norm_loss = loss / self.grad_accum_interval
                    scaler.scale(norm_loss).backward()

                    # Weights update.
                    if (
                        ((batch_idx + 1) % self.grad_accum_interval == 0) or
                        (batch_idx + 1 == len(self.trainloader))
                    ):
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                    self.step += 1

                    # Logging.
                    if self.step % self.args.log_interval == 0:
                        self.logwriter.log_training({"loss":loss}, self.step)
                        self.logwriter.log_lr(self.lr_scheduler.get_last_lr()[0], self.step)

                    # Perform validation at interval.
                    if self.step % self.args.validation_interval == 0:
                        self.validate(epoch)
            # TODO: Training and validation logic here



    def validate(self, epoch):
        self.audio_encoder.eval()
        # Validation loop
        with tqdm(self.valloader, unit="batch") as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=self.datatype):
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
                "audio_encoder": self.audio_encoder.state_dict(),
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
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.999, help='Beta2 for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--validation_interval', type=int, default=1000, help='Interval for validation')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging')
    parser.add_argument('--model', type=str, default='GeneZC/MiniChat-2-3B', help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='annomi', help='Dataset to use for training')
    parser.add_argument('--task', type=str, default='classification', help='Task type (e.g., classification, forecasting)')
    parser.add_argument('--mode', type=str, nargs='+', choices=['train', 'test'], default=['train'], help='Mode to run the experiment (train and/or test)')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config/config_full.yaml', help='Path to the configuration file')
    parser.add_argument('--grad_accum_interval', type=int, default=16, help='Gradient accumulation interval')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint to resume training')
    parser.add_argument('--modality', type=str, choices=['speech', 'text'], required=True, help='Data mode to use (speech or text)')
    parser.add_argument('--max_audio_s', default=100, type=int, help='Maximum number of seconds to use for audio')
    parser.add_argument('--datatype', type=str, default='float16', help='Data type to use for training')
    args = parser.parse_args()
    return args

def main(args):
    trainer = MyTrainer(args)
    trainer.run_experiment()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
