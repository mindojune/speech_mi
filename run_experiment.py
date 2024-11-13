import os
os.environ['HF_HOME'] = '/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/.cache'
import sys
import json
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

from process_data import process 
from log_writer import LogWriter 
from audio_encoder import AudioEncoder
from audio_llama import AudioLlamaForCausalLM
from utils import set_all_seeds, create_attention_mask, compute_num_audio_embeds, add_noise
import librosa
from peft import PeftConfig, PeftModel

# Import LoRA
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, classification_report

# import ROUGE, BLEU, BERTScore, METEOR
import evaluate

class MyTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.datatype = torch.float16 if args.datatype == 'float16' else torch.float32
        self.config = OmegaConf.load(args.config)
        self.setup_logging()
        self.logwriter = LogWriter(self.config, os.path.join(self.args.save_dir, f"{self.args.task}_experiment", self.args.run_name))
        # 'experiment', args.run_name))
        # Setup Model
        self.model = AudioLlamaForCausalLM.from_pretrained(
            args.model,
            use_cache=True,
            torch_dtype=self.datatype,
        ).eval()

        if args.lora_checkpoint_path:
            # lora_config = PeftConfig.from_pretrained(args.lora_checkpoint_path)
            # self.model = get_peft_model(self.model, lora_config)
            # peft_config = PeftConfig.from_pretrained(self.args.lora_checkpoint_path)
            # self.model.load_from_checkpoint(self.args.lora_checkpoint_path)
            self.model = PeftModel.from_pretrained(self.model, self.args.lora_checkpoint_path, is_trainable=True)
            logging.info(f"Loaded LoRA checkpoint from {self.args.lora_checkpoint_path}.\n")
            self.model.print_trainable_parameters()
            # logging.info("LoRA module loaded.")
        elif args.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                # target_modules=["q", "v"], # testing ablation
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logging.info("LoRA module added to the model.")
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        logging.info(f"Loaded model {args.model}.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
            padding_side="left",
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        
        if self.args.modality == "speech":
            self.audio_encoder = AudioEncoder(self.config)
            if "train" in self.args.mode:
                if self.args.audio_encoder_weight:
                    self.audio_encoder.load_state_dict(torch.load(self.args.audio_encoder_weight, map_location=self.device),strict=False)
                    logging.info(f"Loaded audio encoder from {self.args.audio_encoder_weight}.\n")
                if self.args.freeze_encoder:
                    for param in self.audio_encoder.parameters():
                        param.requires_grad = False
                    logging.info("Freezing the audio encoder.")
            self.audio_encoder.to(self.device)
        self.model.to(self.device)
        self.feature_extractor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")


        if self.args.use_lora:
            self.pad_token_embed = self.model.model.model.embed_tokens(torch.tensor([self.tokenizer.pad_token_id]).to(self.device)).squeeze(0)
        else:
            self.pad_token_embed = self.model.model.embed_tokens(torch.tensor([self.tokenizer.pad_token_id]).to(self.device)).squeeze(0)
        
        # Setup Dataloaders
        self.set_dataloaders()

        # Setup Training Parameters
        self.step = 0
        self.val_step = 0
        self.start_epoch = 0
        self.grad_accum_interval = args.grad_accum_interval
        self.num_epochs = self.args.epochs
        # Setup optimizer parameters
        optimizer_params = []
        if self.args.modality == "speech":
            optimizer_params.append({'params': self.audio_encoder.parameters()})
        optimizer_params.append({'params': self.model.parameters()})

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
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
        log_dir = os.path.join(self.args.save_dir, f"{self.args.task}_experiment", self.args.run_name)
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, 'experiment.log'), 
                            level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'),
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info('Logging setup complete.')

    def set_dataloaders(self):
        self.dataloaders = process(self.args)
        self.trainloader = self.dataloaders["train"]
        self.valloader = self.dataloaders["dev"]
        self.testloader = self.dataloaders["test"]

        # advance trainloader by 418 so that it starts from 419
        # self.trainloader = iter(self.trainloader)
        # for _ in range(418):
        #     next(self.trainloader)

        # log stats here
        logging.info(f"Trainloader length: {len(self.trainloader)}")
        logging.info(f"Valloader length: {len(self.valloader)}")
        logging.info(f"Testloader length: {len(self.testloader)}")


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if self.args.modality == "speech":
            self.audio_encoder.load_state_dict(checkpoint["audio_encoder"])
        
        if self.args.mode == "train":
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        if self.device == torch.device(f"cuda:{self.args.gpu_idx}"):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.args.gpu_idx)

        logging.info(f"Loaded checkpoint from {checkpoint_path}.\n")


    def prepare_batch(self, batch):
        encoding = self.tokenizer(batch["inputs"], return_tensors="pt", padding="longest", truncation=True, max_length=self.args.max_length)
        label_encoding = self.tokenizer(batch["labels"], return_tensors="pt", padding="longest", truncation=True, max_length=self.args.max_length)
        encoding["labels"] = label_encoding["input_ids"]
        # pad to -100
        encoding["labels"] = encoding["labels"].masked_fill(encoding["labels"] == self.tokenizer.pad_token_id, -100)
        encoding["label_mask"] = label_encoding["attention_mask"]

        
        if self.args.modality == "speech":
            #audio_paths = [x["audio_path"] for x in batch["audio_infos"]]
            audio_infos = batch["audio_infos"]
            
            audio_features = []
            for idx, audio_info in enumerate(audio_infos):
                fname = audio_info["audio_path"]
                begin = audio_info["begin"]
                end = audio_info["end"]

                if False:
                    # ~3s / 1 batch (bsz=2)
                    waveform, sample_rate = torchaudio.load(fname)
                    waveform = audresample.remix(waveform, mixdown=True).squeeze()
                    signal = audresample.resample(waveform, sample_rate, 16000)
                    audio = signal.squeeze()
                elif False:
                    # 
                    # audio, sr = librosa.load(fname, sr=16000, mono=True)
                    # audio = torch.tensor(audio).float().squeeze()
                    signal, source_rate = audiofile.read(fname)
                    # print(audiofile.duration(filename))
                    signal = audresample.resample(signal, source_rate, 16000)
                    signal = audresample.remix(
                        signal,
                        mixdown=True,
                    )
                    audio = signal.squeeze()
                elif False:
                    import scipy.io.wavfile as wav
                    sr, audio = wav.read(fname)
                    waveform = audresample.remix(audio, mixdown=True).squeeze()
                    signal = audresample.resample(waveform, sr, 16000)
                    audio = signal.squeeze()
                elif False:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(fname, format='wav')
                    waveform = np.array(audio.get_array_of_samples())
                    signal = audresample.resample(waveform, audio.frame_rate, 16000)
                    signal = audresample.remix(
                        signal,
                        mixdown=True,
                    )
                    audio = signal.squeeze()
                elif True:
                    signal, source_rate = audiofile.read(fname)
                    # skipping resampling bc it's already 16k
                    # signal = audresample.resample(signal, source_rate, 16000)
                    audio = signal.squeeze()


                else:
                    audio, sr = librosa.load(fname, sr=16000, mono=True)
                    audio = torch.tensor(audio).float().squeeze()

                # turn 'begin': '00:00:39', 'end': '00:00:41' into seconds
                begin_time = sum(x * int(t) for x, t in zip([3600, 60, 1], begin.split(":")))
                end_time = sum(x * int(t) for x, t in zip([3600, 60, 1], end.split(":")))

                audio = audio[begin_time * 16000:end_time * 16000]
                if self.args.max_audio_s:
                    audio = audio[:self.args.max_audio_s*16000]  
                
                if self.args.noise_level == -666:
                    # replace the audio with zero
                    audio = np.zeros_like(audio)
                elif self.args.noise_level >= 0.0:
                    # unsqueeze  np array
                    audio = audio[None, :]
                    audio = add_noise(audio, 16000, self.args.noise_level, self.device).squeeze()

                audio_feature = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding="longest", return_attention_mask=True )
                audio_features += [audio_feature]

            feat_lens = ([ len(x.input_values[0]) for x in audio_features])
            features = pad_sequence([torch.Tensor(x.input_values[0]) for x in audio_features], batch_first=True)

            audio_attention_mask = create_attention_mask(feat_lens)

            encoding["input_features"] = features 
            encoding["audio_attention_mask"] = audio_attention_mask 
    

        inputs = encoding.to(self.device)
        return inputs      


    def embed_audio_and_concatenate(self, batch, train=True):
        BSZ = batch["input_ids"].size(0)
        # print(batch["input_ids"])
        if self.args.use_lora:
            embedded_text = self.model.model.model.embed_tokens(batch["input_ids"])
        else:
            embedded_text= self.model.model.embed_tokens(batch["input_ids"])
        if self.args.modality == "text":
            if train:
                # concatenate the response to the context
                response_input_ids = batch["labels"]
                # map -100 to pad_token_id again (other wise decode error)
                response_input_ids = response_input_ids.masked_fill(response_input_ids == -100, self.tokenizer.pad_token_id)
                if self.args.use_lora:
                    response_embeds = self.model.model.model.embed_tokens(response_input_ids)
                else:
                    response_embeds = self.model.model.embed_tokens(response_input_ids)
                embedded_text = torch.cat([embedded_text, response_embeds], dim=1)
                label_mask = batch["label_mask"]
                attention_mask = torch.cat([batch["attention_mask"],  label_mask], dim=1)
                return embedded_text, attention_mask
            else:
                attention_mask = batch["attention_mask"]
                return embedded_text, attention_mask
        
        padded_audios = batch["input_features"]
        padded_audios = padded_audios.to(self.device)



        padded_audio_embeds = self.audio_encoder(padded_audios, batch["audio_attention_mask"], None)
        unpadded_audio_embeds = padded_audio_embeds

        if False:
            # print the length of input_ids and input_features
            print("*"*30)
            print("inside embed_audio_and_concatenate")
            print("input_ids:", batch["input_ids"].size())
            print("input_features:", unpadded_audio_embeds.size())
            

        if self.args.use_audio_eos:
            num_audio_embeds = [compute_num_audio_embeds(
                len_audio, sr=16000
                ) + 1 for len_audio in torch.sum(batch["audio_attention_mask"], dim=-1).tolist() ]
            unpadded_audio_embeds = pad_sequence([torch.cat([torch.Tensor(x[:y-1, :]), self.audio_encoder.eos_embedding],dim=0) for x,y in zip(unpadded_audio_embeds, num_audio_embeds)], batch_first=True).to(unpadded_audio_embeds.device)
        else:
            num_audio_embeds = [compute_num_audio_embeds(
                len_audio, sr=16000
                ) for len_audio in torch.sum(batch["audio_attention_mask"], dim=-1).tolist() ]
            
        audio_mask = create_attention_mask(num_audio_embeds).to(unpadded_audio_embeds.device)
        
        # BSZ = self.args.batch_size
        text_lengths = batch["attention_mask"].sum(dim=1)
        audio_lengths = audio_mask.sum(dim=1)
 
        # Calculate the total length after concatenation
        total_lengths = text_lengths + audio_lengths

        # Create a new tensor to hold the concatenated sequences
        max_length = total_lengths.max().item()
        max_length = int(max_length)

        # embedded_text_padded = torch.zeros((BSZ, max_length, embedded_text.size(-1)), dtype=torch.float).to(self.device)
        # init embedded_text_padded with self.pad_token_embed along the sequence dimension
        embedded_text_padded = self.pad_token_embed.expand(BSZ, max_length, -1).clone()
 

        attention_mask_padded = torch.zeros((BSZ, max_length), dtype=torch.long).to(self.device)

        if False:
            # Fill in the text and audio sequences
            for i in range(BSZ):
                text_len = int(text_lengths[i].item())
                audio_len = int(audio_lengths[i].item())

                embedded_text_padded[i, :text_len] = embedded_text[i, :text_len]
                embedded_text_padded[i, text_len:text_len + audio_len] = unpadded_audio_embeds[i, :audio_len]

                attention_mask_padded[i, :text_len] = batch["attention_mask"][i, :text_len]
                attention_mask_padded[i, text_len:text_len + audio_len] = audio_mask[i, :audio_len]

        elif True:
            # Rearrange embedded_text_padded and attention_mask_padded so that they are left padded
            for i in range(BSZ):
                text_len = int(text_lengths[i].item())
                audio_len = int(audio_lengths[i].item())
                total_len = text_len + audio_len
                start_pos = max_length - total_len

                # Shift the text embeddings and audio embeddings to the right
                embedded_text_padded[i, start_pos:start_pos + text_len] = embedded_text[i, -text_len:]
                embedded_text_padded[i, start_pos + text_len:start_pos + total_len] = unpadded_audio_embeds[i, :audio_len]

                # Shift the attention masks to the right
                attention_mask_padded[i, start_pos:start_pos + text_len] = batch["attention_mask"][i, -text_len:]
                attention_mask_padded[i, start_pos + text_len:start_pos + total_len] = audio_mask[i, :audio_len]

        if train:
            # concatenate the response to the context
            response_input_ids = batch["labels"]
            response_input_ids = response_input_ids.masked_fill(response_input_ids == -100, self.tokenizer.pad_token_id)
            if self.args.use_lora:
                response_embeds = self.model.model.model.embed_tokens(response_input_ids)
            else:
                response_embeds = self.model.model.embed_tokens(response_input_ids)
            embedded_text_padded = torch.cat([embedded_text_padded, response_embeds], dim=1)
            label_mask = batch["label_mask"]
            attention_mask_padded = torch.cat([attention_mask_padded,  label_mask], dim=1)
            return embedded_text_padded, attention_mask_padded
        else:
            return embedded_text_padded, attention_mask_padded
    
    def run_experiment(self):
        
        # self.set_seed()
        set_all_seeds(self.args.seed)

        logging.info(f"Running experiment with the following parameters:")
        logging.info(f"Learning Rate: {self.args.learning_rate}")
        logging.info(f"Optimizer Beta1: {self.args.optimizer_beta1}")
        logging.info(f"Optimizer Beta2: {self.args.optimizer_beta2}")
        logging.info(f"Batch Size: {self.args.batch_size}")
        logging.info(f"Test Batch Size: {self.args.test_batch_size}")
        logging.info(f"Epochs: {self.args.epochs}")
        logging.info(f"Validation Interval: {self.args.validation_interval}")
        logging.info(f"Log Interval: {self.args.log_interval}")
        logging.info(f"Model: {self.args.model}")
        logging.info(f"Dataset: {self.args.dataset}")
        logging.info(f"Data Length: {self.args.data_length}")
        logging.info(f"Task: {self.args.task}")
        logging.info(f"Mode: {self.args.mode}")
        logging.info(f"Save Directory: {self.args.save_dir}")
        logging.info(f"Run Name: {self.args.run_name}")
        logging.info(f"Seed: {self.args.seed}")
        logging.info(f"Config: {self.args.config}")
        logging.info(f"Gradient Accumulation Interval: {self.args.grad_accum_interval}")
        logging.info(f"Checkpoint Path: {self.args.checkpoint_path}")
        logging.info(f"Modality: {self.args.modality}")
        logging.info(f"Max Audio Seconds: {self.args.max_audio_s}")
        logging.info(f"Data Type: {self.args.datatype}")
        logging.info(f"Use LoRA: {self.args.use_lora}")
        logging.info(f"LoRA Checkpoint Path: {self.args.lora_checkpoint_path}")
        logging.info(f"Max Length: {self.args.max_length}")
        logging.info(f"Max New Tokens: {self.args.max_new_tokens}")
        logging.info(f"Audio Encoder Weight: {self.args.audio_encoder_weight}")
        logging.info(f"Freeze Encoder: {self.args.freeze_encoder}")
        logging.info(f"Use Audio EOS: {self.args.use_audio_eos}")
        logging.info(f"Noise Level (SNR): {self.args.noise_level}")
        logging.info(f"Use Audio EOS: {self.args.use_audio_eos}")
        logging.info(f"Only use hq sessions: {self.args.only_hq_sessions}")
        logging.info(f"Use handcrafted speech features: {self.args.use_handcrafted_speech_features}")

        if 'train' in self.args.mode:
            self.train()

        if 'test' in self.args.mode:
            self.test()

    def train(self):
        logging.info("Running in train mode")
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):

            logging.info(f"Epoch {epoch+1}/{self.args.epochs}")

            # Training loop.
            if self.args.modality == "speech":
                self.audio_encoder.train()
            self.optimizer.zero_grad()

            with tqdm(self.trainloader, unit="batch") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    with torch.autocast(device_type='cuda', dtype=self.datatype):
                        inputs = self.prepare_batch(batch)
                        input_embeds, attention_mask = self.embed_audio_and_concatenate(inputs)
                        response_input_ids = inputs["labels"]#.to(self.device)
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

                    if self.step == self.args.steps:
                        self.validate(epoch)
                        return
                    # Perform validation at interval.
                    if self.step % self.args.validation_interval == 0:
                        self.validate(epoch)

            

    def validate(self, epoch):
        if self.args.modality == "speech":
            self.audio_encoder.eval()
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Validation loop
        with tqdm(self.valloader, unit="batch") as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=self.datatype):
                        inputs = self.prepare_batch(batch)
                        
                        input_embeds, attention_mask = self.embed_audio_and_concatenate(inputs)
                        response_input_ids = inputs["labels"]#.to(self.device)

                        output = self.model(
                            inputs_embeds=input_embeds,
                            labels=response_input_ids,
                            output_hidden_states=True,
                            attention_mask=attention_mask,
                        )
                        loss = output.loss
                        total_loss += loss.item()
                        num_batches += 1
                        vepoch.set_postfix(loss=loss.item())

                self.val_step += 1
                self.logwriter.log_validation({"loss": loss}, self.val_step)
                # break
        avg_loss = total_loss / num_batches
        logging.info(f"Validation loss after epoch {epoch}: {avg_loss}")

        # save_path = os.path.join("experiment", self.args.run_name, f"checkpoint_epoch_{epoch}_step_{self.step}.pt")
        save_path = os.path.join(self.args.save_dir, f"{self.args.task}_experiment", self.args.run_name, f"checkpoint_epoch_{epoch}_step_{self.step}.pt")
        torch.save(
            {
                "audio_encoder": self.audio_encoder.state_dict() if self.args.modality == "speech" else None,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "epoch": epoch,
                "step": self.step,
            },
            save_path,
        )
        logging.info(f"Saved checkpoint for epoch {epoch} to {save_path}.\n")
        # peft save
        if self.args.use_lora:
            lora_save_path = os.path.join(self.args.save_dir, f"{self.args.task}_experiment", self.args.run_name, f"lora_checkpoint_epoch_{epoch}_step_{self.step}")
            #self.model.save_pretrained(os.path.join("experiment", self.args.run_name, f"lora_checkpoint_epoch_{epoch}_step_{self.step}"))
            self.model.save_pretrained(lora_save_path)
            logging.info(f"Saved LoRA checkpoint for epoch {epoch} to {lora_save_path}.\n")
        
        


    def test(self):
        logging.info("Running in test mode")
        if self.args.modality == "speech":
            self.audio_encoder.eval()
        self.model.eval()
        total_loss = 0
        num_batches = 0
        generated_texts = []

        # Validation loop
        with tqdm(self.testloader, unit="batch") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=self.datatype):
                        interlocutors = [x["interlocutor"] for x in batch["audio_infos"]]
                        inputs = self.prepare_batch(batch)
                        input_embeds, attention_mask = self.embed_audio_and_concatenate(inputs, train=False)
                        response_input_ids = inputs["labels"] #.to(self.device)\
                        # map -100 to pad_token_id again (other wise decode error)
                        response_input_ids = response_input_ids.masked_fill(response_input_ids == -100, self.tokenizer.pad_token_id)

                        generation_output = self.model.generate(
                            input_ids=None,
                            inputs_embeds=input_embeds,
                            max_new_tokens=self.args.max_new_tokens,
                            attention_mask=attention_mask,
                            use_cache=True,
                            # past_key_values=None,
                            # max_length=512,
                            # num_beams=5,
                            # early_stopping=True
                        )

                        prompt = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        #context = [self.tokenizer.decode(x[:input_embeds.size(1)], skip_special_tokens=True, clean_up_tokenization_spaces=True) for x in generation_output]
                        # logits = [ x[input_embeds.size(1):] for x in generation_output]#
                        # decoded_output = self.tokenizer.batch_decode(logits, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        decoded_output = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        decoded_labels = self.tokenizer.batch_decode(response_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                        # decoded_output = [ x.split("\n")[0] for x in decoded_output]
                        processed = []
                        for do in decoded_output:
                            try:
                                processed.append(do.split("\n")[0])
                                # processed.append(do.split("[CLS]")[0].split("\n")[0])
                            except:
                                processed.append(do)
                        decoded_output = processed
                        decoded_labels = [ x.split("\n")[0] for x in decoded_labels]

                        generated_texts.extend(zip(prompt, decoded_output, decoded_labels, interlocutors))

                        if False:
                            loss = self.model(
                                inputs_embeds=input_embeds,
                                labels=response_input_ids,
                                output_hidden_states=True,
                                attention_mask=attention_mask,
                            ).loss
                            tepoch.set_postfix(loss=loss.item())
                            total_loss += loss.item()
                        num_batches += 1

        #avg_loss = total_loss / num_batches
        #logging.info(f"Test loss: {avg_loss}")
        logging.info(f"Generated texts and labels: {generated_texts}")
        # compute accuracy and log

        if self.args.task == "response_generation":
            bleu = evaluate.load("bleu")
            rouge = evaluate.load("rouge")
            meteor = evaluate.load("meteor")
            bertscore = evaluate.load("bertscore")

            results = []
            # for prompt, generated, label, interlocutor in generated_texts:
            # use tqdm
            for prompt, generated, label, interlocutor in tqdm(generated_texts):
                g = generated#.split("\n")[0]
                c = label#.split("\n")[0]

                # print("1")
                bleu_score = bleu.compute(predictions=[g], references=[[c]])["bleu"] if g != "" else 0 
                # print("2")
                tokenize = lambda x: x.split() # this is to suppress warning (stup)
                rouge_score = rouge.compute(predictions=[g], references=[c], tokenizer=tokenize)["rougeL"] 
                # print("3")
                meteor_score = meteor.compute(predictions=[g], references=[c])["meteor"] 
                # print("4")
                bertscore_score = bertscore.compute(predictions=[g], references=[c], lang="en")["f1"]
                res_dic = {
                    "prompt": prompt,
                    "generated": g,
                    "label": c,
                    "interlocutor": interlocutor,
                    "bleu": bleu_score,
                    "rouge": rouge_score,
                    "meteor": meteor_score,
                    "bertscore": bertscore_score
                }
                results.append(res_dic)

            # compute average scores
            avg_bleu = np.mean([res["bleu"] for res in results])
            avg_rouge = np.mean([res["rouge"] for res in results])
            avg_meteor = np.mean([res["meteor"] for res in results])
            avg_bertscore = np.mean([res["bertscore"] for res in results])

            logging.info(f"Average BLEU Score: {avg_bleu}")
            logging.info(f"Average ROUGE Score: {avg_rouge}")
            logging.info(f"Average METEOR Score: {avg_meteor}")
            logging.info(f"Average BERTScore: {avg_bertscore}")

            log_dir = os.path.join(self.args.save_dir, f"{self.args.task}_experiment", self.args.run_name, "test_results.json")
            # added formatted generated_texts
            formatted_generated_texts = []
            for prompt, generated, label, interlocutor in generated_texts:
                formatted_generated_texts.append({
                    "prompt": prompt,
                    "generated": generated,
                    "label": label,
                    "interlocutor": interlocutor
                })
            dic ={
                    "bleu": avg_bleu,
                    "rouge": avg_rouge,
                    "meteor": avg_meteor,
                    "bertscore": avg_bertscore
                }
            dic["generated_texts"] = formatted_generated_texts
            with open(log_dir, 'w') as f:
                json.dump(dic, f, indent=4)


        else:
            correct = 0.0
            total = 0.0
            correct_client = 0.0
            total_client = 0.0
            correct_therapist = 0.0
            total_therapist = 0.0

            for prompt, generated, label, interlocutor in generated_texts:
                if generated.strip() == label.strip():
                    correct += 1
                    if interlocutor == "client":
                        correct_client += 1
                    elif interlocutor == "therapist":
                        correct_therapist += 1
                if interlocutor == "client":
                    total_client += 1
                elif interlocutor == "therapist":
                    total_therapist += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            accuracy_client = correct_client / total_client if total_client > 0 else 0
            accuracy_therapist = correct_therapist / total_therapist if total_therapist > 0 else 0

            logging.info(f"Test accuracy: {accuracy}")
            logging.info(f"Test accuracy (client): {accuracy_client}")
            logging.info(f"Test accuracy (therapist): {accuracy_therapist}")


            # write code to compute macro & micro f1s, along with class-wise f1s
            # given generated_texts
            # Extract the true labels and predicted labels
            true_labels = [label.strip() for _, _, label, _ in generated_texts]
            predicted_labels = [generated.strip() for _, generated, _, _ in generated_texts]

            # Compute macro and micro F1 scores
            macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
            micro_f1 = f1_score(true_labels, predicted_labels, average='micro')

            logging.info(f"Macro F1 Score: {macro_f1}")
            logging.info(f"Micro F1 Score: {micro_f1}")

            # Compute class-wise F1 scores
            class_wise_f1 = classification_report(true_labels, predicted_labels, output_dict=True)
            logging.info(f"Class-wise F1 Scores: {class_wise_f1}")

            # Log the detailed classification report
            logging.info(f"Classification Report:\n{classification_report(true_labels, predicted_labels)}")

            # save all this into a json file
            # save_path = os.path.join("experiment", self.args.run_name, "test_results.json")
            log_dir = os.path.join(self.args.save_dir, f"{self.args.task}_experiment", self.args.run_name, "test_results.json")
            # added formatted generated_texts
            formatted_generated_texts = []
            for prompt, generated, label, interlocutor in generated_texts:
                formatted_generated_texts.append({
                    "prompt": prompt,
                    "generated": generated,
                    "label": label,
                    "interlocutor": interlocutor
                })
            dic ={
                    "accuracy": accuracy,
                    "accuracy_client": accuracy_client,
                    "accuracy_therapist": accuracy_therapist,
                    "macro_f1": macro_f1,
                    "micro_f1": micro_f1,
                    "class_wise_f1": class_wise_f1
                }
            dic["generated_texts"] = formatted_generated_texts
        with open(log_dir, 'w') as f:
            json.dump(dic, f, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description='ML Experiment')
    parser.add_argument('--save_dir', type=str, default='/scratch/mihalcea_owned_root/mihalcea_owned1/dojmin/speech_mi_logs/', help='Absolute path of the storage for checkpoints and logs')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index to use for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.999, help='Beta2 for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--steps', type=int, default=20000, help='Number of steps to train')
    parser.add_argument('--validation_interval', type=int, default=1000, help='Interval for validation')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging')
    parser.add_argument('--model', type=str, default='GeneZC/MiniChat-2-3B', help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='annomi', help='Dataset to use for training')
    parser.add_argument('--task', type=str, default='classification', help='Task type (e.g., classification, forecasting, response_generation)')
    parser.add_argument('--mode', type=str, nargs='+', choices=['train', 'test'], default=['train'], help='Mode to run the experiment (train and/or test)')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config/config_full.yaml', help='Path to the configuration file')
    parser.add_argument('--grad_accum_interval', type=int, default=16, help='Gradient accumulation interval')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint to resume training')
    parser.add_argument('--modality', type=str, choices=['speech', 'text'], required=True, help='Data mode to use (speech or text)')
    parser.add_argument('--max_audio_s', default=100, type=int, help='Maximum number of seconds to use for audio')
    parser.add_argument('--datatype', type=str, default='float16', help='Data type to use for training')
    parser.add_argument('--use_lora', action="store_true", help='Use LoRA for model adaptation')
    parser.add_argument('--lora_checkpoint_path', type=str, help='Path to the LoRA checkpoint')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum length of the input sequence')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum number of tokens to generate')
    parser.add_argument('--data_length', type=int, nargs=3, default=[-1, -1, -1], help='Data length for training, \
                        validation, and testing. -1 means use all data.')
    parser.add_argument('--audio_encoder_weight', type=str, default="./data/speech_llm_audio_encoder.pt",  help='Path to the audio encoder weight')
    parser.add_argument('--freeze_encoder', action='store_true', default=False, help='Freeze the encoder')
    parser.add_argument('--use_audio_eos', action='store_true', default=False, help='Use audio eos')
    parser.add_argument('--noise_level', type=float, default=-1, help='Noise level in Signal-to-Noise (SNR) Ratio for audio augmentation')
    parser.add_argument('--only_hq_sessions', action='store_true', default=False, help='Use only high quality sessions')
    parser.add_argument('--use_handcrafted_speech_features', action='store_true', default=False, help='Use handcrafted speech features')

    args = parser.parse_args()
    return args

def main(args):
    trainer = MyTrainer(args)
    trainer.run_experiment()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
