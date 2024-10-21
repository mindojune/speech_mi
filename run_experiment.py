import argparse
import os
import logging

from process_data import process # TODO
from log_writer import LogWriter 
from audio_encoder import AudioEncoder

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from omegaconf import OmegaConf


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

    def set_seed(self):
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
        logging.info(f"Random seed set to {self.args.seed}")

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

    def run_experiment(self):
        self.setup_logging()
        self.set_seed()

        logging.info(f"Running experiment with the following parameters:")
        logging.info(f"Learning Rate: {self.args.learning_rate}")
        logging.info(f"Batch Size: {self.args.batch_size}")
        logging.info(f"Epochs: {self.args.epochs}")
        logging.info(f"Model: {self.args.model}")
        logging.info(f"Dataset: {self.args.dataset}")
        logging.info(f"Task: {self.args.task}")
        logging.info(f"Mode: {self.args.mode}")

        self.data = process(self.args)

        if 'train' in self.args.mode:
            self.train()

        if 'test' in self.args.mode:
            self.test()

    def train(self):
        logging.info("Running in train mode")
        for epoch in range(self.args.epochs):
            logging.info(f"Epoch {epoch+1}/{self.args.epochs}")
            # Training and validation logic here

    def test(self):
        logging.info("Running in test mode")
        # Add your testing logic here


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

    args = parser.parse_args()
    return args

def main(args):
    trainer = MyTrainer(args)
    trainer.run_experiment()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
