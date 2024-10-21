import argparse
import os
import logging

from process_data import process # TODO
from log_writer import LogWriter 

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from omegaconf import OmegaConf

def parse_arguments():
    parser = argparse.ArgumentParser(description='ML Experiment')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--model', type=str, default='GeneZC/MiniChat-2-3B', help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='AnnoMI', help='Dataset to use for training')
    parser.add_argument('--task', type=str, default='classification', help='Task type (e.g., classification, forecasting)')
    parser.add_argument('--mode', type=str, nargs='+', choices=['train', 'test'], default=['train'], help='Mode to run the experiment (train and/or test)')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the run for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config/config_full.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    return args

def setup_logging(run_name):
    log_dir = os.path.join('experiment', run_name)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'experiment.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging setup complete.')

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logging.info(f"Random seed set to {seed}")

def experiment(args):
    setup_logging(args.run_name)
    set_seed(args.seed)

    config = OmegaConf.load(args.config)
    logwriter = LogWriter(config, os.path.join('experiment', args.run_name))
    
    logging.info(f"Running experiment with the following parameters:")
    logging.info(f"Learning Rate: {args.learning_rate}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Task: {args.task}")
    logging.info(f"Mode: {args.mode}")

    # (Step) prepare dataset
    # contains { "train": train, "dev": dev, "test": test}
    data = process(args)

    # (Step) load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if 'train' in args.mode:
        # Training logic here
        logging.info("Running in train mode")
        for epoch in range(args.epochs):
            logging.info(f"Epoch {epoch+1}/{args.epochs}")
            # Training and validation logic here
    
    if 'test' in args.mode:
        # Testing logic here
        logging.info("Running in test mode")
        # Add your testing logic here

def main(args):
    experiment(args)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
