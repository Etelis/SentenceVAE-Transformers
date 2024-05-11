import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PTB
from model import SentenceVAE
from utils import save_model, plot_metrics
from tqdm import tqdm
import numpy as np
import os

def setup_datasets_and_loaders(args):
    """Sets up the datasets and data loaders for training and validation.

    Args:
        args: Command line arguments including data directory, batch size, and max sequence length.

    Returns:
        Tuple containing dictionaries of datasets and dataloaders for 'train', 'valid', and optionally 'test'.
    """
    datasets = {'train': PTB(args.data_dir, 'train', args.max_seq_length),
                'valid': PTB(args.data_dir, 'valid', args.max_seq_length)}
    if args.test:
        datasets['test'] = PTB(args.data_dir, 'test', args.max_seq_length)

    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True if x == 'train' else False, num_workers=4, pin_memory=True) for x in datasets}
    return datasets, dataloaders

def setup_model(args, datasets):
    """Creates and initializes the SentenceVAE model.

    Args:
        args: Command line arguments with model configuration.
        datasets: Datasets to extract tokenizer and special token details.

    Returns:
        Initialized SentenceVAE model.
    """
    special_tokens = {
        'sos_token': datasets['train'].sos_token_id,
        'eos_token': datasets['train'].eos_token_id,
        'pad_token': datasets['train'].tokenizer.pad_token_id,
        'unk_token': datasets['train'].tokenizer.unk_token_id
    }
    vocab_size = datasets['train'].tokenizer.vocab_size
    model = SentenceVAE(
        latent_dim=args.latent_dim, 
        hidden_size=args.hidden_size, 
        vocab_size=vocab_size, 
        special_tokens=special_tokens, 
        max_seq_length=args.max_seq_length, 
        gru_layers=args.gru_layers
    )
    return model

def setup_loss_function(datasets):
    """Configures the loss function for training the SentenceVAE model.

    Args:
        datasets: Datasets to get tokenizer configurations for ignoring padding in loss calculations.

    Returns:
        Configured NLLLoss function.
    """
    return torch.nn.NLLLoss(ignore_index=datasets['train'].tokenizer.pad_token_id, reduction='sum')

def train(model, data_loader, optimizer, loss_fn, device):
    """Conducts a training epoch over the provided data loader.

    Args:
        model: The SentenceVAE model to train.
        data_loader: DataLoader for the training data.
        optimizer: Optimizer for the training.
        loss_fn: Loss function for the training.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple containing average loss, average NLL loss, and average KL loss for the epoch.
    """
    model.train()
    total_loss, total_nll_loss, total_kl_loss = 0, 0, 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target_ids']
        input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, mu, log_sigma_squared = model(input_ids, attention_mask)
        log_probs = torch.log_softmax(outputs, dim=-1).transpose(1, 2)
        nll_loss = loss_fn(log_probs, targets)
        kl_loss = -0.5 * torch.sum(1 + log_sigma_squared - mu.pow(2) - log_sigma_squared.exp()) / input_ids.size(0)
        loss = nll_loss + kl_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_nll_loss += nll_loss.item()
        total_kl_loss += kl_loss.item()

    num_batches = len(data_loader)
    return (total_loss / num_batches, total_nll_loss / num_batches, total_kl_loss / num_batches)

def validate(model, data_loader, loss_fn, device):
    """Validates the model on the provided data loader.

    Args:
        model: The SentenceVAE model to validate.
        data_loader: DataLoader for the validation data.
        loss_fn: Loss function for the validation.
        device: Device to run the validation on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple containing average loss, average NLL loss, and average KL loss for the validation.
    """
    model.eval()
    total_loss, total_nll_loss, total_kl_loss = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target_ids']
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)

            outputs, mu, log_sigma_squared = model(input_ids, attention_mask)
            log_probs = torch.log_softmax(outputs, dim=-1).transpose(1, 2)
            nll_loss = loss_fn(log_probs, targets)
            kl_loss = -0.5 * torch.sum(1 + log_sigma_squared - mu.pow(2) - log_sigma_squared.exp()) / input_ids.size(0)
            loss = nll_loss + kl_loss

            total_loss += loss.item()
            total_nll_loss += nll_loss.item()
            total_kl_loss += kl_loss.item()

    num_batches = len(data_loader)
    return (total_loss / num_batches, total_nll_loss / num_batches, total_kl_loss / num_batches)
